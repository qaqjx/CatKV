

import torch


class AllocateStrategy:
    UNIFORM = 'uniform'
    STATIC = 'static'
    DYNAMIC = 'dynamic'


class AllocateAlgorithm:
    def __init__(self, strategy: str, **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs

    def _calculate_ranks(self, total_budget_bytes: int, num_layers: int, dim: int, token: int) -> int:
        budget_per_rank = 4 * 16 + (token + dim) * 4  # 4 bytes per float
        return total_budget_bytes * 8 // budget_per_rank

    def _calculate_rank_from_memory(self, memory_bytes: int, token: int, dim: int) -> int:
        """Calculates the max possible rank for a given memory budget in bytes."""
        return memory_bytes * 8 // (token * 4 + 2 * 16 + dim * 4 + 2 * 16)

    def _calculate_memory_bytes(self, rank, token, dim) -> int:
        """Calculates memory in bytes for a single KV cache (Key or Value)."""
        return rank * (token * 4 + 2 * 16 + dim * 4 + 2 * 16) // 8

    def allocate(
        self,
        total_budget_bytes: int,
        num_layers: int,
        dim: int,
        token_num: int,
        score: list = None, max_rank: int = 0
    ) -> list:
        pass


def allocate_low_rank(total_budget_bytes: int, dim: int, token_num: int , high_rank: int = 512) -> int:
    budget_per_rank = 4 * 16 + (token_num + dim) * 4

    low_rank = (total_budget_bytes * 8  - (high_rank * token_num * 4 + high_rank * 2 * 16 + high_rank * dim * 16)) // budget_per_rank

    if low_rank < 0:
        low_rank = 0
        high_rank = total_budget_bytes * 8 // (token_num * 4 + 2 * 16 + dim * 16)

    return low_rank , high_rank
     

class UniformAllocate(AllocateAlgorithm):
    def __init__(self):
        super().__init__(AllocateStrategy.UNIFORM)

    def allocate(self, total_budget_bytes: int, num_layers: int, dim: int, token_num: int,
                 score: list = None, max_rank: int = 0) -> list:
        all_rank = self._calculate_ranks(total_budget_bytes, num_layers, dim, token_num)
        
        rank_per_layer = int(all_rank // num_layers)
        rank = rank_per_layer * num_layers
        
        if token_num >= 8192:
            max_rank = 256 
        elif token_num >= 4096:
            max_rank = 384
        else:
            max_rank = 512

        return  min(max_rank, rank, 1024, token_num)

class StaticAllocate(AllocateAlgorithm):
    def __init__(self):
        super().__init__(AllocateStrategy.STATIC)

    def allocate(self, total_budget_bytes: int, num_layers: int, dim: int, token_num: int,
                 score: list = None, max_rank: int = 0) -> list:
        all_rank = self._calculate_ranks(total_budget_bytes, num_layers, dim, token_num)

        # Similar to CacheGen, split the layers into three parts and allocate ranks in a 3:2:1 ratio
        rank_per_layer = int(all_rank // num_layers)
        ranks = [rank_per_layer] * num_layers

        for i in range(num_layers // 3):
            ranks[i] = int(ranks[i] * 1.5)
            ranks[-(i+1)] = int(ranks[-(i+1)] * 0.5)

        return ranks
    
class DynamicAllocate(AllocateAlgorithm):
    def __init__(self):
        super().__init__(AllocateStrategy.DYNAMIC)

    def allocate(self, total_budget_bytes: int, num_layers: int, dim: int, token_num: int,
                 score , max_rank: int = 0) -> list:
        """
        Calculates adaptive ranks by directly manipulating memory budgets to avoid precision loss from
        repeated float-to-int conversions, which caused the rank reduction bug.

        The logic is as follows:
        1. Allocate initial budgets proportionally.
        2. In a loop, identify any layers whose budget would result in a rank > max_rank.
        3. For these "overbudgeted" layers:
            a. Fix their rank to max_rank.
            b. Calculate the budget they actually need for max_rank.
            c. The difference between their allocated budget and needed budget is the "surplus".
            d. Mark these layers as "finalized" so they are excluded from future steps.
        4. Redistribute the total surplus budget among the remaining "unfinalized" layers.
        5. Repeat until no layers are overbudgeted.
        6. Finally, calculate the ranks for all unfinalized layers from their final budget allocation.
        """
        device = score[0].device if isinstance(score, list) and hasattr(score[0], 'device') else "cpu"
        
        score_tensor = torch.tensor(score, dtype=torch.float32, device=device)
        max_ranks_tensor = torch.tensor([max_rank] * num_layers, dtype=torch.int32, device=device)
        
        is_finalized = torch.zeros(num_layers, dtype=torch.bool, device=device)
        final_ranks = torch.zeros(num_layers, dtype=torch.int32, device=device)
        current_budgets = total_budget_bytes * (score_tensor / torch.sum(score_tensor))       

        for _ in range(num_layers):
            unfinalized_indices = torch.where(~is_finalized)[0]
            if len(unfinalized_indices) == 0:
                break 

            # Calculate tentative ranks ONLY for the unfinalized layers to check against their max
            # This is the only place we convert budget-to-rank inside the loop
            tentative_ranks_unfinalized = torch.tensor([
                self._calculate_rank_from_memory(layer_budget_byte.item(), token_num, dim)
                for layer_budget_byte in current_budgets[unfinalized_indices]
            ], dtype=torch.int32, device=device)

            max_ranks_unfinalized = max_ranks_tensor[unfinalized_indices]
            
            # Identify which of the active layers are overbudgeted
            overbudgeted_mask_unfinalized = tentative_ranks_unfinalized > max_ranks_unfinalized
            
            if not torch.any(overbudgeted_mask_unfinalized):
                break

            # Get the global indices of layers to finalize in this iteration
            indices_to_finalize = unfinalized_indices[overbudgeted_mask_unfinalized]

            # Calculate the total surplus budget from all layers being finalized now
            surplus_budget = 0.0
            for idx in indices_to_finalize:
                final_ranks[idx] = max_ranks_tensor[idx]
                is_finalized[idx] = True

                consumed_budget = self._calculate_memory_bytes(final_ranks[idx].item(), token_num, dim)
                allocated_budget = current_budgets[idx].item()
                
                surplus_budget += (allocated_budget - consumed_budget)
                current_budgets[idx] = 0 # This budget is now handled, zero it out from the pool

            # Redistribute the collective surplus to the *new* set of unfinalized layers
            unfinalized_indices = torch.where(~is_finalized)[0]
            if len(unfinalized_indices) == 0 or surplus_budget < 1e-6:
                break

            remaining_scores = score_tensor[unfinalized_indices]
            total_remaining_score = torch.sum(remaining_scores)

            budget_additions = surplus_budget * (remaining_scores / total_remaining_score)
            
            current_budgets[unfinalized_indices] += budget_additions

        unfinalized_indices = torch.where(~is_finalized)[0]
        if len(unfinalized_indices) > 0:
            # Calculate their ranks from their final budget allocation
            final_ranks_unfinalized = torch.tensor([
                self._calculate_rank_from_memory(layer_budget_bytes.item(), token_num, dim)
                for layer_budget_bytes in current_budgets[unfinalized_indices]
            ], dtype=torch.int32, device=device)
            
            # As a safeguard, clip against their max rank
            final_ranks_unfinalized = torch.minimum(final_ranks_unfinalized, max_ranks_tensor[unfinalized_indices])
            final_ranks[unfinalized_indices] = final_ranks_unfinalized

        return [int(r.item()) for r in final_ranks]
    
