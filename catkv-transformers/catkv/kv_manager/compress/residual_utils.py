

import torch

def calculate_elbow(singular_values: torch.Tensor , max_x = None , min_s = None) -> int:
    """Find elbow point in singular values using distance-to-chord heuristic."""
    if singular_values.dim() == 1:
        singular_values = singular_values.unsqueeze(0)

    device = singular_values.device
    B, N = singular_values.shape

    x_coords = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)

    all_coords = torch.stack([x_coords.expand(B, N), singular_values], dim=-1)
    first_point = all_coords[:, 0, :]
    if max_x is not None:
      last_point = torch.tensor([max_x, min_s], device=device).unsqueeze(0)
    else:
      last_point = all_coords[:, -1, :]

    line_vec = last_point - first_point
    point_vec = all_coords - first_point.unsqueeze(1)

    cross_product = torch.abs(
        line_vec[:, 0:1] * point_vec[:, :, 1] - line_vec[:, 1:2] * point_vec[:, :, 0]
    )

    line_norm = torch.linalg.norm(line_vec, dim=1)

    small_norm_mask = line_norm < 1e-10
    line_norm = torch.where(small_norm_mask, torch.ones_like(line_norm), line_norm)

    distances = cross_product / line_norm.unsqueeze(1)

    _, indices = torch.max(distances, dim=1)
    mean_elbow_dim = int(indices.float().mean().item())

    return mean_elbow_dim

import torch

def calculate_elbow_geometric(singular_values: torch.Tensor) -> int:
    """
    Find the maximum-curvature elbow point with a geometric Kneedle heuristic.
    This avoids a preset threshold and picks a cost-effective rank k.
    
    Args:
        singular_values (torch.Tensor): Singular-value vector.
        
    Returns:
        int: Suggested retained rank k.
    """
    # Run on CPU to avoid unnecessary GPU synchronization overhead.
    y = singular_values.detach().cpu()
    x = torch.arange(len(y), dtype=torch.float32)
    
    # Normalize x and y to [0, 1] so distances share the same scale.
    # If singular values span a very large range, log(singular_values) can
    # also be used. This implementation normalizes the raw values directly.
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    
    # Build the chord from the first point to the last point.
    # The elbow is the point farthest from this line.
    start_point = torch.tensor([x_norm[0], y_norm[0]])
    end_point = torch.tensor([x_norm[-1], y_norm[-1]])
    line_vec = end_point - start_point
    
    # Compute each point's distance to the line with a 2D cross product.
    # Vector from start to each point
    vec_from_start = torch.stack([x_norm - start_point[0], y_norm - start_point[1]], dim=1)
    
    # Distance formula using cross product logic in 2D
    # distance = |cross_product| / |line_length|
    # cross_product = x1*y2 - x2*y1
    cross_prod = vec_from_start[:, 0] * line_vec[1] - vec_from_start[:, 1] * line_vec[0]
    distances = torch.abs(cross_prod) / torch.norm(line_vec)
    
    # Select the index with the maximum distance.
    elbow_index = torch.argmax(distances).item()
    
    return elbow_index + 1
