import torch

def mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 32, eps: float = 1e-10) -> float:
    assert x.shape[0] == y.shape[0], "Mismatched number of samples"
    x = x.squeeze(0)
    y = y.squeeze(0)
    N, D = x.shape
    mi_list = []
    for i in range(D):
        x_i = x[:, i]
        y_i = y[:, i]
        x_i = (x_i - x_i.min()) / (x_i.max() - x_i.min() + eps)
        y_i = (y_i - y_i.min()) / (y_i.max() - y_i.min() + eps)
        x_bin = torch.clamp((x_i * bins).long(), 0, bins - 1)
        y_bin = torch.clamp((y_i * bins).long(), 0, bins - 1)
        flat = x_bin * bins + y_bin
        joint = torch.bincount(flat, minlength=bins * bins).reshape(bins, bins).float()
        joint = joint / joint.sum()
        x_marg = joint.sum(dim=1, keepdim=True)
        y_marg = joint.sum(dim=0, keepdim=True)
        pxpy = x_marg @ y_marg
        mask = joint > 0
        mi = joint[mask] * torch.log(joint[mask] / (pxpy[mask] + eps) + eps)
        mi_list.append(mi.sum())
    return torch.stack(mi_list).mean().item()

def linear_kernel(X, Y):
    return X @ Y.T

def center_gram_matrix(K):
    """
    This is equivalent to H @ K @ H, where H = I - 1/n.
    
    Args:
        K (torch.Tensor): Gram matrix, shape [n, n].

    Returns:
        torch.Tensor: Centered Gram matrix.
    """
    mean_rows = K.mean(dim=1, keepdim=True)
    mean_cols = K.mean(dim=0, keepdim=True)
    mean_all = K.mean()
    return K - mean_rows - mean_cols + mean_all

def cka(X, Y, kernel=linear_kernel):
    """
    Compute the centered kernel alignment (CKA) score between X and Y.

    Args:
        X (torch.Tensor): First representation matrix with shape
                          [n_samples, n_features_x]. Rows are samples and
                          columns are features.
        Y (torch.Tensor): Second representation matrix with shape
                          [n_samples, n_features_y]. It must have the same
                          number of samples as X.
        kernel (function): Kernel function used to compute Gram matrices.
                           Defaults to the linear kernel.

    Returns:
        torch.Tensor: CKA score, a scalar in [0, 1].
    """
    # 1. Compute Gram matrices.
    # K is the sample-similarity matrix for X; L is the same for Y.
    K = kernel(X, X)
    L = kernel(Y, Y)

    # 2. Center the Gram matrices.
    K_c = center_gram_matrix(K)
    L_c = center_gram_matrix(L)

    # 3. Compute the CKA score.
    # HSIC(K, L) is efficiently computed as the Frobenius inner product of
    # centered Gram matrices.
    hsic_numerator = torch.sum(K_c * L_c)

    # The denominator is the product of the square roots of each self-HSIC.
    hsic_denominator_k = torch.sqrt(torch.sum(K_c * K_c))
    hsic_denominator_l = torch.sqrt(torch.sum(L_c * L_c))

    # Avoid division by zero.
    if hsic_denominator_k == 0 or hsic_denominator_l == 0:
        return torch.tensor(0.0, device=X.device)

    cka_score = hsic_numerator / (hsic_denominator_k * hsic_denominator_l)

    return cka_score

def get_centered_gram(tensor, token_size):
    reshaped_tensor = tensor.reshape(token_size, -1)
    gram_matrix = linear_kernel(reshaped_tensor, reshaped_tensor)
    return center_gram_matrix(gram_matrix)

def batch_cka(data):
    scores = []
    layer_num = data.size(0)
    sequence_num = data[0].size(0) 

    cg_previous_key = get_centered_gram(data[0], sequence_num)
    norm_previous_key = torch.linalg.norm(cg_previous_key, ord='fro')

    for layer_idx in range(1, layer_num):
        cg_current_key = get_centered_gram(data[layer_idx], sequence_num)
        norm_current_key = torch.linalg.norm(cg_current_key, ord='fro')
        
        hsic_numerator = torch.sum(cg_current_key * cg_previous_key)
        hsic_denominator = norm_current_key * norm_previous_key
        
        score = hsic_numerator / (hsic_denominator + 1e-8)
        scores.append(1 - score.item())
        
        cg_previous_key = cg_current_key
        norm_previous_key = norm_current_key
    return scores

def relative_outline(X, Y):
    return (torch.norm(X - Y, p=2, dim=-1) / torch.norm(Y, p=2, dim=-1)).mean().item()
