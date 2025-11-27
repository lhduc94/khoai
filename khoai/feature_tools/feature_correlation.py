import numpy as np
import pandas as pd
import networkx as nx
def calculate_correlation_distance_blockwise(
    X: np.ndarray,
    Y: np.ndarray,
    block_size: int = 100,
    eps: float = 1e-12
):
    """
    Tính ma trận khoảng cách 1 - |corr(x_i, center_j)| nhưng:
    - Không tạo Xn, Cn kích thước (N, D) và (K, D)
    - Tính dot product theo từng block cột để giảm peak RAM

    X:      (N, D)
    centers:(K, D)

    Trả về:
        dists: (N, K)
    """
    X = np.asarray(X)
    centers = np.asarray(Y)

    N, D = X.shape
    K, D2 = centers.shape
    assert D == D2, "X và centers phải có cùng số chiều D"

    # -- Mean & std cho từng vector --
    X_mean = X.mean(axis=1, keepdims=True)        # (N, 1)
    X_std  = X.std(axis=1, keepdims=True)
    X_std  = np.maximum(X_std, eps)

    Y_mean = Y.mean(axis=1, keepdims=True)  # (K, 1)
    Y_std  = Y.std(axis=1, keepdims=True)
    Y_std  = np.maximum(Y_std, eps)

    # -- Tích lũy dot = sum_d x_id * c_jd --
    # num sẽ là (N, K)
    num = np.zeros((N, K), dtype=np.float64)

    for start in range(0, D, block_size):
        end = min(start + block_size, D)

        X_block = X[:, start:end]          # (N, B)
        Y_block = Y[:, start:end]    # (K, B)

        # X_block @ C_block.T -> (N, K), cộng dần
        num += X_block @ Y_block.T
    # -- Chuyển dot product thành corr --
    # (N, 1) * (1, K) broadcast -> (N, K)
    mean_prod = X_mean @ Y_mean.T         # (N, K)
    std_prod  = X_std  @ Y_std.T          # (N, K)

    corr = (num - D * mean_prod) / ((D - 1) * std_prod)

    return corr

def feature_correlation_matrix(df: pd.DataFrame):
    corr_matrix = calculate_correlation_distance_blockwise(np.array(df.transpose(), dtype=np.float32), np.array(df.transpose(), dtype=np.float32))
    return corr_matrix

def _feature_correlation_community_detection(corr_df: pd.DataFrame, seed: int = 123):
    edges_df = (
    corr_df
    .reset_index()
    .melt(id_vars='index')
    .rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'corr'})
)
    edges_df['abs_corr'] = abs(edges_df['corr'])
    edges = list(edges_df[['var1', 'var2', 'abs_corr']].itertuples(index=False, name=None))

    g = nx.Graph()
    g.add_weighted_edges_from(edges)
    clusters = nx.community.louvain_communities(g, seed=seed)
    return clusters

def _feature_correlation_connected_components(corr_df: pd.DataFrame, cutoff: float = 0.5):
    edges_df = (
    corr_df
    .reset_index()
    .melt(id_vars='index')
    .rename(columns={'index': 'var1', 'variable': 'var2', 'value': 'corr'})
)
    edges_df['abs_corr'] = abs(edges_df['corr'])
    cutoff_edges = list(edges_df[edges_df['abs_corr'] >= cutoff][['var1', 'var2', 'abs_corr']].itertuples(index=False, name=None))  
    g = nx.Graph()
    g.add_weighted_edges_from(cutoff_edges)
    connected_components = list(nx.connected_components(g))
    return connected_components

def feature_correlation_community_detection(df: pd.DataFrame):
    corr_matrix = feature_correlation_matrix(df)
    corr_df = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
    clusters = _feature_correlation_community_detection(corr_df)
    return clusters

def feature_correlation_connected_components(df: pd.DataFrame, cutoff: float = 0.5):
    corr_matrix = feature_correlation_matrix(df)
    corr_df = pd.DataFrame(corr_matrix, index=df.columns, columns=df.columns)
    connected_components = _feature_correlation_connected_components(corr_df, cutoff=cutoff)
    return connected_components

def feature_correlation_groups(df: pd.DataFrame, method: str = 'community_detection', cutoff: float = 0.5):
    if method == 'community_detection':
        return feature_correlation_community_detection(df)
    elif method == 'connected_components':
        return feature_correlation_connected_components(df, cutoff=cutoff)
    else:
        raise ValueError(f"Invalid method: {method}")
