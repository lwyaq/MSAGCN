import os
import h5py
import numpy as np
import sklearn
import torch
import pandas as pd
import scanpy as sc
from MSAGCN import MSAGCN_pyG
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
from sklearn.neighbors import NearestNeighbors

# Environment configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.2'

# Specify data type
data_type = 'Spatial-epigenome-transcriptome'

# Fix random seed
from MSAGCN.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)

# Read data
file_fold = 'data/mousebrain/'
adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_peaks_normalized.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()
from MSAGCN.preprocess import clr_normalize_each_cell, pca

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.filter_cells(adata_omics1, min_genes=200)

sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

adata_omics2 = adata_omics2[adata_omics1.obs_names].copy() # .obsm['X_lsi'] represents the dimension reduced feature
if 'X_lsi' not in adata_omics2.obsm.keys():
    sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
    lsi(adata_omics2, use_highly_variable=False, n_components=50)

adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

from MSAGCN.preprocess import construct_neighbor_graph
# 使用增强的图构建功能
data = construct_neighbor_graph(
    adata_omics1, adata_omics2,
    datatype=data_type,
    use_multiscale=True,  # 启用多尺度空间感知
    spatial_scales=None   # 按datatype固定尺度
)

from MSAGCN.MSAGCN_pyG import Train_MSAGCN
# 使用增强的训练器
model = Train_MSAGCN(
    data,
    datatype=data_type,
    device=device,
    use_multiscale=True,        # 启用多尺度空间感知
    use_adaptive_weights=True,  # 启用自适应模态权重
    spatial_scales=None,        # 按datatype固定尺度
    save_dir='MSAGCN_results/',  # 保存目录
    log_interval=50             # 减少打印频率
)

# train model
output = model.train()

adata = adata_omics1.copy()
adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1'].copy()
adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2'].copy()
adata.obsm['MSAGCN'] = output['MSAGCN'].copy()
adata.obsm['alpha'] = output['alpha']
adata.obsm['alpha_omics1'] = output['alpha_omics1']
adata.obsm['alpha_omics2'] = output['alpha_omics2']

from MSAGCN.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='MSAGCN', add_key='MSAGCN', n_clusters=18, method=tool, use_pca=True)

# 4. 无监督聚类评价指标

from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# 获取数据
embeddings = adata.obsm['MSAGCN']
cluster_labels = adata.obs['MSAGCN'].astype(str).values
spatial_coords = adata.obsm['spatial']

# 4.1 轮廓系数 (Silhouette Coefficient)
sc_score = silhouette_score(embeddings, cluster_labels)

# 4.2 Davies-Bouldin指数 (DB Index)
db_score = davies_bouldin_score(embeddings, cluster_labels)

# 4.3 Moran's I 空间自相关指数
def compute_morans_i(cluster_labels, spatial_coords, k=6):
    # 将聚类标签转换为数值
    unique_labels = sorted(set(cluster_labels))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_num[label] for label in cluster_labels])

    n = len(numeric_labels)

    # 构建k近邻权重矩阵
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(spatial_coords)
    distances, indices = nbrs.kneighbors(spatial_coords)

    W = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k+1):
            neighbor_idx = indices[i, j]
            if distances[i, j] > 0:
                W[i, neighbor_idx] = 1.0 / distances[i, j]

    # 行标准化
    row_sums = W.sum(axis=1)
    for i in range(n):
        if row_sums[i] > 0:
            W[i, :] = W[i, :] / row_sums[i]

    # 计算Moran's I
    mean_val = np.mean(numeric_labels)
    numerator = 0
    denominator = 0

    for i in range(n):
        for j in range(n):
            numerator += W[i, j] * (numeric_labels[i] - mean_val) * (numeric_labels[j] - mean_val)
        denominator += (numeric_labels[i] - mean_val) ** 2

    if denominator == 0:
        return 0

    morans_i = (n / np.sum(W)) * (numerator / denominator)
    return morans_i

morans_i = compute_morans_i(cluster_labels, spatial_coords)

# adata.write('MSADA_mousebrain_embedding.h5ad')

