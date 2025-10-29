import os
import h5py
import matplotlib
import numpy as np
import sklearn
import torch
import pandas as pd
import scanpy as sc
from MSAGCN import MSAGCN_pyG
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
# Environment configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.2'
random_seed = 2022


# Importing mouse visual cortex STARMap data
df_data = pd.read_csv('data/mouse_cotex/MVC_counts.csv',sep=",",header=0,na_filter=False,index_col=0)
df_meta = pd.read_csv('data/mouse_cotex/MVC_meta.csv',sep=",",header=0,na_filter=False,index_col=0)
df_pixels = df_meta.iloc[:,2:4] #空间位置信息，所有行，第三和第四列
df_labels = list(df_meta.iloc[:,1]) #标签，所有行，第二列


adata = sc.AnnData(X = df_data)
adata.obs['LayerName'] = df_labels # Combining HPC and CC
adata.obs['LayerName_2'] = list(df_meta.iloc[:,4]) # Separating HPC and CC

# Spatial positions
adata.obsm['spatial'] = np.array(df_pixels)
adata.obs['x_pos'] = adata.obsm['spatial'][:,0]
adata.obs['y_pos'] = adata.obsm['spatial'][:,1]
label_type = ['L1','L2/3','L4','L5','L6','HPC/CC']

for label in label_type:
    count = sum([1 for l in df_labels if l == label])

# Shuffling L4/L5 and L5/L6 of the original data, respectively.
index_all = [np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[0]])]
for k in range(1,len(label_type)):
    temp_idx = np.array([i for i in range(len(df_labels)) if df_labels[i] == label_type[k]])
    index_all.append(temp_idx)
index_int1 = np.array(list(index_all[2]) + list(index_all[3]))  # L4 + L5
index_int2 = np.array(list(index_all[4]) + list(index_all[3]))  # L6 + L5


# Adding Gaussian noise to each omics

# 模态1 (RNA-like)
adata_omics1 = adata.copy()
np.random.seed(random_seed)
data_noise_1 = 1 + np.random.normal(0,0.05,adata.shape)
adata_omics1.X[index_int1,:] = np.multiply(adata.X,data_noise_1)[np.random.permutation(index_int1),:]

# 模态2 (Protein-like)
adata_omics2 = adata.copy()
np.random.seed(random_seed+1)
data_noise_2 = 1 + np.random.normal(0,0.05,adata.shape)
adata_omics2.X[index_int2,:] = np.multiply(adata.X,data_noise_2)[np.random.permutation(index_int2),:]

# 数据预处理
data_type = 'SPOTS'

# 预处理模态1 (RNA-like)
sc.pp.pca(adata_omics1, n_comps=50)
adata_omics1.obsm['feat'] = adata_omics1.obsm['X_pca']

# 预处理模态2 (Protein-like)

sc.pp.pca(adata_omics2, n_comps=50)
adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca']


# 构建邻接图
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
# 保存为H5AD
# adata.write('SpatialGlue_picture/spatialglue人类淋巴_embeddings.h5ad')

# we set 'mclust' as clustering tool by default. Users can also select 'leiden' and 'louvain'
from MSAGCN.utils import clustering
tool = 'mclust' # mclust, leiden, and louvain
clustering(adata, key='MSAGCN', add_key='MSAGCN', n_clusters=6, method=tool, start=0.3, end=0.7,use_pca=True)

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score
true_labels = adata.obs['LayerName'].astype(str).values
pred_labels = adata.obs['MSAGCN'].astype(str).values

ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
fmi = fowlkes_mallows_score(true_labels, pred_labels)


# import matplotlib.pyplot as plt
# import scanpy as sc
# import matplotlib.cm as cm
# import numpy as np
# from sklearn.metrics import confusion_matrix
#
# # 设置matplotlib字体为黑体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#
# # 自定义六个小清新高对比度颜色
# custom_colors = [
#     "#2499F8",  # 蓝色
#     "#6C73F5DF",  # 淡紫色
#     "#FFBFCB",  # 粉色
#     "#F2B56E",  # 橙色
#     "#CCCCCC",  # 灰色
#     "#ED6C6C"  # 红色
# ]
#
# # 确保LayerName是分类变量
# adata.obs['LayerName'] = adata.obs['LayerName'].astype('category')
#
# # 首先设置真实标签的颜色
# n_categories = len(adata.obs['LayerName'].cat.categories)
# print(f"LayerName类别数量: {n_categories}")
# print(f"LayerName类别: {list(adata.obs['LayerName'].cat.categories)}")
#
# # 为LayerName分配颜色
# if n_categories <= len(custom_colors):
#     adata.uns['LayerName_colors'] = custom_colors[:n_categories]
# else:
#     # 如果类别数超过自定义颜色数，重复使用颜色
#     extended_colors = custom_colors * ((n_categories // len(custom_colors)) + 1)
#     adata.uns['LayerName_colors'] = extended_colors[:n_categories]
#
# print(f"分配的颜色: {adata.uns['LayerName_colors']}")
#
#
# # 颜色对齐函数
# def align_cluster_colors(adata, true_col='LayerName', pred_col='SEDR'):
#     """
#     基于最大重叠度对齐聚类颜色与真实标签颜色
#     """
#     true_labels = adata.obs[true_col].values
#     pred_labels = adata.obs[pred_col].values
#
#     # 获取唯一标签
#     true_unique = adata.obs[true_col].cat.categories
#     pred_unique = np.unique(pred_labels)
#
#     # 计算混淆矩阵 - 修复标签处理
#     from sklearn.preprocessing import LabelEncoder
#     le_true = LabelEncoder()
#     le_pred = LabelEncoder()
#
#     true_encoded = le_true.fit_transform(true_labels)
#     pred_encoded = le_pred.fit_transform(pred_labels)
#
#     cm = confusion_matrix(true_encoded, pred_encoded)
#
#     # 为每个预测类别找到最匹配的真实类别
#     color_mapping = {}
#     used_true_indices = set()
#     true_colors = adata.uns[f'{true_col}_colors']
#
#     # 按重叠度排序进行匹配
#     pred_true_pairs = []
#     for pred_idx, pred_label in enumerate(pred_unique):
#         for true_idx, true_label in enumerate(true_unique):
#             # 确保索引在有效范围内
#             if true_idx < cm.shape[0] and pred_idx < cm.shape[1]:
#                 overlap = cm[true_idx, pred_idx]
#                 pred_true_pairs.append((overlap, pred_label, true_idx))
#
#     # 按重叠度降序排序
#     pred_true_pairs.sort(reverse=True)
#
#     # 贪心匹配
#     for overlap, pred_label, true_idx in pred_true_pairs:
#         if (pred_label not in color_mapping and
#                 true_idx not in used_true_indices and
#                 true_idx < len(true_colors)):  # 添加边界检查
#             color_mapping[pred_label] = true_colors[true_idx]
#             used_true_indices.add(true_idx)
#
#     # 为未匹配的预测标签分配剩余颜色
#     remaining_colors = [true_colors[i] for i in range(len(true_colors))
#                         if i not in used_true_indices]
#     unmatched_preds = [pred for pred in pred_unique if pred not in color_mapping]
#
#     # 如果剩余颜色不够，添加更多颜色
#     extra_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
#     all_available_colors = remaining_colors + extra_colors
#
#     for i, pred_label in enumerate(unmatched_preds):
#         if i < len(all_available_colors):
#             color_mapping[pred_label] = all_available_colors[i]
#         else:
#             color_mapping[pred_label] = '#CCCCCC'  # 默认灰色
#
#     # 按预测标签顺序创建颜色列表
#     aligned_colors = [color_mapping[pred_label] for pred_label in sorted(pred_unique)]
#     adata.uns[f'{pred_col}_colors'] = aligned_colors
#
#     return color_mapping
#
# domains = 'MSADA'
# num_clusters = len(adata.obs[domains].unique())
# print(f"聚类数量: {num_clusters}")
#
# # 对齐SEDR聚类颜色与真实标签颜色
# color_mapping = align_cluster_colors(adata, 'LayerName', 'MSADA')
# print("颜色对应关系:")
# for pred_label, color in color_mapping.items():
#     print(f"spaceflow Cluster {pred_label}: {color}")
#
#
# # 3. 可视化空间域划分图
# fig3, ax3 = plt.subplots(1, 1, figsize=(4, 4))  # 增加宽度以容纳图例
# sc.pl.embedding(adata, basis='spatial', color='MSADA', ax=ax3, title=f'MSADA',
#                 s=70, show=False, frameon=False, legend_loc='right margin', legend_fontsize=8)
# ax3.set_xticks([])
# ax3.set_yticks([])
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)
# ax3.spines['bottom'].set_visible(False)
# ax3.spines['left'].set_visible(False)
# # 设置标题字体为黑体，不加粗
# plt.tight_layout()
#
# # 保存空间域划分图 PNG和PDF
# spatial_png_path = "./MSADA_results/小鼠视觉皮层_空间域划分.png"
# # spatial_pdf_path = "./SEDR_results/mouse_cotex/mouse_cortex_omics1_空间域划分.pdf"
# plt.savefig(spatial_png_path, format='png', bbox_inches='tight', dpi=600)
# # plt.savefig(spatial_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1, dpi=600)
# print(f"spa空间域划分图已保存为 {spatial_png_path} (PNG格式, 600 DPI)")
# # print(f"SEDR空间域划分图已保存为 {spatial_pdf_path} (PDF格式, 600 DPI)")
# plt.show()
#
# import os
# os.makedirs('MSADA_results', exist_ok=True)
# adata.write('MSADA_results/MSADA_mouse_cotex_embeddings.h5ad')
