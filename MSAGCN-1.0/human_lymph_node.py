import os
import torch
import pandas as pd
import scanpy as sc
import MSAGCN

# Environment configuration
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = 'C:/Program Files/R/R-4.3.2'
random_seed = 2022

# read data
file_fold = 'data/human_lynode/' #please replace 'file_fold' with the download path

adata_omics1 = sc.read_h5ad(file_fold + 'adata_RNA_with_annotation.h5ad')
adata_omics2 = sc.read_h5ad(file_fold + 'adata_ADT_with_annotation.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

# Specify data type
data_type = '10x'

# Fix random seed
from MSAGCN.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)

from MSAGCN.preprocess import clr_normalize_each_cell, pca

# RNA
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high =  adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars-1)

# Protein
adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars-1)

from MSAGCN.preprocess import construct_neighbor_graph

data = construct_neighbor_graph(
    adata_omics1, adata_omics2,
    datatype=data_type,
    use_multiscale=True,  
    spatial_scales=None   
)

from MSAGCN.MSAGCN_pyG import Train_MSAGCN

model = Train_MSAGCN(
    data,
    datatype=data_type,
    device=device,
    use_multiscale=True,        
    use_adaptive_weights=True,  
    spatial_scales=None,        
    log_interval=50             
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
clustering(adata, key='MSAGCN', add_key='MSAGCN', n_clusters=10, method=tool, use_pca=True)

