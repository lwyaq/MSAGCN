import torch
from tqdm import tqdm
import torch.nn.functional as F
import json
import os
from .model import Encoder_overall
from .preprocess import adjacent_matrix_preprocessing


class Train_MSAGCN:
    def __init__(self,
                 data,
                 datatype='SPOTS',
                 device=torch.device('cpu'),
                 random_seed=2022,
                 learning_rate=0.0001,
                 weight_decay=0.00,
                 epochs=600,
                 dim_input=3000,
                 dim_output=64,
                 weight_factors=[1, 5, 1, 1],
                 use_multiscale=True,
                 use_adaptive_weights=True,
                 spatial_scales=None,
                 save_dir='MSAGCN_results/',
                 log_interval=10
                 ):
        '''\

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input, Our current model supports 'SPOTS', 'Stereo-CITE-seq', and 'Spatial-ATAC-RNA-seq'. We plan to extend our model for more data types in the future.
            The default is 'SPOTS'.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight decay to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 1500.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        weight_factors : list, optional
            Weight factors to balance the influcences of different omics data on model training.

        Returns
        -------
        The learned representation 'self.emb_combined'.

        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors
        self.use_multiscale = use_multiscale
        self.use_adaptive_weights = use_adaptive_weights
        self.spatial_scales = spatial_scales
        self.save_dir = save_dir
        self.log_interval = log_interval


        self.dimension_info = {
            'input_dims': {},
            'intermediate_dims': {},
            'output_dims': {},
            'graph_stats': {}
        }

        # adj
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)


        if self.use_multiscale and 'multiscale_adj_spatial' in self.adata_omics1.uns:
            self.multiscale_adjs_omics1 = self._process_multiscale_adjs(
                self.adata_omics1.uns['multiscale_adj_spatial'])
            self.multiscale_adjs_omics2 = self._process_multiscale_adjs(
                self.adata_omics2.uns['multiscale_adj_spatial'])
            self.spatial_scales = self.adata_omics1.uns['spatial_scales']
        else:
            self.multiscale_adjs_omics1 = None
            self.multiscale_adjs_omics2 = None

        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        if self.use_adaptive_weights:
            self.spatial_coords = torch.FloatTensor(self.adata_omics1.obsm['spatial'].copy()).to(self.device)
        else:
            self.spatial_coords = None

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs


        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        if self.datatype == 'SPOTS':
            self.epochs = 1200
            self.weight_factors = [1, 1, 1, 1]

        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1500
            self.weight_factors = [1, 10, 1, 10]

        elif self.datatype == '10x':
            self.epochs =1500
            self.weight_factors = [1, 5, 1, 10]

        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 1600
            self.weight_factors = [1, 1, 1, 1]

    def _process_multiscale_adjs(self, multiscale_graphs):

        from .preprocess import transform_adjacent_matrix, preprocess_graph

        processed_adjs = []
        for adj_df in multiscale_graphs:

            adj_sparse = transform_adjacent_matrix(adj_df)
            adj_array = adj_sparse.toarray()


            adj_array = adj_array + adj_array.T
            adj_array = (adj_array > 0).astype(float)


            adj_processed = preprocess_graph(adj_array).to(self.device)
            processed_adjs.append(adj_processed)

        return processed_adjs

    def train(self):

        self.model = Encoder_overall(
            self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2,
            use_multiscale=self.use_multiscale,
            use_adaptive_weights=self.use_adaptive_weights,
            spatial_scales=self.spatial_scales
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        self.model.train()

        if self.spatial_scales:
            _ = self.spatial_scales

        for epoch in tqdm(range(self.epochs), desc="Training Progress", disable=False):
            self.model.train()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                spatial_coords=self.spatial_coords,
                multiscale_adjs_omics1=self.multiscale_adjs_omics1,
                multiscale_adjs_omics2=self.multiscale_adjs_omics2
            )


            if epoch == 0:
                self._record_dimensions(results)

            # reconstruction loss
            self.loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            self.loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # correspondence loss
            self.loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            self.loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            loss = self.weight_factors[0] * self.loss_recon_omics1 + self.weight_factors[1] * self.loss_recon_omics2 + \
                   self.weight_factors[2] * self.loss_corr_omics1 + self.weight_factors[3] * self.loss_corr_omics2
            spatial_reg_loss_value = 0.0
            if self.use_adaptive_weights and results['adaptive_modality_weights'] is not None:
                spatial_reg_loss = self._compute_spatial_regularization_loss(
                    results['adaptive_modality_weights'], self.spatial_coords)
                loss += 0.1 * spatial_reg_loss
                # loss += 1.0 * spatial_reg_loss
                spatial_reg_loss_value = spatial_reg_loss.item()


            self._print_training_progress(epoch, {
                'total': loss.item(),
                'recon_omics1': self.loss_recon_omics1.item(),
                'recon_omics2': self.loss_recon_omics2.item(),
                'corr_omics1': self.loss_corr_omics1.item(),
                'corr_omics2': self.loss_corr_omics2.item(),
                'spatial_reg': spatial_reg_loss_value
            })

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        

        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                spatial_coords=self.spatial_coords,
                multiscale_adjs_omics1=self.multiscale_adjs_omics1,
                multiscale_adjs_omics2=self.multiscale_adjs_omics2
            )

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'MSAGCN': emb_combined.detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
            'alpha': results['alpha'].detach().cpu().numpy()
        }


        if results['scale_weights_omics1'] is not None:
            output['scale_weights_omics1'] = results['scale_weights_omics1'].detach().cpu().numpy()
            output['scale_weights_omics2'] = results['scale_weights_omics2'].detach().cpu().numpy()

        if results['adaptive_modality_weights'] is not None:
            output['adaptive_modality_weights'] = results['adaptive_modality_weights'].detach().cpu().numpy()

        return output

    def _compute_spatial_regularization_loss(self, weights, spatial_coords, lambda_reg=0.1):

        dist_matrix = torch.cdist(spatial_coords, spatial_coords, p=2)


        k = min(6, weights.size(0) - 1)
        _, neighbor_indices = torch.topk(dist_matrix, k + 1, largest=False, dim=1)
        neighbor_indices = neighbor_indices[:, 1:]  


        reg_loss = 0
        for i in range(weights.size(0)):
            neighbors = neighbor_indices[i]
            weight_diff = weights[i].unsqueeze(0) - weights[neighbors]
            reg_loss += torch.mean(torch.sum(weight_diff ** 2, dim=1))

        return lambda_reg * reg_loss / weights.size(0)

    def _record_dimensions(self, results):


        self.dimension_info['input_dims'] = {
            'features_omics1': list(self.features_omics1.shape),
            'features_omics2': list(self.features_omics2.shape),
            'spatial_coords': list(self.spatial_coords.shape) if self.spatial_coords is not None else None,
            'n_cells_omics1': self.n_cell_omics1,
            'n_cells_omics2': self.n_cell_omics2
        }


        self.dimension_info['intermediate_dims'] = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                self.dimension_info['intermediate_dims'][key] = list(value.shape)
            elif value is not None:
                self.dimension_info['intermediate_dims'][key] = str(type(value))

        self.dimension_info['graph_stats'] = {
            'adj_spatial_omics1_nnz': self.adj_spatial_omics1._nnz(),
            'adj_spatial_omics2_nnz': self.adj_spatial_omics2._nnz(),
            'adj_feature_omics1_nnz': self.adj_feature_omics1._nnz(),
            'adj_feature_omics2_nnz': self.adj_feature_omics2._nnz(),
        }

        if self.multiscale_adjs_omics1 is not None:
            self.dimension_info['graph_stats']['multiscale_nnz_omics1'] = [
                adj._nnz() for adj in self.multiscale_adjs_omics1
            ]
            self.dimension_info['graph_stats']['multiscale_nnz_omics2'] = [
                adj._nnz() for adj in self.multiscale_adjs_omics2
            ]
            self.dimension_info['graph_stats']['spatial_scales'] = self.spatial_scales

           

    def _print_training_progress(self, epoch, loss_dict):
        return











