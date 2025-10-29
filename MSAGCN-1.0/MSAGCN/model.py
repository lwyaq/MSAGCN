import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Encoder_overall(Module):
    """\
    Overall encoder with multi-scale spatial awareness and adaptive modality weighting.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2.
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.
    use_multiscale: bool
        Whether to use multi-scale spatial encoding.
    use_adaptive_weights: bool
        Whether to use adaptive modality weighting.
    spatial_scales: list
        List of spatial scales for multi-scale encoding.

    Returns
    -------
    results: a dictionary including representations and modality weights.

    """

    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 dropout=0.0, act=F.relu, use_multiscale=True, use_adaptive_weights=True,
                 spatial_scales=[3, 6, 12, 24]):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        self.use_multiscale = use_multiscale
        self.use_adaptive_weights = use_adaptive_weights
        self.spatial_scales = spatial_scales

        # 选择编码器类型
        if use_multiscale:
            self.encoder_omics1 = MultiScaleEncoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1,
                                                   scales=spatial_scales, dropout=dropout)
            self.encoder_omics2 = MultiScaleEncoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2,
                                                   scales=spatial_scales, dropout=dropout)
        else:
            self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
            self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)

        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        # 注意力层
        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

        # 自适应模态权重模块
        if use_adaptive_weights:
            self.adaptive_weighting = AdaptiveModalityWeighting(
                emb_dim=max(self.dim_out_feat_omics1, self.dim_out_feat_omics2),
                spatial_dim=2,
                n_modalities=2
            )

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1,
                adj_spatial_omics2, adj_feature_omics2, spatial_coords=None,
                multiscale_adjs_omics1=None, multiscale_adjs_omics2=None):
        """
        Enhanced forward pass with multi-scale and adaptive weighting support

        Parameters
        ----------
        spatial_coords : torch.Tensor, optional
            Spatial coordinates for adaptive weighting [n_cells, 2]
        multiscale_adjs_omics1/2 : list, optional
            Multi-scale adjacency matrices for each modality
        """

        # 多尺度编码 or 传统编码
        if self.use_multiscale and multiscale_adjs_omics1 is not None:
            # 多尺度空间编码
            emb_latent_spatial_omics1, scale_weights_omics1 = self.encoder_omics1(
                features_omics1, multiscale_adjs_omics1)
            emb_latent_spatial_omics2, scale_weights_omics2 = self.encoder_omics2(
                features_omics2, multiscale_adjs_omics2)

            # 特征图编码（使用传统方法）
            emb_latent_feature_omics1 = self.encoder_omics1.scale_encoders[0](
                features_omics1, adj_feature_omics1)
            emb_latent_feature_omics2 = self.encoder_omics2.scale_encoders[0](
                features_omics2, adj_feature_omics2)
        else:
            # 传统编码方式
            emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
            emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
            emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
            emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)
            scale_weights_omics1 = scale_weights_omics2 = None

        # within-modality attention aggregation layer
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        # 自适应模态权重融合 or 传统跨模态注意力
        if self.use_adaptive_weights and spatial_coords is not None:
            # 使用自适应权重进行模态融合
            adaptive_weights, emb_latent_combined = self.adaptive_weighting(
                [emb_latent_omics1, emb_latent_omics2],
                spatial_coords,
                adj_spatial_omics1
            )
            alpha_omics_1_2 = adaptive_weights  # 重新定义alpha为自适应权重
        else:
            # 传统的跨模态注意力
            emb_latent_combined, alpha_omics_1_2 = self.atten_cross(emb_latent_omics1, emb_latent_omics2)
            adaptive_weights = None

        # reverse the integrated representation back into the original expression space
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # consistency encoding (需要适配多尺度编码器)
        if self.use_multiscale:
            # 对于多尺度编码器，使用第一个尺度进行一致性编码
            emb_latent_omics1_across_recon = self.encoder_omics2.scale_encoders[0](
                self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
            emb_latent_omics2_across_recon = self.encoder_omics1.scale_encoders[0](
                self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)
        else:
            emb_latent_omics1_across_recon = self.encoder_omics2(
                self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
            emb_latent_omics2_across_recon = self.encoder_omics1(
                self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        results = {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha_omics1': alpha_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_omics_1_2,
            # 新增的输出
            'scale_weights_omics1': scale_weights_omics1,
            'scale_weights_omics2': scale_weights_omics2,
            'adaptive_modality_weights': adaptive_weights
        }

        return results


class Encoder(Module):
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Latent representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class Decoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.

    Returns
    -------
    Reconstructed representation.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)

        return x


class AttentionLayer(Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha


class MultiScaleEncoder(Module):
    """\
    多尺度空间感知编码器

    Parameters
    ----------
    in_feat : int
        输入特征维度
    out_feat : int
        输出特征维度
    scales : list
        多个空间尺度
    dropout : float
        Dropout概率
    """

    def __init__(self, in_feat, out_feat, scales=[3, 6, 12, 24], dropout=0.0):
        super(MultiScaleEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.scales = scales
        self.dropout = dropout

        # 为每个尺度创建独立的编码器
        self.scale_encoders = nn.ModuleList([
            Encoder(in_feat, out_feat, dropout) for _ in scales
        ])

        # 尺度注意力层
        self.scale_attention = ScaleAttentionLayer(out_feat, len(scales))

        # 尺度融合层
        self.scale_fusion = nn.Sequential(
            nn.Linear(out_feat * len(scales), out_feat * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_feat * 2, out_feat)
        )

    def forward(self, features, multiscale_adjs):
        """
        前向传播

        Parameters
        ----------
        features : torch.Tensor
            输入特征 [n_cells, in_feat]
        multiscale_adjs : list
            多尺度邻接矩阵列表

        Returns
        -------
        fused_embedding : torch.Tensor
            融合后的多尺度嵌入
        scale_weights : torch.Tensor
            各尺度的注意力权重
        """
        scale_embeddings = []

        # 在每个尺度上进行编码
        for i, adj in enumerate(multiscale_adjs):
            emb = self.scale_encoders[i](features, adj)
            scale_embeddings.append(emb)

        # 堆叠所有尺度的嵌入
        stacked_embeddings = torch.stack(scale_embeddings, dim=1)  # [n_cells, n_scales, out_feat]

        # 计算尺度注意力权重
        scale_weights = self.scale_attention(stacked_embeddings)  # [n_cells, n_scales]

        # 加权融合多尺度嵌入
        weighted_embeddings = torch.sum(
            stacked_embeddings * scale_weights.unsqueeze(-1), dim=1
        )  # [n_cells, out_feat]

        # 可选：通过全连接层进一步融合
        # concatenated = torch.cat(scale_embeddings, dim=-1)  # [n_cells, out_feat * n_scales]
        # fused_embedding = self.scale_fusion(concatenated)

        return weighted_embeddings, scale_weights


class ScaleAttentionLayer(Module):
    """\
    尺度注意力层，用于学习不同空间尺度的重要性
    """

    def __init__(self, emb_dim, n_scales):
        super(ScaleAttentionLayer, self).__init__()
        self.emb_dim = emb_dim
        self.n_scales = n_scales

        # 注意力计算网络
        self.attention_net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Linear(emb_dim // 2, 1)
        )

    def forward(self, scale_embeddings):
        """
        计算尺度注意力权重

        Parameters
        ----------
        scale_embeddings : torch.Tensor
            多尺度嵌入 [n_cells, n_scales, emb_dim]

        Returns
        -------
        attention_weights : torch.Tensor
            注意力权重 [n_cells, n_scales]
        """
        # 为每个尺度计算注意力分数
        attention_scores = self.attention_net(scale_embeddings)  # [n_cells, n_scales, 1]
        attention_scores = attention_scores.squeeze(-1)  # [n_cells, n_scales]

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


class AdaptiveModalityWeighting(Module):
    """\
    自适应模态权重学习模块
    根据空间位置和局部特征动态调整不同模态的重要性

    Parameters
    ----------
    emb_dim : int
        嵌入维度
    spatial_dim : int
        空间坐标维度（通常为2）
    n_modalities : int
        模态数量
    """

    def __init__(self, emb_dim, spatial_dim=2, n_modalities=2, hidden_dim=64):
        super(AdaptiveModalityWeighting, self).__init__()
        self.emb_dim = emb_dim
        self.spatial_dim = spatial_dim
        self.n_modalities = n_modalities
        self.hidden_dim = hidden_dim

        # 空间上下文编码器
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 局部特征编码器
        self.local_feature_encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 模态权重预测器
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_modalities),
            nn.Softmax(dim=-1)
        )

        # 空间平滑层（可选）
        self.spatial_smoothing = SpatialSmoothingLayer(n_modalities)

    def forward(self, embeddings_list, spatial_coords, adj_spatial=None):
        """
        前向传播

        Parameters
        ----------
        embeddings_list : list of torch.Tensor
            不同模态的嵌入列表 [emb1, emb2, ...]
        spatial_coords : torch.Tensor
            空间坐标 [n_cells, spatial_dim]
        adj_spatial : torch.Tensor, optional
            空间邻接矩阵，用于空间平滑

        Returns
        -------
        adaptive_weights : torch.Tensor
            自适应模态权重 [n_cells, n_modalities]
        weighted_embeddings : torch.Tensor
            加权后的融合嵌入
        """
        # 编码空间上下文
        spatial_context = self.spatial_encoder(spatial_coords)  # [n_cells, hidden_dim//2]

        # 计算局部特征上下文（使用所有模态的平均）
        avg_embedding = torch.stack(embeddings_list, dim=0).mean(dim=0)  # [n_cells, emb_dim]
        local_context = self.local_feature_encoder(avg_embedding)  # [n_cells, hidden_dim//2]

        # 融合空间和特征上下文
        combined_context = torch.cat([spatial_context, local_context], dim=-1)  # [n_cells, hidden_dim]

        # 预测模态权重
        raw_weights = self.weight_predictor(combined_context)  # [n_cells, n_modalities]

        # 可选：空间平滑
        if adj_spatial is not None:
            adaptive_weights = self.spatial_smoothing(raw_weights, adj_spatial)
        else:
            adaptive_weights = raw_weights

        # 应用权重进行模态融合
        weighted_embeddings = self._apply_weights(embeddings_list, adaptive_weights)

        return adaptive_weights, weighted_embeddings

    def _apply_weights(self, embeddings_list, weights):
        """
        应用权重进行模态融合

        Parameters
        ----------
        embeddings_list : list of torch.Tensor
            模态嵌入列表
        weights : torch.Tensor
            模态权重 [n_cells, n_modalities]

        Returns
        -------
        weighted_embedding : torch.Tensor
            加权融合后的嵌入
        """
        # 堆叠所有模态嵌入
        stacked_embeddings = torch.stack(embeddings_list, dim=1)  # [n_cells, n_modalities, emb_dim]

        # 应用权重
        weighted_embedding = torch.sum(
            stacked_embeddings * weights.unsqueeze(-1), dim=1
        )  # [n_cells, emb_dim]

        return weighted_embedding


class SpatialSmoothingLayer(Module):
    """\
    空间平滑层，确保相邻细胞的模态权重相似
    """

    def __init__(self, n_modalities, smoothing_strength=0.1):
        super(SpatialSmoothingLayer, self).__init__()
        self.n_modalities = n_modalities
        self.smoothing_strength = smoothing_strength

    def forward(self, weights, adj_spatial):
        """
        对权重进行空间平滑

        Parameters
        ----------
        weights : torch.Tensor
            原始权重 [n_cells, n_modalities]
        adj_spatial : torch.Tensor
            空间邻接矩阵

        Returns
        -------
        smoothed_weights : torch.Tensor
            平滑后的权重
        """
        # 计算邻居权重的平均
        neighbor_weights = torch.spmm(adj_spatial, weights)  # [n_cells, n_modalities]

        # 加权平均：原始权重 + 邻居平均权重
        smoothed_weights = (1 - self.smoothing_strength) * weights + \
                          self.smoothing_strength * neighbor_weights

        # 重新归一化
        smoothed_weights = F.softmax(smoothed_weights, dim=-1)

        return smoothed_weights