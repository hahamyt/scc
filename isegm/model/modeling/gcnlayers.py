import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from torch.nn import init
from isegm.model.modeling.gnnlayers import SimBlock, AdjBlock, CrossAttention, mean_query


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.mm(adj, support)  # Use batch matrix multiplication
        return output

class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNBlock, self).__init__()
        self.gc1 = GraphConvolution(input_dim, input_dim)
        self.gc2 = GraphConvolution(input_dim, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        return self.gc2(x, adj)


class GCNFusionBlock(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm ):
        super(GCNFusionBlock, self).__init__()
        self.sim_p = SimBlock(dim)
        self.sim_n = SimBlock(dim)
        self.adj = AdjBlock(dim)
        self.gnn_p = GCNBlock(dim, dim)
        self.gnn_n = GCNBlock(dim, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.im2query = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.query2img = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm5 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim)
        )
        self.img2token_gamma = torch.nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.token2img_gamma = torch.nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mask, pos_emb):
        # 创建正点击和负点击的布尔掩码
        pos_mask = (mask.unsqueeze(-1) == 1).float()
        neg_mask = (mask.unsqueeze(-1) == -1).float()
        # 用布尔掩码筛选正点击的tokens
        pos_query = mean_query(x, pos_mask)
        neg_query = mean_query(x, neg_mask)

        # 计算与用户点击相似的tokens，这些tokens作为节点参与GAT计算
        pos_sim = F.sigmoid(self.sim_p(x, pos_query)).squeeze()
        neg_sim = F.sigmoid(self.sim_n(x, neg_query)).squeeze()
        ## 选取特定数量的tokens用于查询
        # 正
        weight_p, topk_idx_p = torch.topk(pos_sim, k=pos_sim.shape[-1] // 16)
        weight_n, topk_idx_n = torch.topk(neg_sim, k=neg_sim.shape[-1] // 32)
        if weight_p.dim() == 1:
            weight_p, topk_idx_p = weight_p.unsqueeze(0), topk_idx_p.unsqueeze(0)
        if weight_n.dim() == 1:
            weight_n, topk_idx_n = weight_n.unsqueeze(0), topk_idx_n.unsqueeze(0)

        def batch_forward(idx_p, w_p, idx_n, w_n, feat):
            node_p = feat[idx_p, :] * (w_p > 0.6).int().view(-1, 1)
            ## 计算邻接加权矩阵，即当前query与全部token间相似的节点作为当前节点
            adj_mat_p = self.adj(node_p)
            # 开始利用GCN做特征聚合
            query_p = self.gnn_p(node_p, adj_mat_p)  + pos_emb[:, idx_p, :].squeeze(0)
            node_n = feat[idx_n, :] * (w_n > 0.6).int().view(-1, 1)
            ## 计算邻接加权矩阵，即当前query与全部token间相似的节点作为当前节点
            adj_mat_n = self.adj(node_n)
            # 开始利用GCN做特征聚合
            query_n = self.gnn_n(node_n, adj_mat_n)  + pos_emb[:, idx_n, :].squeeze(0)
            feat = feat.unsqueeze(0)
            query = torch.cat([query_p, query_n]).unsqueeze(0)
            query = query + self.img2token_gamma * self.im2query(self.norm1(query), self.norm2(feat))
            feat = feat + self.token2img_gamma * self.query2img(self.norm3(feat), self.norm4(query))
            feat = feat + self.mlp(self.norm5(feat))
            return feat

        # 使用vmap将函数应用于输入的每一批
        x = torch.vmap(batch_forward, randomness='different')(topk_idx_p, weight_p, topk_idx_n, weight_n, x)
        return x, {'pos_sim': pos_sim, 'neg_sim': neg_sim}


class GCNFusionBlockPosOnly(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super(GCNFusionBlockPosOnly, self).__init__()
        self.sim_p = SimBlock(dim)
        self.sim_n = SimBlock(dim)
        self.adj = AdjBlock(dim)
        self.gnn_p = GCNBlock(dim, dim)  # GAT
        self.gnn_n = GCNBlock(dim, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.im2query = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.query2img = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm5 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim)
        )
        self.img2token_gamma = torch.nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
        self.token2img_gamma = torch.nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, mask, pos_emb):
        # 创建正点击和负点击的布尔掩码
        pos_mask = (mask.unsqueeze(-1) == 1).float()
        # 用布尔掩码筛选正点击的tokens
        pos_query = mean_query(x, pos_mask)

        # 计算与用户点击相似的tokens，这些tokens作为节点参与GAT计算
        pos_sim = F.sigmoid(self.sim_p(x, pos_query)).squeeze()
        ## 选取特定数量的tokens用于查询
        # 正
        weight_p, topk_idx_p = torch.topk(pos_sim, k=pos_sim.shape[-1] // 16)
        if weight_p.dim() == 1:
            weight_p, topk_idx_p = weight_p.unsqueeze(0), topk_idx_p.unsqueeze(0)

        def batch_forward(idx_p, w_p, feat):
            node_p = feat[idx_p, :] * (w_p > 0.6).int().view(-1, 1)
            ## 计算邻接加权矩阵，即当前query与全部token间相似的节点作为当前节点
            adj_mat_p = self.adj(node_p)
            # 开始利用GCN做特征聚合
            query_p = self.gnn_p(node_p, adj_mat_p)  + pos_emb[:, idx_p, :].squeeze(0)
            ## 计算邻接加权矩阵，即当前query与全部token间相似的节点作为当前节点
            # 开始利用GNN做特征聚合
            feat = feat.unsqueeze(0)
            query_p = query_p.unsqueeze(0)
            query_p = query_p + self.img2token_gamma * self.im2query(self.norm1(query_p), self.norm2(feat))
            feat = feat + self.token2img_gamma * self.query2img(self.norm3(feat), self.norm4(query_p))
            feat = feat + self.mlp(self.norm5(feat))
            return feat

        # 使用vmap将函数应用于输入的每一批
        x = torch.vmap(batch_forward, randomness='different')(topk_idx_p, weight_p, x)
        return x, {'pos_sim': pos_sim, 'neg_sim': None}
