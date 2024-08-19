import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import init
from isegm.model.modeling.gnnlayers import AdjBlock, SimBlock, CrossAttention, mean_query


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)       # FP32
        # zero_vec = -6e4*torch.ones_like(e)         # FP16
        attention = torch.where(adj > 0.5, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=0.1, alpha=0.2, nheads=8):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x

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

class GATFusionBlock(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super(GATFusionBlock, self).__init__()
        self.sim_p = SimBlock(dim)
        self.sim_n = SimBlock(dim)
        self.adj = AdjBlock(dim)
        self.gnn_p = GAT(dim, dim)  # GAT
        self.gnn_n = GAT(dim, dim)
        self.im2query = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.query2img = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def clamp_nodes(self, w, topk=32, thr=0.97):
        w_m = w > thr
        num_true = torch.sum(w_m).item()
        if num_true > topk:
            _, idxs = torch.topk(w, topk)
            w_m = torch.zeros_like(w_m)
            w_m[idxs] = True

        return w_m

    def forward(self, x, mask, pos_emb):
        # 创建正点击和负点击的布尔掩码
        pos_mask = (mask.unsqueeze(-1) == 1).float()
        neg_mask = (mask.unsqueeze(-1) == -1).float()
        have_posclick = pos_mask.squeeze(-1).sum(dim=1) > 0
        have_negclick = neg_mask.squeeze(-1).sum(dim=1) > 0
        # 用布尔掩码筛选正点击的tokens
        pos_query = mean_query(x, pos_mask)
        neg_query = mean_query(x, neg_mask)
        # 计算与用户点击相似的tokens，这些tokens作为节点参与GAT计算
        pos_sim = F.sigmoid(self.sim_p(x, pos_query)).squeeze(-1) * have_posclick.unsqueeze(-1)
        neg_sim = F.sigmoid(self.sim_n(x, neg_query)).squeeze(-1) * have_negclick.unsqueeze(-1)
        feats = []
        for i in range(x.shape[0]):
            w_p, w_n, feat = pos_sim[i], neg_sim[i], x[i]
            p_mask = self.clamp_nodes(w_p)  # > 0.97)
            node_p = feat[p_mask, :]
            if len(node_p) == 0:
                node_p = torch.zeros_like(feat[0, :]).unsqueeze(0)
            adj_mat_p = self.adj(node_p)
            query_p = self.gnn_p(node_p, adj_mat_p) + pos_emb[:, p_mask, :].squeeze(0)

            n_mask = self.clamp_nodes(w_n) # > 0.97)
            node_n = feat[n_mask, :]
            if len(node_n) == 0:
                node_n = torch.zeros_like(feat[0, :]).unsqueeze(0)
            adj_mat_n = self.adj(node_n)
            query_n = self.gnn_n(node_n, adj_mat_n) + pos_emb[:, n_mask, :].squeeze(0)

            feat = feat.unsqueeze(0)
            query = torch.cat([query_p, query_n]).unsqueeze(0)
            query, _ = self.im2query(query, feat)
            feat_, attn = self.query2img(feat, query)
            feats.append(feat_)
        x = torch.cat(feats, dim=0)
        return x, {'pos_sim': pos_sim, 'neg_sim': neg_sim, 'adj_mat_p': adj_mat_p, 'adj_mat_n': adj_mat_n, 'attn': attn}


class GATFusionBlockPosOnly(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super(GATFusionBlockPosOnly, self).__init__()
        self.sim_p = SimBlock(dim)
        self.sim_n = SimBlock(dim)
        self.adj = AdjBlock(dim)
        self.gnn_p = GAT(dim, dim)  # GAT
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        # self.im2query = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                attn_drop=attn_drop, proj_drop=drop)
        self.query2img = CrossAttention(dim, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        attn_drop=attn_drop, proj_drop=drop)
        self.norm5 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim)
        )
        # self.img2token_gamma = torch.nn.Parameter(torch.randn(1, 1, dim), requires_grad=True)
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
        feats = []
        for i in range(x.shape[0]):
            w_p, feat = pos_sim[i], x[i]
            p_mask = w_p > 0.97
            if p_mask.sum()>=1:
                node_p = feat[p_mask, :]
                adj_mat_p = self.adj(node_p)
                query_p = self.gnn_p(node_p, adj_mat_p) + pos_emb[:, p_mask, :].squeeze(0)
            else:
                query_p = torch.zeros([1,feat.shape[-1]]).to(x.device)
            feat = feat.unsqueeze(0)
            query = torch.cat([query_p, ]).unsqueeze(0)
            # query = query + self.img2token_gamma * self.im2query(self.norm1(query), self.norm2(feat))
            feats.append(feat + self.token2img_gamma * self.query2img(self.norm3(feat), self.norm4(query)))
        x = torch.cat(feats, dim=0)
        return x, {'pos_sim': pos_sim, 'neg_sim': None}

