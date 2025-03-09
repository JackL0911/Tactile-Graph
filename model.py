import torch
import random
import torch.nn as nn
from torchvision import models
# import wandb
import torch.nn.functional as F
import torch
import dgl
import train.aug as aug
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from superpixels_graph_classification.gcn_net import GCNNet


# 对比损失函数（MoCo损失）
def moco_contrastive_loss(q, k, T):
    """
    计算查询（q）和键（k）表示之间的对比损失。

    参数:
    - q: 查询表示
    - k: 键表示
    - T: 温度系数，用于缩放对比损失

    返回:
    - 损失值
    """
    # 规范化查询和键表示，使它们的L2范数为1
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)

    # 计算查询和键的相似度矩阵（使用点积并缩放）
    logits = torch.mm(q, k.T.detach()) / T  # 将k.detach()用于停止反向传播
    labels = torch.arange(logits.shape[0], dtype=torch.long).to(q.device)  # 构造标签，正例标签为相同样本的索引

    # 计算交叉熵损失
    return nn.CrossEntropyLoss()(logits, labels), logits, labels


# InfoNCE损失函数（主要用于多个样本的对比学习）
def info_nce_loss(q, k, T):
    """
    计算查询（q）和键（k）表示之间的InfoNCE对比损失。

    参数:
    - q: 查询表示
    - k: 键表示
    - T: 温度系数，用于缩放对比损失

    返回:
    - 损失值
    """
    # 拼接查询和键表示
    features = torch.cat([q, k], dim=0)
    batch_size = features.shape[0] // 2  # 每个批次包含两种视图（查询和键）

    # 创建标签矩阵
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    labels = labels.to(device)

    # 规范化查询和键表示
    features = F.normalize(features, dim=1)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(features, features.T)

    # 创建掩码，去除对角线上的元素
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # 选择正样本（对角线上的相似度）和负样本（其余样本）
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # 拼接正样本和负样本，构造logits
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    # 使用温度缩放logits并计算交叉熵损失
    logits = logits / T
    loss = torch.nn.CrossEntropyLoss().to(device)(logits, labels)
    return loss, logits, labels


# 使用动量更新键编码器，确保键编码器逐渐与查询编码器对齐
@torch.no_grad()  # 禁用梯度计算，确保不对键编码器的参数进行反向传播
def momentum_update_key_encoder(base_q, base_k, head_intra_q, head_intra_k, m):
    """
    基于查询编码器更新键编码器的动量。这确保了键编码器随着时间的推移逐渐与查询编码器对齐，
    而无需直接反向传播。

    参数:
    - base_q: 查询编码器的基础部分
    - base_k: 键编码器的基础部分
    - head_intra_q: 查询编码器的内模头
    - head_intra_k: 键编码器的内模头
    - m: 动量因子，用于控制键编码器的更新步伐
    """
    for param_q, param_k in zip(base_q.parameters(), base_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)
    for param_q, param_k in zip(head_intra_q.parameters(), head_intra_k.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)


# 定义多模态MoCo模型
class MultiModalMoCo(nn.Module):
    """
    多模态动量对比（MoCo）模型，用于视觉和触觉模态。
    该模型利用这两种模态之间的对比学习来学习有意义的表示。
    """

    def __init__(self, n_channels=3, m=0.99, T=1.0, nn_model=None, intra_dim=128, inter_dim=128, weight_inter_tv=1,
                 weight_inter_vt=1, weight_intra_vision=1, weight_intra_tactile=1, pretrained_encoder=False):
        """
        初始化MultiModalMoCo模型。

        参数:
        - m: 用于键编码器更新的动量因子
        - T: 对比损失的温度系数
        - nn_model: 神经网络模型类型（如'resnet18', 'resnet50'）
        - intra_dim: 内模表示的维度
        - inter_dim: 跨模表示的维度
        - weight_inter_tv: 视觉和触觉之间的对比损失权重
        - weight_inter_vt: 触觉和视觉之间的对比损失权重
        - weight_intra_vision: 视觉模态内对比损失的权重
        - weight_intra_tactile: 触觉模态内对比损失的权重
        - pretrained_encoder: 是否使用预训练的编码器
        """
        super(MultiModalMoCo, self).__init__()
        self.m = m
        self.T = T
        self.nn_model = nn_model
        self.intra_dim = intra_dim
        self.inter_dim = inter_dim
        self.weight_inter_tac_vis = weight_inter_tv
        self.weight_inter_vis_tac = weight_inter_vt
        self.weight_intra_vision = weight_intra_vision
        self.weight_intra_tactile = weight_intra_tactile
        self.pretrained_encoder = pretrained_encoder

        # 定义 GCN 网络时，不再使用 net_params 字典，
        # 而是在此处预定义一组默认超参数，便于调用：
        gcn_in_dim = 128           # 输入特征维度，根据你的数据设定
        gcn_hidden_dim = 256       # 隐藏层维度
        gcn_out_dim = 256          # 输出维度（图级嵌入维度）
        gcn_n_classes = 10         # 分类任务的类别数（如果不用于分类，可随便设个值）
        gcn_in_feat_dropout = 0.1  # 输入特征 dropout
        gcn_dropout = 0.5          # 每层 dropout
        gcn_n_layers = 3           # GCN 层数
        gcn_readout = "mean"       # 节点聚合方法，可选 "sum", "max", "mean"
        gcn_graph_norm = True      # 是否使用图归一化
        gcn_batch_norm = True      # 是否使用批归一化
        gcn_residual = False       # 是否使用残差连接

        # 直接将这些参数传递给 GCNNet 的构造函数
        self.gnn_encoder = GCNNet(
            gcn_in_dim, gcn_hidden_dim, gcn_out_dim, gcn_n_classes,
            gcn_in_feat_dropout, gcn_dropout, gcn_n_layers, gcn_readout,
            gcn_graph_norm, gcn_batch_norm, gcn_residual
        )

        # 定义视觉模态的编码器
        self.vision_base_q, self.vision_head_intra_q, self.vision_head_inter_q = self.create_encoder(n_channels=3)
        self.vision_base_k, self.vision_head_intra_k, self.vision_head_inter_k = self.create_encoder(n_channels=3)

        # 定义触觉模态的编码器
        self.tactile_base_q, self.tactile_head_intra_q, self.tactile_head_inter_q = self.create_encoder(n_channels)
        self.tactile_base_k, self.tactile_head_intra_k, self.tactile_head_inter_k = self.create_encoder(n_channels)

        # 初始化键编码器权重为查询编码器的权重
        momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
                                    self.vision_head_intra_k, self.m)
        momentum_update_key_encoder(self.tactile_base_q, self.tactile_base_k, self.tactile_head_intra_q,
                                    self.tactile_head_intra_k, self.m)

    def create_encoder(self, n_channels):
        """
        创建编码器（包括基础网络和头部）基于指定的神经网络模型。

        返回:
        - base: 基础编码器（通常是深度CNN）
        - head_intra: 内模头部（用于内模表示）
        - head_inter: 跨模头部（用于跨模表示）
        """

        def create_mlp_head(output_dim):
            """为给定的输出维度创建一个MLP头部"""
            if self.nn_model == 'resnet18':
                return nn.Sequential(
                    nn.Linear(512, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, output_dim)
                )
            elif self.nn_model == 'resnet50':
                return nn.Sequential(
                    nn.Linear(2048, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, output_dim)
                )

        def create_resnet_encoder(n_channels):
            """根据指定的模型类型创建ResNet编码器"""
            if self.nn_model == 'resnet18':
                resnet = models.resnet18(pretrained=self.pretrained_encoder)
            elif self.nn_model == 'resnet50':
                resnet = models.resnet50(pretrained=self.pretrained_encoder)
            if n_channels != 3:  # 如果输入通道不是3，则修改第一层卷积
                resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            features = list(resnet.children())[:-2]
            features.append(nn.AdaptiveAvgPool2d((1, 1)))
            features.append(nn.Flatten())
            return nn.Sequential(*features)

        base = create_resnet_encoder(n_channels)
        head_intra = create_mlp_head(self.intra_dim)
        head_inter = create_mlp_head(self.inter_dim)
        return base, head_intra, head_inter



    def forward(self, x_vision_q, x_vision_k, x_tactile_q, x_tactile_k):
        """
        多模态MoCo模型的前向传播。将输入传入查询和键编码器，并计算对比损失。

        参数:
        - x_vision_q: 视觉模态的查询输入
        - x_vision_k: 视觉模态的键输入
        - x_tactile_q: 触觉模态的查询输入
        - x_tactile_k: 触觉模态的键输入

        返回:
        - combined_loss: 综合对比损失
        """

        def compute_similarity_matrix(embeddings):
            """
            使用 PyTorch 计算余弦相似度矩阵，避免数据从 GPU 移动到 CPU。

            参数：
            - embeddings: (sequence_length, embedding_dim) 的张量，可能位于 GPU 上

            返回：
            - similarity_matrix: (sequence_length, sequence_length) 的张量
            """
            embeddings = F.normalize(embeddings, p=2, dim=1)  # 归一化
            similarity_matrix = torch.mm(embeddings, embeddings.T)  # 计算余弦相似度
            return similarity_matrix

        def generate_graph_from_sequence(seq_embeddings, k_neighbors=5):
            """
            直接在 GPU 上生成 DGL 图，避免数据传输开销。

            参数：
            - seq_embeddings: (sequence_length, embedding_dim) 的张量（应位于 GPU）
            - k_neighbors: 每个节点的邻居数量

            返回：
            - DGLGraph 对象（位于 GPU）
            """
            device = seq_embeddings.device  # 获取 GPU 设备
            seq_len = seq_embeddings.shape[0]

            # 计算余弦相似度
            similarity_matrix = compute_similarity_matrix(seq_embeddings)

            # 初始化邻接矩阵（在 GPU 上）
            adj_matrix = torch.zeros((seq_len, seq_len), device=device)

            # 添加时间序列边
            for i in range(seq_len - 1):
                adj_matrix[i, i + 1] = 1

            # 选择最相似的 k 个邻居
            for i in range(seq_len):
                neighbors = torch.topk(similarity_matrix[i], k_neighbors + 1).indices
                neighbors = neighbors[neighbors != i]  # 去掉自己
                adj_matrix[i, neighbors] = 1

            # 获取边索引
            src, dst = torch.where(adj_matrix == 1)

            # **创建 DGL 图，并直接放在 GPU**
            g = dgl.graph((src, dst), num_nodes=seq_len, device=device)

            # **确保图的特征也在 GPU**
            g.ndata['feat'] = seq_embeddings.float()

            # 获取图中边的数量
            num_edges = g.num_edges()
            # 初始化边特征为全零，假设特征维度为 d（你可以根据实际情况设置）
            edge_feat_dim = 2  # 或其他维度，根据需求
            edge_feat = torch.empty(num_edges, edge_feat_dim, device=device)
            nn.init.xavier_uniform_(edge_feat)
            g.edata['feat'] = edge_feat

            return g

        def graph_contrastive_step(query_graph_list, key_graph_list, model, device, temp):
            """
            单次对比：把 query 和 key 的图列表 collate 成 batched graph，然后做图对比学习
            返回: contrastive_loss, logits, labels
            """
            # 进行图数据增强：drop节点
            # query_graph_list = aug_drop_node_list(query_graph_list, drop_percent=0.2)
            # key_graph_list = aug_drop_node_list(key_graph_list, drop_percent=0.2)

            # 1. 合并图列表 => batched graph
            batch_q, snorm_n_q, snorm_e_q = aug.collate_batched_graph(query_graph_list)
            batch_k, snorm_n_k, snorm_e_k = aug.collate_batched_graph(key_graph_list)

            # 2. 放到 GPU
            batch_q = batch_q.to(device)
            batch_k = batch_k.to(device)
            snorm_n_q = snorm_n_q.to(device)
            snorm_e_q = snorm_e_q.to(device)
            snorm_n_k = snorm_n_k.to(device)
            snorm_e_k = snorm_e_k.to(device)

            # 3. 取出节点特征、边特征
            q_x = batch_q.ndata['feat']
            q_e = batch_q.edata['feat']
            k_x = batch_k.ndata['feat']
            k_e = batch_k.edata['feat']

            # 4. 分别放进GNN进行前向传播，得到图级嵌入
            q_emb = model.forward(batch_q, q_x, q_e, snorm_n_q, snorm_e_q, mlp=False, head=None)
            k_emb = model.forward(batch_k, k_x, k_e, snorm_n_k, snorm_e_k, mlp=False, head=None)

            # 5. 计算相似度矩阵(通过点积计算查询和键的相似度矩阵 sim_matrix，并使用温度系数 T 对其进行缩放)
            sim_matrix = aug.sim_matrix2(q_emb, k_emb, temp=temp)

            # 6. 计算对比损失（行、列方向各做 LogSoftmax 后取对角线之和）
            row_softmax = nn.LogSoftmax(dim=1)
            row_softmax_matrix = -row_softmax(sim_matrix)

            col_softmax = nn.LogSoftmax(dim=0)
            col_softmax_matrix = -col_softmax(sim_matrix)

            row_diag_sum = aug.compute_diag_sum(row_softmax_matrix)
            col_diag_sum = aug.compute_diag_sum(col_softmax_matrix)

            contrastive_loss = (row_diag_sum + col_diag_sum) / (2 * len(row_softmax_matrix))

            # 使用相似度矩阵作为 logits，并构造标签（正样本对应对角线，标签为 0~N-1）
            logits = sim_matrix
            labels = torch.arange(sim_matrix.size(0)).to(device)

            return contrastive_loss, logits, labels
        def aug_drop_node_list(graph_list, drop_percent):

            graph_num = len(graph_list)  # number of graphs
            aug_list = []
            for i in range(graph_num):
                aug_graph = aug_drop_node(graph_list[i], drop_percent)
                aug_list.append(aug_graph)
            return aug_list

        def aug_drop_node(graph, drop_percent=0.2):
            num = graph.number_of_nodes()
            drop_num = int(num * drop_percent)

            all_node_list = list(range(num))
            drop_node_list = set(random.sample(all_node_list, drop_num))  # 选出要删除的节点
            keep_nodes = [n for n in all_node_list if n not in drop_node_list]  # 保留的节点

            # 1. 旧索引 -> 新索引的映射
            node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_nodes)}

            # 2. 获取保留的边，并重新映射索引
            src, dst = graph.edges()
            mask = [(u in keep_nodes and v in keep_nodes) for u, v in zip(src.tolist(), dst.tolist())]
            new_src = [node_mapping[u.item()] for u in src[mask]]
            new_dst = [node_mapping[v.item()] for v in dst[mask]]

            # 3. 重新构造图
            aug_graph = dgl.graph((new_src, new_dst), num_nodes=len(keep_nodes))

            device = graph.device
            aug_graph = aug_graph.to(device)

            # 4. 复制特征
            if 'feat' in graph.ndata:
                aug_graph.ndata['feat'] = graph.ndata['feat'][keep_nodes].clone()
            if 'feat' in graph.edata:
                aug_graph.edata['feat'] = graph.edata['feat'][mask].clone()

            return aug_graph

        # 图片-嵌入向量-graph
        vis_q_graph_intra = []
        vis_q_graph_inter = []
        vis_k_graph_intra = []
        vis_k_graph_inter = []
        tac_q_graph_intra = []
        tac_q_graph_inter = []
        tac_k_graph_intra = []
        tac_k_graph_inter = []

        for i in range(len(x_vision_k)):
            vision_base_q = self.vision_base_q(x_vision_q[i])
            vis_queries_intra = self.vision_head_intra_q(vision_base_q)
            vis_queries_inter = self.vision_head_inter_q(vision_base_q)
            vis_q_graph_intra.append(generate_graph_from_sequence(vis_queries_intra))
            vis_q_graph_inter.append(generate_graph_from_sequence(vis_queries_inter))

            # 对视觉模态的键进行前向传播（使用no_grad来禁止反向传播）
            with torch.no_grad():
                vision_base_k = self.vision_base_k(x_vision_k[i])
                vis_keys_intra = self.vision_head_intra_k(vision_base_k)
                vis_keys_inter = self.vision_head_inter_k(vision_base_k)
                vis_k_graph_intra.append(generate_graph_from_sequence(vis_keys_intra))
                vis_k_graph_inter.append(generate_graph_from_sequence(vis_keys_inter))

            # 对触觉模态进行前向传播
            tactile_base_q = self.tactile_base_q(x_tactile_q[i])# (3,224,224)->(512)
            tac_queries_intra = self.tactile_head_intra_q(tactile_base_q)# (512)->(128)
            tac_queries_inter = self.tactile_head_inter_q(tactile_base_q)
            tac_q_graph_intra.append(generate_graph_from_sequence(tac_queries_intra))
            tac_q_graph_inter.append(generate_graph_from_sequence(tac_queries_inter))

            # 对触觉模态的键进行前向传播（使用no_grad来禁止反向传播）
            with torch.no_grad():
                tactile_base_k = self.tactile_base_k(x_tactile_k[i])
                tac_keys_intra = self.tactile_head_intra_k(tactile_base_k)
                tac_keys_inter = self.tactile_head_inter_k(tactile_base_k)
                tac_k_graph_intra.append(generate_graph_from_sequence(tac_keys_intra))
                tac_k_graph_inter.append(generate_graph_from_sequence(tac_keys_inter))

        # optimizer.zero_grad()

        vis_intra_loss, logits_vis_intra, labels_vis_intra = graph_contrastive_step(
            query_graph_list=vis_q_graph_intra,
            key_graph_list=vis_k_graph_intra,
            model=self.gnn_encoder,
            device=vis_k_graph_intra[0].device,
            temp=self.T
        )

        # (2) 触觉模态 intra 对比
        tac_intra_loss, logits_tact_intra, labels_tac_intra = graph_contrastive_step(
            query_graph_list=tac_q_graph_intra,
            key_graph_list=tac_k_graph_intra,
            model=self.gnn_encoder,
            device=vis_k_graph_intra[0].device,
            temp=self.T
        )

        # (3) 跨模态对比：视觉 query 与触觉 key
        vis_tac_inter_loss, logits_vis_tac_inter, labels_vision_tactile_inter = graph_contrastive_step(
            query_graph_list=vis_q_graph_inter,
            key_graph_list=tac_k_graph_inter,
            model=self.gnn_encoder,
            device=vis_k_graph_intra[0].device,
            temp=self.T
        )

        # (4) 跨模态对比：触觉 query 与视觉 key
        tac_vis_inter_loss, logits_tac_vis_inter, labels_tactile_vision_inter = graph_contrastive_step(
            query_graph_list=tac_q_graph_inter,
            key_graph_list=vis_k_graph_inter,
            model=self.gnn_encoder,
            device=vis_k_graph_intra[0].device,
            temp=self.T
        )

        # 合并所有 loss（可加权求和）
        combined_loss = (
                self.weight_intra_vision * vis_intra_loss
                + self.weight_intra_tactile * tac_intra_loss
                + self.weight_inter_tac_vis * vis_tac_inter_loss
                + self.weight_inter_vis_tac * tac_vis_inter_loss
        )

        # combined_loss.backward()
        # optimizer.step()

        # 动量更新 key encoder
        momentum_update_key_encoder(self.vision_base_q, self.vision_base_k, self.vision_head_intra_q,
                                    self.vision_head_intra_k, self.m)
        momentum_update_key_encoder(self.tactile_base_q, self.tactile_base_k, self.tactile_head_intra_q,
                                    self.tactile_head_intra_k, self.m)

        # 拼接所有对比学习过程得到的 logits 和 labels
        logits = torch.cat([logits_vis_intra, logits_tact_intra, logits_vis_tac_inter, logits_tac_vis_inter], dim=0)
        labels = torch.cat(
            [labels_vis_intra, labels_tac_intra, labels_vision_tactile_inter, labels_tactile_vision_inter], dim=0)
        return combined_loss, vis_intra_loss, tac_intra_loss, vis_tac_inter_loss, tac_vis_inter_loss, logits, labels


