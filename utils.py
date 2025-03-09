import os

import torch
import random
import numpy as np
# import wandb
from torch import nn
from torchvision.transforms import transforms
import torch
import dgl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def denormalize(tensor, mean, std):
    """Denormalizes a tensor based on the provided mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def compute_tsne(model, test_dataloader, epoch):
    """Computes t-SNE embeddings and logs them."""
    pass


def find_knn(query_point, data_points, k=5):
    """Finds the k-nearest neighbors of a query point."""
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def generate_graph_from_sequences(seq1_embeddings, seq2_embeddings, k_neighbors=3):
    """
    从两组图像的嵌入向量生成图，体现时间序列关系和图像之间的关联。

    seq1_embeddings, seq2_embeddings: 形状为 (N, embedding_dim) 的张量
        每组图像的嵌入向量
    k_neighbors: 每个节点的邻居数量
    """
    N = seq1_embeddings.shape[0]
    # 将两个序列的嵌入合并成一个大的特征矩阵
    embeddings = torch.cat((seq1_embeddings, seq2_embeddings), dim=0)

    num_nodes = 2 * N  # 2N个节点

    # 计算所有节点之间的余弦相似度
    similarity_matrix = cosine_similarity(embeddings.detach().numpy())

    # 创建邻接矩阵
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # 时间序列边
    for i in range(N - 1):
        adj_matrix[i, i + 1] = 1  # sequence_1 时间序列
        adj_matrix[N + i, N + i + 1] = 1  # sequence_2 时间序列

    # 序列间关联边
    for i in range(N):
        adj_matrix[i, N + i] = 1  # sequence_1 和 sequence_2 之间的关联边

    # 基于相似度的边（比如最近的k个邻居）
    for i in range(num_nodes):
        neighbors = np.argsort(similarity_matrix[i])[-(k_neighbors + 1):-1]
        adj_matrix[i, neighbors] = 1

    # 创建 DGL 图
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)

    # 添加边
    src, dst = np.where(adj_matrix == 1)
    g.add_edges(src.tolist(), dst.tolist())

    # 添加节点特征
    g.ndata['feat'] = embeddings.float()

    return g

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # save both the vision and tactile models
    torch.save(state, filename)

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    加载训练的检查点，包括模型权重、优化器状态和上次的 epoch。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found.")
        return model, optimizer, 0  # 返回初始模型和 optimizer，epoch 从 0 开始

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    start_epoch = checkpoint['epoch'] + 1  # 继续训练
    print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")

    return model, optimizer, start_epoch