from algorithm.Base import BaseServer, BaseClient
from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F

import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj


def prototype_contrastive_and_regularization_loss(data_features, knn_features, adaptive_features,
                                                  data_logits, knn_logits, adaptive_logits,
                                                  labels, train_mask, data, data_argu, temperature=0.1, beta=1.0,
                                                  gamma=0.1):
    def compute_prototypes(features, logits, labels, train_mask):
        num_classes = logits.size(1)
        prototypes = []
        for class_idx in range(num_classes):
            mask = train_mask & (labels == class_idx)
            conf = torch.where(mask, torch.ones_like(logits[:, 0]), torch.max(logits, dim=1).values)
            class_features = features[labels == class_idx]
            class_conf = conf[labels == class_idx]
            if class_conf.sum() > 0:
                proto = torch.sum(class_features * class_conf.unsqueeze(1), dim=0) / class_conf.sum()
            else:
                proto = torch.zeros(features.size(1), device=features.device)
            prototypes.append(proto)
        return torch.stack(prototypes, dim=0)

    def compute_contrastive_loss(proto1, proto2, temperature):
        sim_matrix = F.cosine_similarity(proto1.unsqueeze(1), proto2.unsqueeze(0), dim=2)
        exp_sim = torch.exp(sim_matrix / temperature)
        loss = 0
        for i in range(proto1.size(0)):
            pos_sim = exp_sim[i, i]
            neg_sim_1 = exp_sim[i].sum() - pos_sim
            neg_sim_2 = exp_sim[:, i].sum() - pos_sim
            loss += -torch.log(pos_sim / (neg_sim_1 + 1e-8)) - torch.log(pos_sim / (neg_sim_2 + 1e-8))
        return loss / (2 * proto1.size(0))

    def compute_regularization_loss(data, data_argu):
        adj_matrix = to_dense_adj(data.edge_index)[0]
        adj_matrix_argu = to_dense_adj(data_argu.edge_index)[0]
        adj_matrix_argu = torch.clamp(adj_matrix_argu, min=1e-8, max=1 - 1e-8)  # 防止log(0)的问题
        reg_loss = - (adj_matrix * torch.log(adj_matrix_argu) + (1 - adj_matrix) * torch.log(1 - adj_matrix_argu))
        return reg_loss.mean()

    data_prototypes = compute_prototypes(data_features, data_logits, labels, train_mask)
    knn_prototypes = compute_prototypes(knn_features, knn_logits, labels, train_mask)
    adaptive_prototypes = compute_prototypes(adaptive_features, adaptive_logits, labels, train_mask)

    contrastive_loss_data_knn = compute_contrastive_loss(data_prototypes, knn_prototypes, temperature)
    contrastive_loss_data_adaptive = compute_contrastive_loss(data_prototypes, adaptive_prototypes, temperature)
    contrastive_loss_knn_adaptive = compute_contrastive_loss(knn_prototypes, adaptive_prototypes, temperature)

    total_contrastive_loss = (
                                         contrastive_loss_data_knn + contrastive_loss_data_adaptive + contrastive_loss_knn_adaptive) / 3

    regularization_loss = compute_regularization_loss(data, data_argu)

    total_loss = beta * total_contrastive_loss + gamma * regularization_loss

    return total_loss


def adaptive_graph_topology(data, feature, tau=0.1):
    similarity_matrix = torch.mm(feature, feature.t())
    Omega = torch.sigmoid(similarity_matrix)
    adj_matrix = to_dense_adj(data.edge_index, max_num_nodes=feature.size(0))[0]
    B = 0.5 * (adj_matrix + Omega)
    gumbel_noise = -torch.empty_like(B).exponential_().log()
    Ag = torch.sigmoid((torch.log(B) + gumbel_noise) / tau)
    Ag_discrete = (Ag > 0.5).float()
    new_edge_index, _ = dense_to_sparse(Ag_discrete)
    new_data = torch_geometric.data.Data(x=data.x, edge_index=new_edge_index, y=data.y)

    return new_data


def modify_adjacency_matrix(data, feature, K):
    edge_index = data.edge_index  # 原始的边信息 (i.e. A_ij)

    adj_matrix = to_dense_adj(edge_index, max_num_nodes=feature.size(0))[0]  # [N, N]

    feature_norm = F.normalize(feature, p=2, dim=1)  # [N, F]

    # 计算余弦相似度
    similarity_matrix = torch.mm(feature_norm, feature_norm.t())  # [N, N]

    # Step 3: 构建新的邻接矩阵 A'
    N = adj_matrix.size(0)
    new_adj_matrix = adj_matrix.clone()  # 复制原始邻接矩阵

    for i in range(N):
        # 如果 i != j 并且 j 是 TopK(i) 的邻居，则 A'_ij = 1
        topk_neighbors = torch.topk(similarity_matrix[i], K + 1, largest=True).indices  # 包括自己
        topk_neighbors = topk_neighbors[topk_neighbors != i]  # 去除自己

        # 更新邻接矩阵
        new_adj_matrix[i, topk_neighbors] = 1  # 添加 K 近邻

    new_edge_index, _ = dense_to_sparse(new_adj_matrix)

    new_data = torch_geometric.data.Data(x=data.x, edge_index=new_edge_index, y=data.y)

    return new_data


def cluster_prototypes_by_label(client_prototypes_list, num_clusters):

    global_memory_bank = {}

    for prototypes in client_prototypes_list:
        for label, prototype in prototypes.items():
            if label not in global_memory_bank:
                global_memory_bank[label] = []
            global_memory_bank[label].append(prototype)

    clustered_memory_bank = {}
    for label, features in global_memory_bank.items():
        if len(features) > 1:
            kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
            centroids = kmeans.cluster_centers_
            clustered_memory_bank[label] = [(centroid, label) for centroid in centroids]
        else:
            clustered_memory_bank[label] = [(features[0], label)]

    return clustered_memory_bank

def compute_prototypes(features, labels, train_mask, confidence):
    unique_labels = np.unique(labels[train_mask])
    prototypes = {}

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        labeled_features = features[indices]
        if len(labeled_features) > 0:
            prototypes[label] = np.mean(labeled_features, axis=0)

    for label in unique_labels:
        unlabeled_indices = np.where(labels != label)[0]
        weights = confidence[unlabeled_indices]  # 取无标签节点的置信度
        weighted_features = features[unlabeled_indices] * weights[:, np.newaxis]
        weighted_sum = np.sum(weighted_features[labels[unlabeled_indices] == label], axis=0)
        total_weight = np.sum(weights[labels[unlabeled_indices] == label])

        if total_weight > 0:
            prototypes[label] = (prototypes.get(label, np.zeros(features.shape[1])) + weighted_sum) / (1 + total_weight)

    return prototypes
class FGGPClient(BaseClient):
    def __init__(self, args, model, data):
        super(FGGPClient, self).__init__(args, model, data)
        self.model_g = None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out, features, confidence = self.model(self.data)
        new_data1 = modify_adjacency_matrix(self.data, features, 3)
        new_data2 = adaptive_graph_topology(self.data, features, tau=0.1)
        _, knn_features, confidence1 = self.model(self.new_data1)
        _, adaptive_features, confidence2 = self.model(self.new_data2)
        self.prototypes = compute_prototypes(features, self.data.y, self.data.train_mask, confidence)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_GKIC = prototype_contrastive_and_regularization_loss(features, knn_features, adaptive_features,
                                                      confidence, confidence1, confidence2,
                                                      self.data.y, self.data.train_mask, self.data, new_data2, 0.1, self.args.FGGP_contrastive,
                                                      self.args.FGGP_regu)
        loss += loss_GKIC
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def local_validate(self):
        clients_val_loss = []
        clients_val_acc = []
        self.model.eval()
        with torch.no_grad():
            for client in self.clients:
                with torch.no_grad():
                    out, features, confidence = self.model(client.data)
                    loss = F.nll_loss(out[client.data.val_mask], client.data.y[client.data.val_mask])
                    pred = out[client.data.val_mask].max(dim=1)[1]

                    all_centroids = []
                    labels = []
                    for label, centroids in self.global_memory_bank.items():
                        all_centroids.extend(centroids)
                        labels.extend([label] * len(centroids))
                    all_centroids = torch.stack(all_centroids).cuda()
                    features = features.cuda()
                    similarities = F.cosine_similarity(features.unsqueeze(1), all_centroids.unsqueeze(0), dim=2)
                    topk_similarities, topk_indices = torch.topk(similarities, 3, dim=1)

                    preds = []
                    for i in range(features.size(0)):
                        topk_labels = [labels[idx.item()] for idx in topk_indices[i]]

                        weighted_counts = {}
                        for label, similarity in zip(topk_labels, topk_similarities[i]):
                            if label not in weighted_counts:
                                weighted_counts[label] = 0
                            weighted_counts[label] += similarity.item()

                        pred_label = max(weighted_counts, key=weighted_counts.get)
                        preds.append(pred_label)

                    acc = preds.eq(client.data.y[client.data.val_mask]).sum().item() / client.data.val_mask.sum().item()
                    clients_val_loss.append(loss.item())
                    clients_val_acc.append(acc)
        mean_val_loss = np.mean(clients_val_loss)
        std_val_loss = np.std(clients_val_loss)
        mean_val_acc = np.mean(clients_val_acc)
        std_val_acc = np.std(clients_val_acc)
        print("mean_val_loss :"+format(mean_val_loss, '.4f'))
        self.logger.write_mean_val_loss(mean_val_loss)
        print("std_val_loss :"+format(std_val_loss, '.4f'))
        print("mean_val_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)
        print("std_val_acc :"+format(std_val_acc, '.4f'))


class FGGPServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FGGPServer, self).__init__(args, clients, model, data, logger)

    def aggregate(self):
        self.prototype_list = []
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            self.prototype_list.append(self.clients[cid].prototypes)
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param
        self.global_memory_bank = cluster_prototypes_by_label(self.prototype_list, 2)

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

        for client in self.clients:
            client.global_memory_bank = self.global_memory_bank

