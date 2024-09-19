from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn.functional as F
def compute_fnsc_loss(feature, feature_g, data, tau=0.1):
    labels = data.y
    train_mask = data.train_mask
    feature = feature[train_mask]
    feature_g = feature_g[train_mask]
    labels = labels[train_mask]
    N_train = feature.size(0)
    h_m_norms = torch.norm(feature, dim=1, keepdim=True)
    h_g_norms = torch.norm(feature_g, dim=1, keepdim=True)

    s = torch.mm(feature, feature_g.T)

    norm_matrix = h_m_norms * h_g_norms.T
    norm_matrix = norm_matrix + 1e-8

    s_normalized = (s / norm_matrix) / tau

    log_probs = F.log_softmax(s_normalized, dim=1)

    labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)
    eye_mask = torch.eye(N_train, dtype=torch.bool, device=labels.device)
    positive_mask = labels_equal & ~eye_mask
    positive_mask = positive_mask.float()

    num_pos = positive_mask.sum(dim=1)
    valid_mask = num_pos > 0
    num_pos = num_pos[valid_mask]
    loss_matrix = -positive_mask[valid_mask] * log_probs[valid_mask]

    loss_per_node = loss_matrix.sum(dim=1) / num_pos

    total_loss = loss_per_node.mean()

    return total_loss



def compute_similarity_loss(data, feature, feature_g):

    edge_index = data.edge_index
    src = edge_index[0]
    dst = edge_index[1]

    feature_src = feature[src]  # [num_edges, feature_dim]
    feature_dst = feature[dst]
    feature_g_src = feature_g[src]
    feature_g_dst = feature_g[dst]

    similarity = F.cosine_similarity(feature_src, feature_dst, dim=1)
    similarity_g = F.cosine_similarity(feature_g_src, feature_g_dst, dim=1)

    sigma2 = 1.0
    nll = 0.5 * torch.log(2 * torch.pi * sigma2) + ((similarity - similarity_g) ** 2) / (2 * sigma2)

    loss = nll.sum()

    return loss


class FGSSLClient(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FGSSLClient, self).__init__(args, clients, model, data, logger)

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

        for client in self.clients:
            client.model_g = self.model




class FGSSLClient(BaseClient):
    def __init__(self, args, model, data):
        super(FGSSLClient, self).__init__(args, model, data)
        self.model_g = None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out, feature = self.model(self.data)
        out_g, feature_g = self.model_g(self.data)
        loss_distill = compute_similarity_loss(self.data, feature, feature_g)
        loss_contrast = compute_fnsc_loss(feature, feature_g, self.data, tau=0.1)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss += loss_distill * self.args.distill_loss + loss_contrast

        loss.backward()
        self.optimizer.step()
        return loss.item()



