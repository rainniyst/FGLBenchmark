from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFGLServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(AdaFGLServer, self).__init__(args, clients, model, data, logger)
        listy = self.data.y.tolist()
        self.num_classes = len(np.unique(listy))

    def run(self):
        for client in self.clients:
            client.num_classes = self.num_classes

        # step 1
        for round in range(self.num_rounds):
            print("round "+str(round)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            avg_train_loss = 0
            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.num_epochs):
                    loss = self.clients[cid].train()
                    avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.local_validate()
            self.local_evaluate()

        # step 2
        print("\n#################start personalized training#################\n")
        for client in self.clients:
            for client_param, server_param in zip(client.model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)
            client.init_adafgl_model()

        for round in range(self.args.adafgl_step2_num_rounds):
            print("round "+str(round+1)+":")
            mean_test_acc = 0
            mean_val_acc = 0
            for client in self.clients:
                loss = client.train_step2()
                weight = client.num_samples / self.num_total_samples
                val_acc, test_acc = client.evaluate()
                mean_val_acc += weight * val_acc
                mean_test_acc += weight * test_acc

            print("mean_val_acc :" + format(mean_val_acc, '.4f'))
            print("mean_client_test_acc :"+format(mean_test_acc, '.4f'))

        print("train finished. final result:")
        for i, client in enumerate(self.clients):
            val_acc, test_acc = client.evaluate()
            print("client "+str(i)+" : val_acc="+format(val_acc, '.4f')+" ,test_acc="+format(test_acc, '.4f'))




class AdaFGLClient(BaseClient):
    def __init__(self, args, model, data):
        super(AdaFGLClient, self).__init__(args, model, data)
        self.num_classes = 0
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.loss_fn1 = nn.CrossEntropyLoss()

    def init_adafgl_model(self):
        self.adafgl_model = AdaFGLModel(prop_steps=self.args.adafgl_prop_steps, feat_dim=self.data.num_node_features, hidden_dim=self.args.hidden_dim, output_dim=self.num_classes,
                                        train_mask=self.data.train_mask, val_mask=self.data.val_mask, test_mask=self.data.test_mask, r=self.args.adafgl_r)
        self.data = adj_initialize(self.data)
        _, out = self.model.forward(self.data)

        embedding = nn.Softmax(dim=1)(out)
        self.adafgl_model.non_para_lp(subgraph=self.data, soft_label=embedding, x=self.data.x,
                                      device=self.device)
        self.adafgl_model.preprocess(adj=self.data.adj)
        self.adafgl_model = self.adafgl_model.to(self.device)
        self.adafgl_optimizer = torch.optim.Adam(self.adafgl_model.parameters(), lr=self.args.adafgl_learning_rate, weight_decay=self.args.adafgl_weight_decay)

    def train_step2(self):
        self.adafgl_model.train()
        self.adafgl_optimizer.zero_grad()
        local_smooth_logits, global_logits = self.adafgl_model.homo_forward(self.device)
        loss_homo_train1 = self.loss_fn1(local_smooth_logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_homo_train2 = nn.MSELoss()(local_smooth_logits, global_logits)
        loss_train_homo = loss_homo_train1 + loss_homo_train2
        local_ori_logits, local_smooth_logits, local_message_propagation = self.adafgl_model.hete_forward(self.device)
        loss_hete_train1 = self.loss_fn1(local_ori_logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_hete_train2 = self.loss_fn1(local_smooth_logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_hete_train3 = self.loss_fn1(local_message_propagation[self.data.train_mask], self.data.y[self.data.train_mask])
        loss_train_hete = loss_hete_train1 + loss_hete_train2 + loss_hete_train3

        loss = loss_train_homo + loss_train_hete
        loss.backward(retain_graph=True)
        self.adafgl_optimizer.step()
        return loss.item()

    def evaluate(self):
        self.adafgl_model.eval()
        with torch.no_grad():
            local_smooth_logits, global_logits = self.adafgl_model.homo_forward(self.device)
            output_homo = (F.softmax(local_smooth_logits.data, 1) + F.softmax(global_logits.data, 1)) / 2

            local_ori_logits, local_smooth_logits, local_message_propagation = self.adafgl_model.hete_forward(self.device)
            output_hete = (F.softmax(local_ori_logits.data, 1) + F.softmax(local_smooth_logits.data, 1) + F.softmax(local_message_propagation.data, 1)) / 3

            homo_weight = self.adafgl_model.reliability_acc
            logits = homo_weight * output_homo + (1 - homo_weight) * output_hete
            pred_val = logits[self.data.val_mask].max(dim=1)[1]
            pred_test = logits[self.data.test_mask].max(dim=1)[1]
        val_acc = pred_val.eq(self.data.y[self.data.val_mask]).sum().item() / self.data.val_mask.sum().item()
        test_acc = pred_test.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
        return val_acc, test_acc

import scipy.sparse as sp
import numpy as np
from torch import Tensor
import platform
from scipy.sparse import csr_matrix
from args import args
import numpy.ctypeslib as ctl
import os.path as osp
import random
from ctypes import c_int
from scipy.sparse import coo_matrix


def adj_initialize(data):
    data.adj = sp.coo_matrix(
        (torch.ones([len(data.edge_index[0])]), (data.edge_index[0].cpu(), data.edge_index[1].cpu())),
        shape=(data.x.shape[0], data.x.shape[0]))
    data.row, data.col, data.edge_weight = data.adj.row, data.adj.col, data.adj.data
    if isinstance(data.adj.row, torch.Tensor) or isinstance(data.adj.col, torch.Tensor):
        data.adj = sp.csr_matrix((data.edge_weight.cpu().numpy(), (data.row.numpy(), data.col.numpy())),
                                 shape=(data.num_nodes, data.num_nodes))
    else:
        data.adj = sp.csr_matrix((data.edge_weight, (data.row, data.col)), shape=(data.num_nodes, data.num_nodes))

    return data

def csr_sparse_dense_matmul(adj, feature):
    file_path = osp.abspath(__file__)
    dir_path = osp.split(file_path)[0]

    ctl_lib = ctl.load_library("./models/csrc/libmatmul.so", dir_path)

    arr_1d_int = ctl.ndpointer(
        dtype=np.int32,
        ndim=1,
        flags="CONTIGUOUS"
    )

    arr_1d_float = ctl.ndpointer(
        dtype=np.float32,
        ndim=1,
        flags="CONTIGUOUS"
    )
    ctl_lib.FloatCSRMulDenseOMP.argtypes = [arr_1d_float, arr_1d_float, arr_1d_int, arr_1d_int, arr_1d_float,
                                            c_int, c_int]
    ctl_lib.FloatCSRMulDenseOMP.restypes = None

    answer = np.zeros(feature.shape).astype(np.float32).flatten()
    data = adj.data.astype(np.float32)
    indices = adj.indices
    indptr = adj.indptr
    mat = feature.flatten()
    mat_row, mat_col = feature.shape

    ctl_lib.FloatCSRMulDenseOMP(answer, data, indices, indptr, mat, mat_row, mat_col)

    return answer.reshape(feature.shape)

class GraphOp:
    def __init__(self, prop_steps):
        self._prop_steps = prop_steps
        self._adj = None

    def construct_adj(self, adj):
        raise NotImplementedError

    def propagate(self, adj, feature):
        self.adj = self.construct_adj(adj)
        if not isinstance(feature, np.ndarray):
            feature = feature.cpu().numpy()
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")


        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self.adj, prop_feat_list[-1])
            else:
                feat_temp = self.adj.dot(prop_feat_list[-1])
            prop_feat_list.append(feat_temp)
        return [torch.FloatTensor(feat) for feat in prop_feat_list]

    def init_lp_propagate(self, adj, feature, init_label, alpha):
        self.adj = self.construct_adj(adj)
        feature = feature.cpu().numpy()
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")


        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self.adj, prop_feat_list[-1])
            else:
                feat_temp = self.adj.dot(prop_feat_list[-1])
            feat_temp = alpha * feat_temp + (1-alpha) * feature
            feat_temp[init_label] += feature[init_label]
            prop_feat_list.append(feat_temp)

        return [torch.FloatTensor(feat) for feat in prop_feat_list]

    def res_lp_propagate(self, adj, feature, alpha):
        self.adj = self.construct_adj(adj)
        if not isinstance(adj, sp.csr_matrix):
            raise TypeError("The adjacency matrix must be a scipy csr sparse matrix!")
        elif not isinstance(feature, np.ndarray):
            raise TypeError("The feature matrix must be a numpy.ndarray!")
        elif self.adj.shape[1] != feature.shape[0]:
            raise ValueError("Dimension mismatch detected for the adjacency and the feature matrix!")


        prop_feat_list = [feature]
        for _ in range(self._prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self.adj, prop_feat_list[-1])
            else:
                feat_temp = self.adj.dot(prop_feat_list[-1])
            feat_temp = alpha * feat_temp + (1-alpha) * feature
            prop_feat_list.append(feat_temp)

        return [torch.FloatTensor(feat) for feat in prop_feat_list]

class MessageOp(nn.Module):
    def __init__(self, start=None, end=None):
        super(MessageOp, self).__init__()
        self._aggr_type = None
        self._start, self._end = start, end

    @property
    def aggr_type(self):
        return self._aggr_type

    def combine(self, feat_list):
        return NotImplementedError

    def aggregate(self, feat_list):
        if not isinstance(feat_list, list):
            return TypeError("The input must be a list consists of feature matrices!")
        for feat in feat_list:
            if not isinstance(feat, Tensor):
                raise TypeError("The feature matrices must be tensors!")

        return self.combine(feat_list)

class LaplacianGraphOp(GraphOp):
    def __init__(self, prop_steps, r=0.5):
        super(LaplacianGraphOp, self).__init__(prop_steps)
        self.r = r

    def construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.r)


        return adj_normalized.tocsr()


class ConcatMessageOp(MessageOp):
    def __init__(self, start, end):
        super(ConcatMessageOp, self).__init__(start, end)
        self._aggr_type = "concat"

    def combine(self, feat_list):
        return torch.hstack(feat_list[self._start:self._end])


def idx_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


class NonParaLP:
    def __init__(self, prop_steps, num_class, alpha, train_mask, val_mask, test_mask, r=0.5):
        self.prop_steps = prop_steps
        self.r = r
        self.num_class = num_class
        self.alpha = alpha
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

        self.graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=self.r)

    def preprocess(self, nodes_embedding, subgraph, device):
        self.subgraph = subgraph
        self.y = subgraph.y
        self.label = F.one_hot(self.y.view(-1), self.num_class).to(torch.float).to(device)
        num_nodes = len(self.train_mask)

        train_idx_list = torch.where(self.train_mask == True)[0].cpu().numpy().tolist()
        num_train = int(len(train_idx_list) / 2)

        random.shuffle(train_idx_list)
        self.lp_train_idx = idx_to_mask(train_idx_list[: num_train], num_nodes)
        self.lp_eval_idx = idx_to_mask(train_idx_list[num_train:], num_nodes)

        unlabel_idx = self.lp_eval_idx | self.val_mask.cpu() | self.test_mask.cpu()
        unlabel_init = torch.full([self.label[unlabel_idx].shape[0], self.label[unlabel_idx].shape[1]],
                                  1 / self.num_class).to(device)
        self.label[self.lp_eval_idx + self.val_mask.cpu() + self.test_mask.cpu()] = unlabel_init

    def propagate(self, adj):
        self.output = self.graph_op.init_lp_propagate(adj, self.label, init_label=self.lp_train_idx, alpha=self.alpha)
        self.output = self.output[-1]

    def eval(self, i=None):
        pred = self.output.max(1)[1].type_as(self.subgraph.y)
        correct = pred[self.lp_eval_idx].eq(self.subgraph.y[self.lp_eval_idx]).double()
        correct = correct.sum()
        reliability_acc = (correct / self.subgraph.y[self.lp_eval_idx].shape[0]).item()
        return reliability_acc


class AdaFGLModel(nn.Module):
    def __init__(self, prop_steps, feat_dim, hidden_dim, output_dim, train_mask, val_mask, test_mask, alpha=0.5, r=0.5):
        super(AdaFGLModel, self).__init__()
        self.prop_steps = prop_steps
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha

        self.NonPLP = NonParaLP(prop_steps=10, num_class=self.output_dim, alpha=alpha, train_mask=train_mask,
                                val_mask=val_mask, test_mask=test_mask, r=r)
        self.pre_graph_op = LaplacianGraphOp(prop_steps=self.prop_steps, r=r)
        self.post_graph_op = LaplacianGraphOp(prop_steps=100, r=r)
        self.post_msg_op = None

    def homo_init(self):
        self.homo_model = HomoPropagateModel(num_layers=2,
                                             feat_dim=self.feat_dim,
                                             hidden_dim=self.hidden_dim,
                                             output_dim=self.output_dim,
                                             dropout=0.5,
                                             prop_steps=self.prop_steps,
                                             bn=False,
                                             ln=False)

        self.total_trainable_params = round(
            sum(p.numel() for p in self.homo_model.parameters() if p.requires_grad) / 1000000, 3)

    def hete_init(self):
        self.hete_model = HetePropagateModel(num_layers=3,
                                             feat_dim=self.feat_dim,
                                             hidden_dim=self.hidden_dim,
                                             output_dim=self.output_dim,
                                             dropout=0.5,
                                             prop_steps=self.prop_steps,
                                             bn=False,
                                             ln=False)

        self.total_trainable_params = round(
            sum(p.numel() for p in self.hete_model.parameters() if p.requires_grad) / 1000000, 3)

    def non_para_lp(self, subgraph, soft_label, x, device):
        self.soft_label = soft_label
        self.ori_feature = x
        self.NonPLP.preprocess(self.soft_label, subgraph, device)
        self.NonPLP.propagate(adj=subgraph.adj)
        self.reliability_acc = self.NonPLP.eval()
        self.homo_init()
        self.hete_init()

    def preprocess(self, adj):
        self.pre_msg_op = ConcatMessageOp(start=0, end=self.prop_steps + 1)
        self.universal_re = getre_scale(self.soft_label)
        self.universal_re_smooth = torch.where(self.universal_re > 0.999, 1, 0)
        self.universal_re = torch.where(self.universal_re > 0.999, 1, 0)
        edge_u = torch.where(self.universal_re_smooth != 0)[0].cpu().numpy()
        edge_v = torch.where(self.universal_re_smooth != 0)[1].cpu().numpy()
        self.universal_re_smooth = np.vstack((edge_u, edge_v))
        universal_re_smooth_adj = sp.coo_matrix((torch.ones([len(self.universal_re_smooth[0])]),
                                                 (self.universal_re_smooth[0], self.universal_re_smooth[1])),
                                                shape=(self.soft_label.shape[0], self.soft_label.shape[0]))
        self.adj = self.alpha * adj + (1 - self.alpha) * universal_re_smooth_adj
        self.adj = self.adj.tocoo()
        row, col, edge_weight = self.adj.row, self.adj.col, self.adj.data
        if isinstance(row, Tensor) or isinstance(col, Tensor):
            self.adj = csr_matrix((edge_weight.numpy(), (row.numpy(), col.numpy())),
                                  shape=(self.soft_label.shape[0], self.soft_label.shape[0]))
        else:
            self.adj = csr_matrix((edge_weight, (row, col)), shape=(self.soft_label.shape[0], self.soft_label.shape[0]))

        self.processed_feat_list = self.pre_graph_op.propagate(self.adj, self.ori_feature)
        self.smoothed_feature = self.pre_msg_op.aggregate(self.processed_feat_list)
        self.processed_feature = self.soft_label

    def homo_forward(self, device):
        local_smooth_logits, global_logits = self.homo_model(
            smoothed_feature=self.smoothed_feature,
            global_logits=self.processed_feature,
            device=device
        )
        return local_smooth_logits, global_logits

    def hete_forward(self, device):
        local_ori_logits, local_smooth_logits, local_message_propagation = self.hete_model(
            ori_feature=self.ori_feature,
            smoothed_feature=self.smoothed_feature,
            processed_feature=self.processed_feature,
            universal_re=self.universal_re,
            device=device)

        return local_ori_logits, local_smooth_logits, local_message_propagation

    def postprocess(self, adj, output):
        if self.post_graph_op is not None:
            output = F.softmax(output, dim=1)
            output = output.detach().numpy()
            output = self.post_graph_op.propagate(adj, output)
            output = self.post_msg_op.aggregate(output)
        return output


class HetePropagateLayer(nn.Module):
    def __init__(self, feat_dim, output_dim, prop_steps, hidden_dim, num_layers, dropout=0.5, beta=0, bn=False,
                 ln=False):
        super(HetePropagateLayer, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.bn = bn
        self.ln = ln
        self.beta = beta

        self.lr_hete_trans = nn.ModuleList()
        self.lr_hete_trans.append(nn.Linear((self.prop_steps + 1) * self.feat_dim, self.hidden_dim))

        for _ in range(num_layers - 2):
            self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_hete_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.norms = nn.ModuleList()
        if self.bn:
            self.norms.append(nn.BatchNorm1d(hidden_dim))
        if self.ln:
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.softmax = nn.Softmax(dim=1)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_hete_tran in self.lr_hete_trans:
            nn.init.xavier_uniform_(lr_hete_tran.weight, gain=gain)
            nn.init.zeros_(lr_hete_tran.bias)

    def forward(self, feature, device, learnable_re=None):

        for i in range(self.num_layers - 1):
            feature = self.lr_hete_trans[i](feature)
            if self.bn is True or self.ln is True:
                feature = self.norms[i](feature)
            feature = self.prelu(feature)
            feature = self.dropout(feature)
        feature_emb = self.lr_hete_trans[-1](feature)

        feature_emb_re = getre_scale(feature_emb)
        learnable_re = self.beta * learnable_re + (1 - self.beta) * feature_emb_re
        learnable_re_mean = torch.mean(learnable_re)
        learnable_re_max = torch.max(learnable_re)

        learnable_re_pos_min = 0
        learnable_re_pos_difference = learnable_re_max - learnable_re_mean - learnable_re_pos_min

        learnable_re_neg_min = -learnable_re_mean
        learnable_re_neg_difference = 0 - learnable_re_neg_min

        learnable_re = learnable_re - learnable_re_mean
        learnable_re = torch.where(learnable_re > 0,
                                   (learnable_re - learnable_re_pos_min) / learnable_re_pos_difference,
                                   -((learnable_re - learnable_re_neg_min) / learnable_re_neg_difference))

        learnable_re = add_diag(learnable_re, device)

        pos_signal = self.prelu(learnable_re)
        neg_signal = self.prelu(-learnable_re)

        prop_pos = self.softmax(torch.mm(pos_signal, feature_emb))
        prop_neg = self.softmax(torch.mm(neg_signal, feature_emb))

        local_message_propagation = ((prop_pos - prop_neg) + feature_emb) / 2

        return local_message_propagation


class HetePropagateModel(nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim, prop_steps, dropout=0.5, bn=False, ln=False):
        super(HetePropagateModel, self).__init__()
        self.num_layers = num_layers
        self.feat_dim = feat_dim
        self.prop_steps = prop_steps
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bn = bn
        self.ln = ln
        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax(dim=1)

        self.lr_smooth_trans = nn.ModuleList()
        self.lr_smooth_trans.append(nn.Linear((self.prop_steps + 1) * self.feat_dim, self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_smooth_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_smooth_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.lr_local_trans = nn.ModuleList()
        self.lr_local_trans.append(nn.Linear(self.feat_dim, self.hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.lr_local_trans.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.hete_propagation = HetePropagateLayer(self.feat_dim, self.output_dim, self.prop_steps, self.hidden_dim,
                                                   self.num_layers)

        self.norms = nn.ModuleList()
        if self.bn:
            if self.num_layers != 1:
                for _ in range(num_layers - 1):
                    self.norms.append(nn.BatchNorm1d(self.hidden_dim))
        if self.ln:
            if self.num_layers != 1:
                for _ in range(num_layers - 1):
                    self.norms.append(nn.LayerNorm(self.hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_local_tran in self.lr_local_trans:
            nn.init.xavier_uniform_(lr_local_tran.weight, gain=gain)
            nn.init.zeros_(lr_local_tran.bias)

        for lr_smooth_tran in self.lr_smooth_trans:
            nn.init.xavier_uniform_(lr_smooth_tran.weight, gain=gain)
            nn.init.zeros_(lr_smooth_tran.bias)

    def forward(self, ori_feature, smoothed_feature, processed_feature, universal_re, device):
        ori_feature = ori_feature.to(device)
        smoothed_feature = smoothed_feature.to(device)
        processed_feature = processed_feature.to(device)

        input_prop_feature = smoothed_feature
        learnable_re = universal_re.to(device)

        for i in range(self.num_layers - 1):
            smoothed_feature = self.lr_smooth_trans[i](smoothed_feature)
            if self.bn is True or self.ln is True:
                smoothed_feature = self.norms[i](smoothed_feature)
            smoothed_feature = self.prelu(smoothed_feature)
            smoothed_feature = self.dropout(smoothed_feature)
        local_smooth_emb = self.lr_smooth_trans[-1](smoothed_feature)

        for i in range(self.num_layers - 1):
            ori_feature = self.lr_local_trans[i](ori_feature)
            if self.bn is True or self.ln is True:
                ori_feature = self.norms[i](ori_feature)
            ori_feature = self.prelu(ori_feature)
            ori_feature = self.dropout(ori_feature)
        local_ori_emb = self.lr_local_trans[-1](ori_feature)

        local_message_propagation = self.hete_propagation(input_prop_feature, device, learnable_re)

        return local_ori_emb, local_smooth_emb, local_message_propagation


class HomoPropagateModel(nn.Module):
    def __init__(self, num_layers, feat_dim, hidden_dim, output_dim, prop_steps, dropout=0.5, bn=False, ln=False):
        super(HomoPropagateModel, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.lr_smooth_trans = nn.ModuleList()
        self.lr_smooth_trans.append(nn.Linear((prop_steps + 1) * feat_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.lr_smooth_trans.append(nn.Linear(hidden_dim, hidden_dim))
        self.lr_smooth_trans.append(nn.Linear(hidden_dim, output_dim))

        self.bn = bn
        self.ln = ln
        self.norms = nn.ModuleList()
        if bn:
            if self.num_layers != 1:
                for _ in range(num_layers - 1):
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
        if ln:
            if self.num_layers != 1:
                for _ in range(num_layers - 1):
                    self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.prelu = nn.PReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        for lr_smooth_tran in self.lr_smooth_trans:
            nn.init.xavier_uniform_(lr_smooth_tran.weight, gain=gain)
            nn.init.zeros_(lr_smooth_tran.bias)

    def forward(self, smoothed_feature, global_logits, device):
        smoothed_feature = smoothed_feature.to(device)
        global_logits = global_logits.to(device)

        for i in range(self.num_layers - 1):
            smoothed_feature = self.lr_smooth_trans[i](smoothed_feature)
            if self.bn is True or self.ln is True:
                smoothed_feature = self.norms[i](smoothed_feature)
            smoothed_feature = self.prelu(smoothed_feature)
            smoothed_feature = self.dropout(smoothed_feature)

        local_smooth_logits = self.lr_smooth_trans[-1](smoothed_feature)

        return local_smooth_logits, global_logits


def getre_scale(emb):
    emb_softmax = nn.Softmax(dim=1)(emb)
    re = torch.mm(emb_softmax, emb_softmax.transpose(0, 1))
    re_self = torch.unsqueeze(torch.diag(re), 1)
    scaling = torch.mm(re_self, torch.transpose(re_self, 0, 1))
    re = re / torch.max(torch.sqrt(scaling), 1e-9 * torch.ones_like(scaling))
    re = re - torch.diag(torch.diag(re))
    return re


def add_diag(re_matrix, device):
    re_diag = torch.diag(re_matrix)
    re_diag_matrix = torch.diag_embed(re_diag)
    re = re_matrix - re_diag_matrix
    re = re_matrix + torch.eye(re.shape[0]).to(device)
    return re

def adj_to_symmetric_norm(adj, r):
    adj = adj + sp.eye(adj.shape[0])
    degrees = np.array(adj.sum(1))
    r_inv_sqrt_left = np.power(degrees, r - 1).flatten()
    r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)

    r_inv_sqrt_right = np.power(degrees, -r).flatten()
    r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)

    adj_normalized = adj.dot(r_mat_inv_sqrt_left).transpose().dot(r_mat_inv_sqrt_right)
    return adj_normalized