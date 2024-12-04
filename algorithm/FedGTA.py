from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn.functional as F
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import degree
import numpy as np
import copy


class FedGTAServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedGTAServer, self).__init__(args, clients, model, data, logger)
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        listy = self.data.y.tolist()
        self.num_classes = len(np.unique(listy))
        self.aggregated_models = [copy.deepcopy(self.model) for _ in range(self.args.num_clients)]
        self.round = 0

    def communicate(self):
        if self.round == 0:
            for client in self.clients:
                for (client_param, global_param) in zip(client.model.parameters(), self.model.parameters()):
                    client_param.data.copy_(global_param)
        else:
            for client_id in self.sampled_clients:
                for (client_param, global_param) in zip(self.clients[client_id].model.parameters(), self.aggregated_models[client_id].parameters()):
                    client_param.data.copy_(global_param)

    def aggregate(self):
        agg_client_list = {}
        for client_id in self.sampled_clients:
            agg_client_list[client_id] = []
            sim = torch.tensor([torch.cosine_similarity(self.clients[client_id].lp_moment_v, self.clients[target_id].lp_moment_v, dim=0)
                                for target_id in self.sampled_clients]).to(self.device)
            accept_idx = torch.where(sim > self.args.fedgta_accept_alpha)
            agg_client_list[client_id] = [self.sampled_clients[idx] for idx in accept_idx[0].tolist()]

        for src, clients_list in agg_client_list.items():
            with torch.no_grad():
                tot_w = [self.clients[client_id].agg_w for client_id in clients_list]
                for it, client_id in enumerate(clients_list):
                    weight = tot_w[it] / sum(tot_w)
                    for (local_param, global_param) in zip(self.clients[client_id].model.parameters(), self.aggregated_models[src].parameters()):
                        if it == 0:
                            global_param.data.copy_(weight * local_param)
                        else:
                            global_param.data += weight * local_param

    def run(self):
        for client in self.clients:
            client.num_classes = self.num_classes
            client.init_onehot_label()

        for round in range(self.num_rounds):
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            avg_train_loss = 0
            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.num_epochs):
                    self.clients[cid].round = round
                    loss = self.clients[cid].train()
                    avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.local_validate()
            self.local_evaluate()
            self.round += 1

        print("\ntrain finished. final local result:")
        self.each_local_evaluate()

    def each_local_evaluate(self):
        for i, client in enumerate(self.clients):
            client.model.eval()
            embedding, out = client.model(client.data)
            pred_test = out[client.data.test_mask].max(dim=1)[1]
            test_acc = pred_test.eq(client.data.y[client.data.test_mask]).sum().item() / client.data.test_mask.sum().item()
            pred_val = out[client.data.val_mask].max(dim=1)[1]
            val_acc = pred_val.eq(client.data.y[client.data.val_mask]).sum().item() / client.data.val_mask.sum().item()
            print("client "+str(i)+" : val_acc="+format(val_acc, '.4f')+" ,test_acc="+format(test_acc, '.4f'))


class FedGTAClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedGTAClient, self).__init__(args, model, data)
        self.num_classes = 0
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.LP = LabelPropagation(num_layers=self.args.fedgta_prop_steps, alpha=self.args.fedgta_lp_alpha)
        self.num_neig = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.long) + degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        self.train_label_onehot = None
        self.lp_moment_v = None
        self.agg_w = None

    def init_onehot_label(self):
        self.train_label_onehot = F.one_hot(self.data.y[self.data.train_mask].view(-1), self.num_classes).to(torch.float).to(self.device)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        # postprocess
        self.model.eval()
        _, logits = self.model.forward(self.data)
        soft_label = F.softmax(logits.detach(), dim=1)
        output = self.LP.forward(y=soft_label, edge_index=self.data.edge_index, mask=self.data.train_mask)
        output_raw = F.softmax(output, dim=1)
        output_dis = F.softmax(output / self.args.fedgta_temperature, dim=1)

        output_raw[self.data.train_mask] = self.train_label_onehot
        output_dis[self.data.train_mask] = self.train_label_onehot
        lp_moment_v = compute_moment(x=output_raw, num_moments=self.args.fedgta_num_moments, dim="v",
                                     moment_type=self.args.fedgta_moment_type)
        self.lp_moment_v = lp_moment_v.view(-1)
        self.agg_w = info_entropy_rev(output_dis, self.num_neig)

        return loss.item()


import math


def info_entropy_rev(vec, num_neig, eps=1e-8):
    return (num_neig.sum()) * vec.shape[1] * math.exp(-1) + torch.sum(torch.multiply(num_neig, torch.sum(torch.multiply(vec, torch.log(vec+eps)), dim=1)))


def raw_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.pow(x, moment)
    return torch.mean(tmp, dim=dim)


def central_moment(x:torch.Tensor, moment, dim=0):
    tmp = torch.mean(x, dim=dim)
    if dim == 0:
        tmp = x - tmp.view(1, -1)
    else:
        tmp = x - tmp.view(-1,1)
    tmp = torch.pow(tmp, moment)
    return  torch.mean(tmp, dim=dim)


def compute_moment(x, num_moments=5, dim="h", moment_type="raw"):
    if moment_type == "raw":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = raw_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "central":
        if dim not in ["h", "v"]:
            raise ValueError
        else:
            if dim == "h":
                dim = 1
            else:
                dim = 0
        moment_type = central_moment
        moment_list = []
        for p in range(num_moments):
            moment_list.append(moment_type(x, moment=p + 1, dim=dim).view(1, -1))
        moment_tensor = torch.cat(moment_list)
        return moment_tensor
    elif moment_type == "hybrid":
        o_ = compute_moment(x, num_moments, dim, moment_type="raw")
        m_ = compute_moment(x, num_moments, dim, moment_type="central")
        return torch.cat((o_, m_))