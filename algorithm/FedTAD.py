from algorithm.Base import BaseServer, BaseClient
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, add_self_loops, dense_to_sparse

class FedTADServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedTADServer, self).__init__(args, clients, model, data, logger)
        self.args = args
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        listy = self.data.y.tolist()
        self.num_classes= len(np.unique(listy))
        
        self.generator = FedTAD_ConGenerator(noise_dim=args.fedtad_noise_dim,
                                             input_dim=args.hidden_dim,
                                             output_dim= self.num_classes,
                                             dropout= 0.5).to(self.device)
        

        
        self.global_model_optimizer = Adam(self.model.parameters(),lr=self.args.learning_rate,weight_decay=args.weight_decay)
        self.generator_optimizer = Adam(self.generator.parameters(),lr=self.args.learning_rate,weight_decay=args.weight_decay)

    def run(self):
        for round in range(self.num_rounds):
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.num_epochs):
                    self.clients[cid].train()

            self.aggregate()
            self.train_generator_and_global_model()
            self.local_validate()
            self.global_evaluate()


    def train_generator_and_global_model(self):
        self.global_model_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()

        #这里的ckr第0维度长度为clients的长度，实际上只有被采样的维度有值
        ckr = torch.zeros((len(self.clients), self.num_classes)).to(self.device)
        for c_id in self.sampled_clients:
            graph_emb = cal_topo_emb(edge_index=self.clients[c_id].data.edge_index, num_nodes=self.clients[c_id].data.x.shape[0], max_walk_length=5).to(self.device)    
            ft_emb = torch.cat((self.clients[c_id].data.x, graph_emb), dim=1).to(self.device)
            # 使用train_mask来获取训练节点的索引
            train_indices = torch.nonzero(self.clients[c_id].data.train_mask, as_tuple=False).squeeze()
            for train_i in train_indices:
                neighbor = self.clients[c_id].data.edge_index[1,:][self.clients[c_id].data.edge_index[0, :] == train_i] 
                node_all = 0
                for neighbor_j in neighbor:
                    node_kr = torch.cosine_similarity(ft_emb[train_i], ft_emb[neighbor_j], dim=0)
                    node_all += node_kr
                node_all += 1
                node_all /= (neighbor.shape[0] + 1)
                
                label = self.clients[c_id].data.y[train_i]
                ckr[c_id, label] += node_all
        normalized_ckr = ckr / (ckr.sum(0)+1e-8)   #防止除0 nan

        num_gen = self.args.fedtad_num_gen

        c_cnt = [0] * self.num_classes
        for class_i in range(self.num_classes):
            c_cnt[class_i] = int(num_gen * 1 / self.num_classes)
        c_cnt[-1] += num_gen - sum(c_cnt)

        print(f"pseudo label distribution: {c_cnt}")
        c = torch.zeros(num_gen).to(self.device).long()
        ptr = 0
        for class_i in range(self.num_classes):
            for _ in range(c_cnt[class_i]):
                c[ptr] = class_i
                ptr += 1        

        each_class_idx = {}
        for class_i in range(self.num_classes):
            each_class_idx[class_i] = c == class_i
            each_class_idx[class_i] = each_class_idx[class_i].to(self.device)

        for c_id in self.sampled_clients:
            self.clients[c_id].model.eval()    



        for _ in range(self.args.fedtad_glb_epochs):
            
            ############ sampling noise ##############
            z = torch.randn((num_gen, self.args.fedtad_noise_dim)).to(self.device)
            
            
            ############ train generator ##############
            self.generator.train()
            self.model.eval()
            for it_g in range(self.args.fedtad_it_g):
                loss_sem = 0
                loss_diverg = 0
                loss_div = 0
                
                
                self.generator_optimizer.zero_grad()
                for c_id in self.sampled_clients:
                    ######  generator forward  ########
                    node_logits = self.generator.forward(z=z, c=c) 
                    node_norm = F.normalize(node_logits, p=2, dim=1)
                    adj_logits = torch.mm(node_norm, node_norm.t())
                    pseudo_graph = construct_graph(
                        node_logits, adj_logits, k=self.args.fedtad_topk)
                    
                    ##### local & global model -> forward #########
                    # if args.fedtad_mode == 'rep_distill':
                    local_pred = self.clients[c_id].model.rep_forward(
                            data=pseudo_graph)
                    global_pred = self.model.rep_forward(
                            data=pseudo_graph)

                        
                        
                    ##########  semantic loss  #############
                    acc_list = [0] * self.num_classes
                    for class_i in range(self.num_classes):
                        loss_sem += normalized_ckr[c_id][class_i] * nn.CrossEntropyLoss()(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc = accuracy(local_pred[each_class_idx[class_i]], c[each_class_idx[class_i]])
                        acc_list[class_i] = f"{acc:.2f}"
                    acc_tot = float(accuracy(local_pred, c))
                    # print(f"[client {c_id}] accuracy on each class for pseudo_graph: {acc_list}, on all classes: {acc_tot:.2f}")


                    ############  diversity loss  ##############
                    loss_div += DiversityLoss(metric='l1').to(self.device)(z.view(z.shape[0],-1), node_logits) 
                
                
                    ############  divergence loss  ############   
                    for class_i in range(self.num_classes):
                        loss_diverg += - normalized_ckr[c_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))


                ############ generator loss #############
                loss_G =  loss_sem + loss_diverg +  loss_div              
                # print(f'[generator] loss_sem: {loss_sem:.4f}\tloss_div: {loss_div:.4f}\tloss_diverg: {loss_diverg:.4f}\tloss_G: {loss_G:.4f}')
                
                loss_G.backward()
                self.generator_optimizer.step()



            ########### train global model ###########

            
            self.generator.eval()
            self.model.train()
            
            
            ######  generator forward  ########
            node_logits = self.generator.forward(z=z, c=c)                    
            node_norm = F.normalize(node_logits, p=2, dim=1)
            adj_logits = torch.mm(node_norm, node_norm.t())
            pseudo_graph = construct_graph(node_logits.detach(), adj_logits.detach(), k=self.args.fedtad_topk)
            
            for it_d in range(self.args.fedtad_it_d):
                self.global_model_optimizer.zero_grad()
                loss_D = 0
                
                for c_id in self.sampled_clients:    
                    #######  local & global model -> forward  #######
                    # if args.fedtad_mode == 'rep_distill':
                    local_pred = self.clients[c_id].model.rep_forward(
                            data=pseudo_graph)
                    global_pred = self.model.rep_forward(
                            data=pseudo_graph)

                    
                    ############  divergence loss  ############   
                    
                    for class_i in range(self.num_classes):
                        loss_D += normalized_ckr[c_id][class_i] * torch.mean(torch.mean(
                            torch.abs(global_pred[each_class_idx[class_i]] - local_pred[each_class_idx[class_i]].detach()), dim=1))

                
                
                # print(f"[global] loss_diverg: {loss_D:.4f}")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                loss_D.backward()
                self.global_model_optimizer.step()



class FedTADClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedTADClient, self).__init__(args, model, data)









class FedTAD_ConGenerator(nn.Module):
    def __init__(self, noise_dim, input_dim, output_dim, dropout):
        super(FedTAD_ConGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.emb_layer = nn.Embedding(output_dim, output_dim)
        
        hid_layers = []
        dims = [noise_dim+output_dim, 64, 128, 256]
        for i in range(len(dims)-1):
            d_in = dims[i]
            d_out = dims[i+1]
            hid_layers.append(nn.Linear(d_in, d_out))
            hid_layers.append(nn.Tanh())
            hid_layers.append(nn.Dropout(p=dropout, inplace=False))
        self.hid_layers = nn.Sequential(*hid_layers)
        self.nodes_layer = nn.Linear(256, input_dim)

    def forward(self, z, c):
        z_c = torch.cat((self.emb_layer.forward(c), z), dim=-1)
        hid = self.hid_layers(z_c)
        node_logits = self.nodes_layer(hid)
        return node_logits


def cal_topo_emb(edge_index, num_nodes, max_walk_length):
    A = to_dense_adj(add_self_loops(edge_index)[0], max_num_nodes=num_nodes).squeeze()
    D = torch.diag(torch.sum(A, dim=1))
    T = A * torch.pinverse(D)
    result_each_length = []
    
    for i in range(1, max_walk_length+1):    
        result_per_node = []
        for start in range(num_nodes):
            result_walk = random_walk_with_matrix(T, i, start)
            result_per_node.append(torch.tensor(result_walk).view(1,-1))
        result_each_length.append(torch.vstack(result_per_node))
    topo_emb = torch.hstack(result_each_length)
    return topo_emb    


def random_walk_with_matrix(T, walk_length, start):
    current_node = start
    walk = [current_node]
    for _ in range(walk_length - 1):
        probabilities = F.softmax(T[current_node, :], dim=0)
        probabilities /= torch.sum(probabilities)
        next_node = torch.multinomial(probabilities, 1).item()
        walk.append(next_node)
        current_node = next_node
    return walk

def construct_graph(node_logits, adj_logits, k=5):
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(node_logits.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = add_self_loops(edge_index)[0]
    data = Data(x=node_logits, edge_index=edge_index)
    return data  

def accuracy(pred, ground_truth):
    y_hat = pred.max(1)[1]
    correct = (y_hat == ground_truth).nonzero().shape[0]
    acc = correct / ground_truth.shape[0]
    return acc * 100

class DiversityLoss(nn.Module):
    def __init__(self, metric):
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))