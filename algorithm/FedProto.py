import copy

import torch.nn.functional as F
from algorithm.Base import BaseServer, BaseClient
import numpy as np
import torch


class FedProtoServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedProtoServer, self).__init__(args, clients, model, data, logger)
        listy = self.data.y.tolist()
        out_dim = len(np.unique(listy))
        self.num_classes = out_dim
        self.global_prototype = None
        self.fedproto_lambda = args.fedproto_lambda
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.global_label_distribution = [0 for _ in range(self.num_classes)]
        for client in self.clients:
            client.num_classes = self.num_classes
            client.get_label_distribution()

        for client in self.clients:
            for class_id in range(self.num_classes):
                self.global_label_distribution[class_id] += client.label_distribution[class_id]

        self.num_total_samples = sum([client.num_samples for client in self.clients])

    def run(self):
        self.init_global_prototype()

        for round in range(self.num_rounds):
            print("round "+str(round+1)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            avg_train_loss = 0
            loss = 0
            print("cid : ", end='')
            for cid in self.sampled_clients:
                self.clients[cid].round = round
                self.clients[cid].update_local_prototype()
                print(cid, end=' ')
                for epoch in range(self.num_epochs):
                    loss = self.clients[cid].train()
                avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples
            print("\n")
            print("avg_train_loss = "+str(avg_train_loss))
            self.update_global_prototype()
            self.local_validate()
            self.local_evaluate()

    def communicate(self):
        for client in self.clients:
            client.global_prototype = self.global_prototype

    def init_global_prototype(self):
        for client in self.clients:
            client.update_local_prototype()
        self.global_prototype = [torch.zeros(self.args.hidden_dim).to(self.device) for _ in range(len(self.clients))]
        self.update_global_prototype()

    def update_global_prototype(self):
        with torch.no_grad():
            self.global_prototype = [torch.zeros(self.args.hidden_dim).to(self.device) for _ in range(len(self.clients))]
            for class_id in range(self.num_classes):
                for i, client in enumerate(self.clients):
                    weight = client.label_distribution[class_id] / self.global_label_distribution[class_id]
                    # weight = client.num_samples / self.num_total_samples
                    # if i == 0:
                    #     self.global_prototype[class_id] = weight * copy.deepcopy(client.local_prototype[class_id])
                    # else:
                    self.global_prototype[class_id] += weight * (client.local_prototype[class_id])
    def local_evaluate(self):
        clients_test_loss = []
        clients_test_acc = []
        self.model.eval()
        mean_test_acc = 0
        std_test_acc = 0
        with torch.no_grad():
            for client in self.clients:
                with torch.no_grad():
                    dict1 = client.model.state_dict()
                    weight = client.num_samples / self.num_total_samples
                    embedding, out = client.model(client.data)
                    loss = F.nll_loss(out[client.data.test_mask], client.data.y[client.data.test_mask])
                    pred = out[client.data.test_mask].max(dim=1)[1]
                    acc = pred.eq(client.data.y[client.data.test_mask]).sum().item() / client.data.test_mask.sum().item()
                    clients_test_loss.append(loss.item())
                    clients_test_acc.append(acc)
                    mean_test_acc += weight * acc
        mean_test_loss = np.mean(clients_test_loss)
        std_test_loss = np.std(clients_test_loss)
        # mean_test_acc = np.mean(clients_test_acc)
        std_test_acc = np.std(clients_test_acc)
        # print("mean_client_test_loss :"+format(mean_test_loss, '.4f'))
        # self.logger.write_mean_val_loss(mean_test_loss)
        # print("std_client_test_loss :"+format(std_test_loss, '.4f'))
        print("mean_client_test_acc :"+format(mean_test_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_test_acc)
        print("std_client_test_acc :"+format(std_test_acc, '.4f'))

    def local_validate(self):
        clients_val_loss = []
        clients_val_acc = []
        self.model.eval()
        mean_val_acc = 0
        with torch.no_grad():
            for client in self.clients:
                with torch.no_grad():
                    weight = client.num_samples / self.num_total_samples
                    embedding, out = client.model(client.data)
                    loss = F.nll_loss(out[client.data.val_mask], client.data.y[client.data.val_mask])
                    pred = out[client.data.val_mask].max(dim=1)[1]
                    acc = pred.eq(client.data.y[client.data.val_mask]).sum().item() / client.data.val_mask.sum().item()
                    clients_val_loss.append(loss.item())
                    clients_val_acc.append(acc)
                    mean_val_acc += weight * acc
        mean_val_loss = np.mean(clients_val_loss)
        std_val_loss = np.std(clients_val_loss)
        # mean_val_acc = np.mean(clients_val_acc)
        std_val_acc = np.std(clients_val_acc)
        # print("mean_val_loss :"+format(mean_val_loss, '.4f'))
        # self.logger.write_mean_val_loss(mean_val_loss)
        # print("std_val_loss :"+format(std_val_loss, '.4f'))
        print("mean_val_acc :"+format(mean_val_acc, '.4f'))
        self.logger.write_mean_val_acc(mean_val_acc)
        print("std_val_acc :"+format(std_val_acc, '.4f'))


class FedProtoClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedProtoClient, self).__init__(args, model, data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.label_distribution = []
        self.global_prototype = None
        self.local_prototype = None
        self.num_classes = None
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.round = 0

    def get_label_distribution(self):
        self.label_distribution = [0 for _ in range(self.num_classes)]

        listy = self.data.y.tolist()
        for node, label in enumerate(listy):
            self.label_distribution[label] += 1
        self.local_prototype = [torch.zeros(self.args.hidden_dim).to(self.device) for _ in range(self.num_classes)]

    def update_local_prototype(self):
        with torch.no_grad():
            embedding, out = self.model.forward(self.data)
            for class_id in range(self.num_classes):
                nodes = torch.logical_and(self.data.train_mask, self.data.y == class_id)
                if nodes.sum() == 0:
                    self.local_prototype[class_id] = torch.zeros(self.args.hidden_dim).to(self.device)
                else:
                    class_nodes_embedding = embedding[nodes]
                    self.local_prototype[class_id] = torch.mean(class_nodes_embedding, dim=0).to(self.device)

    def get_fedproto_loss_fn(self):
        def loss_fn(embedding, out, label, mask):
            fedproto_loss = 0
            for class_id in range(self.num_classes):
                nodes = torch.logical_and(mask, label == class_id)
                if nodes.sum() == 0:
                    continue
                class_nodes_embedding = embedding[nodes]
                target = self.global_prototype[class_id].expand_as(class_nodes_embedding)
                fedproto_loss += torch.nn.MSELoss()(class_nodes_embedding, target)
            loss = self.loss_fn(out[mask], label[mask]) + self.args.fedproto_lambda * fedproto_loss
            return loss
        return loss_fn

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        embedding, out = self.model(self.data)
        fedproto_loss_fn = self.get_fedproto_loss_fn()
        loss = fedproto_loss_fn(embedding, out, self.data.y, self.data.train_mask)

        loss.backward()

        self.optimizer.step()
        return loss.item()
