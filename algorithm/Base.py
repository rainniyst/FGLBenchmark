import random

import numpy as np
import torch.nn.functional as F
import torch


class BaseServer:
    def __init__(self, args, clients, model, data, logger):
        self.logger = logger
        self.sampled_clients = None
        self.clients = clients
        self.model = model
        self.cl_sample_rate = args.cl_sample_rate
        self.num_rounds = args.num_rounds
        self.num_epochs = args.num_epochs
        self.data = data

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
            self.local_validate()
            self.global_evaluate()

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)

    def sample(self):
        num_sample_clients = int(len(self.clients) * self.cl_sample_rate)
        sampled_clients = random.sample(range(len(self.clients)), num_sample_clients)
        self.sampled_clients = sampled_clients

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def global_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.test_mask], self.data.y[self.data.test_mask])
            pred = out[self.data.test_mask].max(dim=1)[1]
            acc = pred.eq(self.data.y[self.data.test_mask]).sum().item() / self.data.test_mask.sum().item()
            print("test_loss : "+format(loss.item(), '.4f'))
            self.logger.write_test_loss(loss.item())
            print("test_acc : "+format(acc, '.4f'))
            self.logger.write_test_acc(acc)

    def local_validate(self):
        clients_val_loss = []
        clients_val_acc = []
        self.model.eval()
        with torch.no_grad():
            for client in self.clients:
                with torch.no_grad():
                    out = self.model(client.data)
                    loss = F.nll_loss(out[client.data.val_mask], client.data.y[client.data.val_mask])
                    pred = out[client.data.val_mask].max(dim=1)[1]
                    acc = pred.eq(client.data.y[client.data.val_mask]).sum().item() / client.data.val_mask.sum().item()
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


class BaseClient:
    # def __init__(self, args, model, optimizer, data, loss_fn):
    #     self.model = model
    #     self.optimizer = optimizer
    #     self.data = data
    #     self.loss_fn = loss_fn
    #     self.num_samples = len(data)

    def __init__(self, args, model, data):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.data = data
        self.loss_fn = F.nll_loss
        self.num_samples = len(data.x)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
