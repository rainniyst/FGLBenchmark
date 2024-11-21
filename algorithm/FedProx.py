from algorithm.Base import BaseServer, BaseClient
import torch
import copy

class FedProxServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedProxServer, self).__init__(args, clients, model, data, logger)

    def communicate(self):
        for cid in self.sampled_clients:
            self.clients[cid].global_model = copy.deepcopy(self.model)
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)


class FedProxClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedProxClient, self).__init__(args, model, data)
        self.global_model = None
        self.mu = args.fedprox_mu

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        fedprox_reg = 0.0
        for client_param, server_param in zip(self.model.parameters(), self.global_model.parameters()):
            fedprox_reg += ((self.mu / 2) * torch.norm((client_param - server_param)) ** 2)
        loss += fedprox_reg
        loss.backward()
        self.optimizer.step()
        return loss.item()
