from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedDCServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedDCServer, self).__init__(args, clients, model, data, logger)

        self.avg_delta = copy.deepcopy(self.model)
        for avg_delta_param in self.avg_delta.parameters():
            avg_delta_param.data.zero_()

    def communicate(self):
        for cid in self.sampled_clients:
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)
            self.clients[cid].server_model = copy.deepcopy(self.model)
            self.clients[cid].avg_delta = copy.deepcopy(self.avg_delta)

    def update_avg_delta(self):
        self.avg_delta = copy.deepcopy(self.model)
        for avg_delta_param in self.avg_delta.parameters():
            avg_delta_param.data.zero_()
        for client in self.clients:
            for avg_delta_param, client_delta_param in zip(self.avg_delta.parameters(), client.drift.parameters()):
                avg_delta_param.data += 1.0 / len(self.sampled_clients) * client_delta_param

    def run(self):
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
                self.clients[cid].update_delta_drift()
            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.update_avg_delta()
            self.local_validate()
            self.local_evaluate()
            self.global_evaluate()

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param, client_drift_param in zip(self.clients[cid].model.parameters(), self.model.parameters(), self.clients[cid].drift.parameters()):
                if i == 0:
                    global_param.data.copy_(w * (client_param + client_drift_param))
                else:
                    global_param.data += w * (client_param + client_drift_param)


class FedDCClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedDCClient, self).__init__(args, model, data)
        self.drift = copy.deepcopy(self.model)
        for drift_param in self.drift.parameters():
            drift_param.data.zero_()
            drift_param.requires_grad = False
        self.delta = copy.deepcopy(self.model)
        for delta_param in self.delta.parameters():
            delta_param.data.zero_()
            delta_param.requires_grad = False
        self.server_model = None
        self.avg_delta = None
        self.round = 0

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        self.model.train()
        self.model.zero_grad()
        l1 = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        l2 = 0
        l3 = 0

        for client_param, server_param, drift_param, delta_param, avg_delta_param in \
                zip(self.model.parameters(), self.server_model.parameters(), self.drift.parameters(), self.delta.parameters(), self.avg_delta.parameters()):
            l2 += torch.sum(torch.pow(drift_param + client_param - server_param, 2))
            l3 += torch.sum(client_param * (delta_param - avg_delta_param))

        if self.round != 0:
            loss = l1 + self.args.feddc_alpha / 2.0 * l2 + 1.0 / self.args.num_epochs * self.args.learning_rate * l3
        else:
            loss = l1
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_delta_drift(self):
        with torch.no_grad():
            for delta_param, drift_param, client_param, server_param in zip(self.delta.parameters(), self.drift.parameters(), self.model.parameters(), self.server_model.parameters()):
                delta_param.data = client_param - server_param
                drift_param.data += delta_param


