import copy

from algorithm.Base import BaseServer, BaseClient
import torch


class MOONServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(MOONServer, self).__init__(args, clients, model, data, logger)

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
                    self.clients[cid].set_global_net()
                    self.clients[cid].round = round
                    loss = self.clients[cid].train()
                    avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples
                    self.clients[cid].set_prev_net()
            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.local_validate()
            self.local_evaluate()
            self.global_evaluate()


class MOONClient(BaseClient):
    def __init__(self, args, model, data):
        super(MOONClient, self).__init__(args, model, data)
        self.temperature = self.args.moon_temperature
        self.mu = self.args.moon_mu
        self.round = 0
        self.prev_net = None
        self.global_net = None
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")

    def set_prev_net(self):
        self.prev_net = copy.deepcopy(self.model)

    def set_global_net(self):
        self.global_net = copy.deepcopy(self.model)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)
        cos = torch.nn.CosineSimilarity(dim=-1)
        if self.round == 0:
            loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        else:
            z_prev, _ = self.prev_net(self.data)
            z_global, _ = self.global_net(self.data)
            sim_global = cos(embedding[self.data.train_mask], z_global[self.data.train_mask]).view(-1, 1)
            sim_prev = cos(embedding[self.data.train_mask], z_prev[self.data.train_mask]).view(-1, 1)
            logits = torch.cat((sim_global, sim_prev), dim=1) / self.args.moon_temperature
            labels = torch.zeros(embedding[self.data.train_mask].size(0)).to(self.device).long()
            contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
            moon_loss = self.args.moon_mu * contrastive_loss
            loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask]) + moon_loss
        loss.backward()
        self.optimizer.step()
        return loss.item()


