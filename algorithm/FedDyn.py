from algorithm.Base import BaseServer, BaseClient
import torch
import copy


class FedDynServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedDynServer, self).__init__(args, clients, model, data, logger)
        self.h = copy.deepcopy(self.model)
        for h_param in self.h.parameters():
            h_param.data.zero_()

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
                self.clients[cid].set_src_model(self.model)
                for epoch in range(self.num_epochs):
                    self.clients[cid].round = round
                    loss = self.clients[cid].train()
                    avg_train_loss += loss * self.clients[cid].num_samples / self.num_total_samples
                self.clients[cid].cal_grad_l()

            print("\n")
            print("avg_train_loss = " + str(avg_train_loss))
            self.aggregate()
            self.local_validate()
            self.local_evaluate()
            self.global_evaluate()

    def aggregate(self):
        for i, cid in enumerate(self.sampled_clients):
            for h_param, client_param, global_param in zip(self.h.parameters(), self.clients[cid].model.parameters(), self.model.parameters()):
                h_param.data = h_param - self.args.feddyn_alpha * 1.0 / len(self.clients) * (client_param - global_param)

        for i, cid in enumerate(self.sampled_clients):
            w = 1.0 / len(self.sampled_clients)
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param
        for h_param, global_param in zip(self.h.parameters(), self.model.parameters()):
            global_param.data = global_param - 1.0 / self.args.feddyn_alpha * h_param


class FedDynClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedDynClient, self).__init__(args, model, data)
        self.grad_l = copy.deepcopy(self.model)
        for grad_l_param in self.grad_l.parameters():
            grad_l_param.data.zero_()
        self.src_model = copy.deepcopy(self.model)
        for src_model_param in self.src_model.parameters():
            src_model_param.data.zero_()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        embedding, out = self.model(self.data)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        self.model.zero_grad()
        l1 = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        l2 = 0
        l3 = 0
        for pgl, pm, ps in zip(self.grad_l.parameters(), self.model.parameters(), self.src_model.parameters()):
            l2 += torch.dot(pgl.view(-1), pm.view(-1))
            l3 += torch.sum(torch.pow(pm - ps, 2))
        loss = l1 - l2 + 0.5 * self.args.feddyn_alpha * l3
        loss.backward()
        optimizer.step()
        return loss.item()

    def set_src_model(self, model):
        self.src_model = copy.deepcopy(model)
        for p in self.src_model.parameters():
            p.requires_grad = False

    def cal_grad_l(self):
        for grad_l_param, src_model_param, client_model_param in zip(self.grad_l.parameters(), self.src_model.parameters(), self.model.parameters()):
            grad_l_param.data = grad_l_param - self.args.feddyn_alpha * (client_model_param - src_model_param)

