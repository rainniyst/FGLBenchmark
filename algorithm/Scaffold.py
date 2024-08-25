from algorithm.Base import BaseServer, BaseClient
import torch


class ScaffoldServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(ScaffoldServer, self).__init__(args, clients, model, data, logger)
        self.global_control = None
        self.local_step = args.scaffold_local_step

    def communicate(self):
        for cid in self.sampled_clients:
            self.clients[cid].global_model = self.model
            self.clients[cid].global_control = self.global_control
            for client_param, server_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                client_param.data.copy_(server_param.data)
            # for client_c_param, server_c_param in zip(self.clients[cid].global_control, self.global_control):
            #     client_c_param.data.copy_(server_c_param.data)

    def aggregate(self):
        num_total_samples = sum([self.clients[cid].num_samples for cid in self.sampled_clients])
        for i, cid in enumerate(self.sampled_clients):
            w = self.clients[cid].num_samples / num_total_samples
            for client_param, global_param in zip(self.clients[cid].model.parameters(), self.model.parameters()):
                if i == 0:
                    global_param.data.copy_(w * client_param)
                else:
                    global_param.data += w * client_param

    def init_controls(self):
        self.global_control = [torch.zeros_like(p.data, requires_grad=False) for p in self.model.parameters()]
        for client in self.clients:
            client.local_control = [torch.zeros_like(p.data, requires_grad=False) for p in client.model.parameters()]

    def update_global_control(self):
        with torch.no_grad():
            for i, client in enumerate(self.clients):
                for it, local_c in enumerate(client.local_control):
                    if i == 0:
                        self.global_control[it].data.copy_(1 / len(self.clients) * local_c)
                    else:
                        self.global_control[it] += 1 / len(self.clients) * local_c

        # for key in self.global_control.keys():
        #     self.global_control[key] += sum(client.local_control[key] - self.global_control[key] for client in
        #                                     self.clients) / len(self.clients)

    def run(self):
        self.init_controls()
        for round in range(self.num_rounds):
            print("round "+str(round)+":")
            self.logger.write_round(round+1)
            self.sample()
            self.communicate()

            print("cid : ", end='')
            for cid in self.sampled_clients:
                print(cid, end=' ')
                for epoch in range(self.num_epochs):
                    self.clients[cid].train()
                self.clients[cid].update_local_control(self.local_step, self.num_epochs)
            self.aggregate()
            self.update_global_control()
            self.local_validate()
            self.global_evaluate()


class ScaffoldClient(BaseClient):
    def __init__(self, args, model, data):
        super(ScaffoldClient, self).__init__(args, model, data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.local_control = None
        self.global_control = None
        self.global_model = None

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data)
        loss = self.loss_fn(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        for p, local_c, global_c in zip(self.model.parameters(), self.local_control, self.global_control):
            if p.grad is None:
                continue
            p.grad.data += global_c - local_c

        self.optimizer.step()
        return loss.item()

    def update_local_control(self, num_epochs, lr):
        with torch.no_grad():
            for it, (local_p, server_p, global_c) in enumerate(
                    zip(self.model.parameters(), self.global_model.parameters(), self.global_control)):
                self.local_control[it].data = self.local_control[it].data - global_c.data + (
                            server_p.data - local_p.data) / (num_epochs * lr)
