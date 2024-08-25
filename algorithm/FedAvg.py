from algorithm.Base import BaseServer, BaseClient


class FedAvgServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedAvgServer, self).__init__(args, clients, model, data, logger)


class FedAvgClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedAvgClient, self).__init__(args, model, data)
