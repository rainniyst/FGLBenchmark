from algorithm.Base import BaseServer, BaseClient
import torch


class FedGTAServer(BaseServer):
    def __init__(self, args, clients, model, data, logger):
        super(FedGTAServer, self).__init__(args, clients, model, data, logger)

class FedGTAClient(BaseClient):
    def __init__(self, args, model, data):
        super(FedGTAClient, self).__init__(args, model, data)