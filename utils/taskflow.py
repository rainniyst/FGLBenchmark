import torch
from backbone import get_model
from datasets.dataset_loader import load_dataset
from datasets.dataset_ds_loader import load_ds_dataset
from algorithm import get_server, get_client
from datasets.partition import *
from datasets.processing import *
import numpy as np
from utils.logger import DefaultLogger
import time


class TaskFlow:
    def __init__(self, args):
        print(args)

        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        logger = DefaultLogger(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+'-'+args.task+'-'+args.dataset+'-'+args.fed_algorithm, args.logs_dir)
        if args.task == "node_classification":
            if args.skew_type == "label_skew":
                self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)
                input_dim = self.dataset.num_node_features
                listy = self.dataset.y.tolist()
                out_dim = len(np.unique(listy))
                server_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers, args.dropout)
                server_model.to(self.device)

                if args.dataset_split_metric == "transductive":
                    # data = split_train_test(self.dataset[0], args.train_val_test_split)
                    # clients_data = partition(args, data)

                    # clients_data = dirichlet_partitioner(args, self.dataset, args.dirichlet_alpha)
                    clients_nodes = dirichlet_partitioner(self.dataset, args.num_clients, args.dirichlet_alpha, args.least_samples, args.dirichlet_try_cnt)
                    clients_data = [get_subgraph_by_node(self.dataset, clients_nodes[i]) for i in range(len(clients_nodes))]
                    clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in range(len(clients_data))]

                    clients = []
                    for cid in range(args.num_clients):
                        client_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                                 args.dropout)
                        client_model.to(self.device)
                        client_data = clients_data[cid]
                        client_data.to(self.device)
                        client = get_client(args.fed_algorithm, args, client_model, client_data)
                        clients.append(client)

                    server_nodes = []
                    for data in clients_data:
                        for node in range(len(data.global_map)):
                            if data.test_mask[node]:
                                server_nodes.append(data.global_map[node])

                    server_data = get_subgraph_by_node(self.dataset, server_nodes)
                    server_data.test_mask = torch.ones(len(server_nodes), dtype=torch.bool)
                    server_data.to(self.device)
                    self.server = get_server(args.fed_algorithm, args, clients, server_model, server_data, logger)

                elif args.dataset_split_metric == "inductive":
                    test_data, train_val_data = split_train_val_test_inductive(self.dataset, args.train_val_test_split)

                    clients_nodes = dirichlet_partitioner(args, self.dataset, args.dirichlet_alpha, args.least_samples, args.dirichlet_try_cnt)
                    clients_data = [get_subgraph_by_node(self.dataset, clients_nodes[i]) for i in range(len(clients_nodes))]
                    clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in range(len(clients_data))]

                    clients = []
                    for cid in range(args.num_clients):
                        client_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                                 args.dropout)
                        client_model.to(self.device)
                        client_data = clients_data[cid]
                        client_data = split_train_val(client_data, [args.train_val_test_split[0], args.train_val_test_split[1]])
                        client_data.to(self.device)
                        client = get_client(args.fed_algorithm, args, client_model, client_data)
                        clients.append(client)

                    test_data.to(self.device)
                    self.server = get_server(args.fed_algorithm, args, clients, server_model, test_data, logger)

            elif args.skew_type == "domain_skew":
                if args.specified_domain_skew_task is not None:
                    self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)
                    input_dim = self.dataset.num_node_features
                    listy = self.dataset.y.tolist()
                    out_dim = len(np.unique(listy))
                    server_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers, args.dropout)
                    server_model.to(self.device)
                    clients_data = load_ds_dataset(args, self.dataset)
                    # data = split_train_test(self.dataset[0], args.train_val_test_split)
                    # clients_data = partition(args, data)
                    num_clients = len(clients_data)
                    clients = []
                    for cid in range(num_clients):
                        client_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                                 args.dropout)
                        client_model.to(self.device)
                        client_data = clients_data[cid]
                        client_data.to(self.device)
                        client = get_client(args.fed_algorithm, args, client_model, client_data)
                        clients.append(client)

                    self.dataset.to(self.device)
                    self.server = get_server(args.fed_algorithm, args, clients, server_model, self.dataset, logger)
                else:
                    self.dataset = load_dataset(args.train_val_test_split, args.dataset_dir, args.dataset)
                    input_dim = self.dataset.num_node_features
                    listy = self.dataset.y.tolist()
                    out_dim = len(np.unique(listy))
                    server_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                             args.dropout)
                    server_model.to(self.device)
                    if args.dataset_split_metric == "transductive":
                        # data = split_train_test(self.dataset[0], args.train_val_test_split)
                        # clients_data = partition(args, data)
                        # clients_data = louvain_partitioner(args, self.dataset)
                        clients_data = louvain_partitioner(self.dataset, args.num_clients)

                        clients = []
                        for cid in range(args.num_clients):
                            client_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                                     args.dropout)
                            client_model.to(self.device)
                            client_data = clients_data[cid]
                            client_data.to(self.device)
                            client = get_client(args.fed_algorithm, args, client_model, client_data)
                            clients.append(client)

                        self.dataset.to(self.device)
                        self.server = get_server(args.fed_algorithm, args, clients, server_model, self.dataset, logger)

                    elif args.dataset_split_metric == "inductive":
                        test_data, train_val_data = split_train_val_test_inductive(self.dataset,
                                                                                   args.train_val_test_split)

                        clients_data = louvain_partitioner(args, self.dataset)

                        clients = []
                        for cid in range(args.num_clients):
                            client_model = get_model(args.model, input_dim, args.hidden_dim, out_dim, args.num_layers,
                                                     args.dropout)
                            client_model.to(self.device)
                            client_data = clients_data[cid]
                            client_data = split_train_val(client_data,
                                                          [args.train_val_test_split[0], args.train_val_test_split[1]])
                            client_data.to(self.device)
                            client = get_client(args.fed_algorithm, args, client_model, client_data)
                            clients.append(client)

                        test_data.to(self.device)
                        self.server = get_server(args.fed_algorithm, args, clients, server_model, test_data, logger)
        elif args.task == "graph_classification":
            return

    def run(self):
        self.server.run()


