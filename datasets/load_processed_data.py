import os

import torch

from datasets.partition import *
from datasets.processing import *


def get_folders(path):
    folders = []
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            folders.append(file)
    return folders


def load_processed_data(args, data):
    current_path = os.path.abspath(__file__)
    processed_data_path = os.path.join(os.path.dirname(current_path), 'processed_data')

    if args.skew_type == "label_skew":
        cur_split_name = args.task + "_" + args.skew_type + "_" + args.dataset + "_" + str(args.num_clients) \
                        + "_clients_seed_" + str(args.seed) + "_alpha_" + str(args.dirichlet_alpha)
    else:
        cur_split_name = args.task + "_" + args.skew_type + "_" + args.dataset + "_" + str(args.num_clients) \
                        + "_clients_seed_" + str(args.seed)

    cur_split_path = os.path.join(processed_data_path, cur_split_name)

    print("splitting data...")
    if not os.path.exists(cur_split_path):
        if args.evaluation_mode == "global_model_on_independent_data":
            client_data, server_data = split_train_val_test_inductive(data, args.train_val_test_split)
            clients_nodes = []
            if args.skew_type == "label_skew":
                clients_nodes = dirichlet_partitioner(client_data, args.num_clients, args.dirichlet_alpha,
                                                      args.least_samples, args.dirichlet_try_cnt)
            elif args.skew_type == "domain_skew":
                clients_nodes = louvain_partitioner(client_data, args.num_clients)

            clients_data = [get_subgraph_by_node(client_data, clients_nodes[i]) for i in range(len(clients_nodes))]
            clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in
                            range(len(clients_data))]

            if args.dataset_split_metric == "inductive":
                clients_data = [split_data_inductive(data) for data in clients_data]

            server_data.test_mask = torch.ones(server_data.num_nodes, dtype=torch.bool)
            clients_data = clients_data
            server_data = server_data

        else:
            clients_nodes = []
            if args.skew_type == "label_skew":
                clients_nodes = dirichlet_partitioner(data, args.num_clients, args.dirichlet_alpha,
                                                      args.least_samples, args.dirichlet_try_cnt)
            elif args.skew_type == "domain_skew":
                clients_nodes = louvain_partitioner(data, args.num_clients)

            clients_data = [get_subgraph_by_node(data, clients_nodes[i]) for i in range(len(clients_nodes))]
            clients_data = [split_train_val_test(clients_data[i], args.train_val_test_split) for i in
                            range(len(clients_data))]

            if args.dataset_split_metric == "inductive":
                clients_data = [split_data_inductive(data) for data in clients_data]

            server_nodes = []
            for client_data in clients_data:
                for node in range(len(client_data.global_map)):
                    if client_data.test_mask[node]:
                        server_nodes.append(client_data.global_map[node])

            server_data = get_subgraph_by_node(data, server_nodes)
            server_data.test_mask = torch.ones(len(server_nodes), dtype=torch.bool)
            server_data = server_data
            clients_data = clients_data

        os.mkdir(cur_split_path)
        for i, client_data in enumerate(clients_data):
            torch.save(client_data, os.path.join(cur_split_path, "client_" + str(i) + ".pt"))
        torch.save(server_data, os.path.join(cur_split_path, "server.pt"))

        print("data saved as " + cur_split_name)
        return clients_data, server_data
    else:
        print("loading data from /" + cur_split_name)
        clients_data = []
        for i in range(args.num_clients):
            client_data = torch.load(os.path.join(cur_split_path, "client_" + str(i) + ".pt"))
            clients_data.append(client_data)

        server_data = torch.load(os.path.join(cur_split_path, "server.pt"))
        return clients_data, server_data

