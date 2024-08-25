from community import community_louvain
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch_geometric.data import Data


# def partition(args, data):
#     if args.partitioner == "louvain":
#         return louvain_partitioner(data, args.num_clients)
#     elif args.partitioner == "dirichlet":
#         return dirichlet_partitioner(data, args.num_clients, args.dirichlet_alpha)


def louvain_partitioner(data, num_clients):
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])
    # 将节点属性添加到图中
    for i, feature in enumerate(data.x.numpy()):
        G.nodes[i]['feature'] = feature

    # nx.draw(G)
    # import matplotlib.pyplot as plt
    # plt.show()

    # 将标签添加到图中
    le = LabelEncoder()
    labels = le.fit_transform(data.y.numpy())
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    partition = community_louvain.best_partition(G)

    # 图可视化
    # import matplotlib.pyplot as plt
    # pos = nx.kamada_kawai_layout(G)  # 使用spring布局进行可视化
    #
    # # 绘制节点
    # cmap = plt.get_cmap('viridis')
    # for community_id in set(partition.values()):
    #     nodes = [node for node in partition if partition[node] == community_id]
    #     nx.draw_networkx_nodes(G, pos, nodes, node_size=50,
    #                            node_color=[cmap(community_id / len(set(partition.values())))])
    # # 绘制边
    # nx.draw_networkx_edges(G, pos, alpha=0.5)
    # plt.show()

    # 将节点分配到不同的社区中
    community_dict = {}
    for node, community in partition.items():
        if community not in community_dict:
            community_dict[community] = []
        community_dict[community].append(node)

    num_communities = len(community_dict)

    clients_nodes = [[] for _ in range(num_clients)]

    while len(community_dict) < num_communities:
        community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
        max_len = max(community_sizes)
        min_len = min(community_sizes)
        max_index = np.argmax(community_sizes)
        if max_len < 2 * min_len:
            min_len = max_len // 2
        max_len_nodes = community_dict[max_index]
        new_list_id = len(community_dict)
        community_dict[new_list_id] = max_len_nodes[:min_len]
        community_dict[max_index] = max_len_nodes[min_len:]
    community_sizes = [len(community_dict[i]) for i in range(len(community_dict))]
    community_ids = np.argsort(community_sizes)
    for comid in community_ids:
        clid = np.argmin([len(cs) for cs in clients_nodes])
        clients_nodes[clid].extend(community_dict[comid])

    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}
        # 提取子图的边
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()

        # for i in range(data.edge_index.size(1)):
        #     if data.edge_index[0, i].item() in nodes and data.edge_index[1, i].item() in nodes:
        #         sub_edge_index.append(data.edge_index[:, i].view(2, 1))
        # sub_edge_index = torch.cat(sub_edge_index, dim=1)

        # 提取子图的特征和标签
        sub_x = data.x[nodes]
        sub_y = data.y[nodes]
        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

        if hasattr(data, "train_mask"):
            train_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                train_mask[node_map[node]] = data.train_mask[node]
            sub_data.train_mask = train_mask

        if hasattr(data, "val_mask"):
            val_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                val_mask[node_map[node]] = data.val_mask[node]
            sub_data.val_mask = val_mask

        if hasattr(data, "test_mask"):
            test_mask = torch.zeros(len(nodes), dtype=torch.bool)
            for node in nodes:
                test_mask[node_map[node]] = data.test_mask[node]
            sub_data.test_mask = test_mask

        clients_data.append(sub_data)

    import matplotlib.pyplot as plt
    listy = data.y.tolist()
    num_classes = len(np.unique(listy))
    label_distribution = [[] for _ in range(num_classes)]
    for cid, client_data in enumerate(clients_data):
        for label in client_data.y:
            label_distribution[label].append(cid)
    plt.hist(label_distribution, stacked=True, label=range(num_classes))
    plt.xlabel("client_id")
    plt.ylabel("num_samples")
    plt.show()

    return clients_data


def dirichlet_partitioner(data, num_clients, alpha):
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    edges = data.edge_index.numpy()
    for i in range(edges.shape[1]):
        G.add_edge(edges[0, i], edges[1, i])
    # 将节点属性添加到图中
    for i, feature in enumerate(data.x.numpy()):
        G.nodes[i]['feature'] = feature

    # 将标签添加到图中
    le = LabelEncoder()
    labels = le.fit_transform(data.y.numpy())
    for i, label in enumerate(labels):
        G.nodes[i]['label'] = label

    labels = nx.get_node_attributes(G, 'label')
    unique_labels = list(set(labels.values()))
    num_classes = len(unique_labels)

    label_distribution = np.random.dirichlet([alpha]*num_clients, num_classes)
    class_indices = [[] for _ in range(num_classes)]
    for i in range(data.num_nodes):
        class_indices[data.y[i]].append(i)

    clients_nodes = [[] for _ in range(num_clients)]
    for k_idcs, fracs in zip(class_indices, label_distribution):
        for i, idcs in enumerate(np.split(k_idcs, (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            clients_nodes[i].extend(idcs)

    # # 初始化子图列表
    # subgraph_nodes = [[] for _ in range(num_clients)]
    #
    # # 根据每个标签生成 Dirichlet 分布样本并分配节点
    # for label in unique_labels:
    #     nodes_with_label = [node for node, node_label in labels.items() if node_label == label]
    #     num_nodes = len(nodes_with_label)
    #
    #     dirichlet_samples = np.random.dirichlet([alpha] * num_clients, num_nodes)
    #     node_assignments = np.argmax(dirichlet_samples, axis=1)
    #
    #     for i, node in enumerate(nodes_with_label):
    #         subgraph_nodes[node_assignments[i]].append(node)
    #
    # clients_nodes = subgraph_nodes

    # clients_nodes = [[] for _ in range(num_clients)]
    # for i in range(len(node_assignments)):
    #     clients_nodes[node_assignments[i]].append(i)

    # subgraphs = [G.subgraph(np.where(node_assignments == i)[0]).copy() for i in range(num_clients)]
    # clients_data = []
    # for subgragh in subgraphs:
    #     clients_data.append(from_networkx(subgragh))

    # data_x = data.x.tolist()
    # data_x = [[int(a) for a in l_x] for l_x in data_x]
    # data_y = data.y.tolist()

    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}
        # 提取子图的边
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()

        # sub_x = []
        # sub_y = []
        # # 提取子图的特征和标签
        # for node in nodes:
        #     sub_x.append(data_x[node])
        #     sub_y.append(data_y[node])
        sub_x = data.x[nodes]
        sub_y = data.y[nodes]

        sub_data = Data(x=sub_x, edge_index=sub_edge_index, y=sub_y)

        if hasattr(data, "train_mask"):
            train_mask = torch.zeros(len(nodes), dtype=torch.bool)
            train_mask1 = data.train_mask.tolist()
            for node in nodes:
                # train_mask[node_map[node]] = data.train_mask[node].item()
                train_mask[node_map[node]] = train_mask1[node]
            sub_data.train_mask = train_mask

        if hasattr(data, "val_mask"):
            val_mask = torch.zeros(len(nodes), dtype=torch.bool)
            val_mask1 = data.val_mask.tolist()
            for node in nodes:
                # test_mask[node_map[node]] = data.test_mask[node].item()
                val_mask[node_map[node]] = val_mask1[node]
            sub_data.val_mask = val_mask

        if hasattr(data, "test_mask"):
            test_mask = torch.zeros(len(nodes), dtype=torch.bool)
            test_mask1 = data.test_mask.tolist()
            for node in nodes:
                # test_mask[node_map[node]] = data.test_mask[node].item()
                test_mask[node_map[node]] = test_mask1[node]
            sub_data.test_mask = test_mask

        clients_data.append(sub_data)

    import matplotlib.pyplot as plt
    label_distribution = [[] for _ in range(num_classes)]
    for cid, client_data in enumerate(clients_data):
        for label in client_data.y:
            label_distribution[label].append(cid)
    plt.hist(label_distribution, stacked=True, label=range(num_classes))
    plt.xlabel("client_id")
    plt.ylabel("num_samples")
    plt.show()

    return clients_data

