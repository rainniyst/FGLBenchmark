import argparse
import os

supported_datasets = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'computers', 'photo',
                        'texas', 'wisconsin', 'cornell', 'squirrel', 'chameleon', 'crocodile', 'actor',
                        'twitch', 'fb100', 'Penn94', 'deezer', 'year', 'snap-patents', 'pokec', 'yelpchi', 'gamer',
                        'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'genius',
                        'roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']

supported_specified_task = ['cora']

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="cora")
current_path = os.path.abspath(__file__)
dataset_path = os.path.join(os.path.dirname(current_path), 'datasets')
root_dir = os.path.join(dataset_path, 'raw_data')
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
parser.add_argument("--dataset_dir", type=str, default=root_dir)

log_path = os.path.join(os.path.dirname(current_path), 'logs')
if not os.path.exists(log_path):
    os.makedirs(log_path)
parser.add_argument("--logs_dir", type=str, default=log_path)

# parser.add_argument("--specified_task", type=str)
# parser.add_argument("--specified_domain_skew_task", type=str, default="ogbn_arxiv_years")
parser.add_argument("--specified_domain_skew_task", type=str, default=None)
parser.add_argument("--task", type=str, default="node_classification")
parser.add_argument("--skew_type", type=str, default="domain_skew")
# parser.add_argument("--train_val_test_split", type=list, default=[0.5, 0.25, 0.25])
# parser.add_argument("--train_val_test_split", type=list, default=[0.09, 0.3, 0.61])

parser.add_argument("--train_val_test_split", type=list, default=[0.2, 0.4, 0.4])
parser.add_argument("--dataset_split_metric", type=str, default="transductive")

parser.add_argument("--num_rounds", type=int, default=50)
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--cl_sample_rate", type=float, default=1.0)
parser.add_argument("--evaluation_mode", type=str, default="global")

parser.add_argument("--fed_algorithm", type=str, default="FGSSL")

parser.add_argument("--model", type=str, default="GCN")
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--weight_decay", type=float, default=4e-4)

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device_id", type=int, default=0)

# parser.add_argument("--partitioner", type=str, default="dirichlet")
# parser.add_argument("--partitioner", type=str, default="louvain")
parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
parser.add_argument("--least_samples", type=int, default=5)
parser.add_argument("--dirichlet_try_cnt", type=int, default=1000)

parser.add_argument("--fedprox_mu", type=float, default=0.01)

parser.add_argument("--scaffold_local_step", type=float, default=0.1)

parser.add_argument("--fedproto_lambda", type=float, default=1.0)

parser.add_argument("--moon_temperature", type=float, default=0.3)
parser.add_argument("--moon_mu", type=float, default=1.0)

parser.add_argument("--feddyn_alpha", type=float, default=1.0)

parser.add_argument("--fednova_eta", type=float, default=1.0)

parser.add_argument("--feddc_alpha", type=float, default=1.0)
# for fedtad
parser.add_argument("--fedtad_noise_dim", type=int, default=32)
parser.add_argument("--fedtad_num_gen", type=int, default=100)
parser.add_argument("--fedtad_glb_epochs", type=int, default=5)
parser.add_argument('--fedtad_it_g', type=int, default=1)
parser.add_argument('--fedtad_it_d', type=int, default=5)
parser.add_argument('--fedtad_topk', type=int, default=5)
parser.add_argument("--fedtad_lam1", type=float, default=1.0)
parser.add_argument("--fedtad_lam2", type=float, default=1.0)
parser.add_argument("--fedtad_distill_mode", type=str, default="rep_distill")

parser.add_argument("--adafgl_alpha", type=float, default=0.05)
parser.add_argument("--adafgl_beta", type=float, default=0.2)
parser.add_argument("--adafgl_step2_num_rounds", type=int, default=50)
parser.add_argument("--adafgl_learning_rate", type=float, default=0.01)
parser.add_argument("--adafgl_weight_decay", type=float, default=4e-4)
parser.add_argument("--adafgl_prop_steps", type=int, default=3)
parser.add_argument("--adafgl_r", type=float, default=0.5)
parser.add_argument("--adafgl_threshold", type=float, default=0.5)


# for FGSSL
parser.add_argument("--fgssl_distill_loss_weight", type=int, default=0.5)
parser.add_argument("--fgssl_contrastive_loss_weight", type=int, default=0.5)


#for FGGP
parser.add_argument("--FGGP_contrastive", type=int, default=0.1)
parser.add_argument("--FGGP_regu", type=int, default=2)


args = parser.parse_args()
