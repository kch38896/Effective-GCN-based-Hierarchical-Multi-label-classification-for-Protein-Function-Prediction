import torch, gc
import torch.nn as nn
import torch.optim as optim
import Making_Graph
from Utils import FC, Embedder, GCN
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
import argparse

gc.collect()
torch.cuda.empty_cache()


class CustomDataset(Dataset):
    def __init__(self):
        super(CustomDataset, self).__init__()
        self.train_data_file = "dataset/BPO/test_data_bpo.pkl"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.train_data_file, 'rb') as f:
            data = pickle.load(f)
        seq = data["seqvec"].loc[idx]
        seq = np.array(seq)
        seq = np.transpose(seq)
        id = data["proteins"].loc[idx]
        return seq, id

    def __len__(self):
        with open(self.train_data_file, 'rb') as f:
            data = pickle.load(f)
        return len(data)


class Net(nn.Module):
    def __init__(self, n_class, args):
        super().__init__()
        self.FC = FC(1024, args.seqfeat)
        self.Graph_Embedder = Embedder(n_class, args.nfeat)
        self.GCN = GCN(args.nfeat, args.nhid)

    def forward(self, seq, node, adj):
        seq_out = self.FC(seq)
        node_embd = self.Graph_Embedder(node)
        graph_out = self.GCN(node_embd, adj)
        graph_out = graph_out.transpose(-2, -1)
        output = torch.matmul(seq_out, graph_out)
        output = torch.sigmoid(output)
        return output


def test_model(args):
    adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph()

    test_dataset = CustomDataset()
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    model = torch.load("Weights/bpo_final.pth").cuda()
    model = model.eval()

    yy = []
    for i, (input, id) in enumerate(test_loader):
        input = input.squeeze().to(device)
        input = input.type(torch.float32)
        one_hot_node = one_hot_node.to(device)
        adj = adj.to(device)
        preds = model(input, one_hot_node, adj)
        preds = preds.tolist()
        if len(preds) == 1 or i == 67:
            yy.append(preds)
        else:
            for j in range(len(preds)):
                yy.append(preds[j])

    np.save("bpo_preds.npy", yy)
    print("finish")


def main():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--dropout', type=int, default=0.1)

    # Model parameters
    """BPO has 80 parameters, MFO has 41 parameters, and CCO has 54 parameters"""
    parser.add_argument("--nfeat", type=int, default=80, help="node feature size")
    parser.add_argument("--nhid", type=int, default=80, help="GCN node hidden size")
    parser.add_argument("--seqfeat", type=int, default=80, help="sequence reduced feature size")

    args = parser.parse_args()

    test_model(args)


if __name__ == '__main__':
    main()
