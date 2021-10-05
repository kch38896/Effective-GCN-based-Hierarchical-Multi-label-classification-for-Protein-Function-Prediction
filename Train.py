import torch, gc
import torch.nn as nn
import argparse
import numpy as np
import pickle
import time
from Utils import FC, Embedder, GCN
import Making_Graph
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

gc.collect()
torch.cuda.empty_cache()

"""To save time and money, we included the results from SeqVec in the training dataset in advance."""

class CustomDataset(Dataset):
    def __init__(self, lable_map):
        super(CustomDataset, self).__init__()
        self.idx_map = lable_map
        self.train_data_file = "dataset/BPO/train_data_bpo.pkl"

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open(self.train_data_file, 'rb') as f:
            data = pickle.load(f)
        seq = data["seqvec"].loc[idx]
        ant = data["annotations"].loc[idx][0]
        ant = ant.split(",")
        cls = [0] * len(self.idx_map)
        for a in ant:
            a = a.strip()
            cls[self.idx_map[a]] = 1
        cls = np.array(cls)
        cls = np.transpose(cls)
        return seq, cls

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


def train_model(args):
    print("training model...")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    # Data load
    adj, one_hot_node, label_map, label_map_ivs = Making_Graph.build_graph()
    tr_dataset = CustomDataset(label_map)
    train_loader = DataLoader(dataset=tr_dataset, batch_size=args.batch_size)

    # model definition
    model = Net(len(label_map), args).to(device)
    print(len(label_map))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()

    total_loss = 0
    print_every = 5
    start = time.time()
    temp = start

    for epoch in range(args.epochs):
        train_loss = 0

        for i, (input, target) in enumerate(train_loader):
            input = torch.stack(input)
            input = input.squeeze().to(device)
            target = target.type(torch.FloatTensor)
            target = target.to(device)
            one_hot_node = one_hot_node.to(device)
            adj = adj.to(device)
            preds = model(input, one_hot_node, adj)

            optimizer.zero_grad()

            criterion = nn.BCELoss()
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            ## record the average training loss, using something like
            train_loss += loss.item()
            batch_idx = i
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = %dm, epoch %d, iter = %d, loss = %.3f,%ds per %d iters" % ((time.time() - start) // 60,
                                                                                         epoch + 1, i + 1, loss_avg,
                                                                                         time.time() - temp,
                                                                                         print_every))
                total_loss = 0
                temp = time.time()

        train_loss = train_loss / batch_idx
        print("batch_idx", i)
        print("train_loss", train_loss)

        # model save
        torch.save({
            'startEpoch': epoch + 1,
            'loss': train_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, os.path.join("Weights/BPO", f'model_{epoch + 1:02d}.pth'))
    torch.save(model, 'Weights/BPO/final.pth')


def main():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model parameters
    """BPO has 80 parameters, MFO has 41 parameters, and CCO has 54 parameters"""
    parser.add_argument("--nfeat", type=int, default=80, help="node feature size")
    parser.add_argument("--nhid", type=int, default=80, help="GCN node hidden size")
    parser.add_argument("--seqfeat", type=int, default=80, help="sequence reduced feature size")

    args = parser.parse_args()

    train_model(args)


if __name__ == '__main__':
    main()
