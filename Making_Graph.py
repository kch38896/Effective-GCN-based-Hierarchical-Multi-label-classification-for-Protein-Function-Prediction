import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from pathlib import Path


def one_hot_encoding(path, go_index, ldx_map_ivs):
    one_hot_vector = []
    go = {}
    with open(path, "r") as read_in:
        for line in read_in:
            splitted_line = line.strip().split('\t')
            if splitted_line[0] not in go_index.keys():
                continue

            tmp = [0] * (len(go_index))

            if len(splitted_line) == 1:
                index = go_index[splitted_line[0]]
                tmp[index] = 1
            else:
                for i in range(len(splitted_line)):
                    if splitted_line[i] not in go_index.keys():
                        continue
                    index = go_index[splitted_line[i]]
                    tmp[index] = 1
            go[splitted_line[0]] = tmp

    for k in range(len(ldx_map_ivs)):
        tmp_go = ldx_map_ivs[k]
        if tmp_go in go.keys():
            one_hot_vector.append(go[tmp_go])
    one_hot_vector = torch.FloatTensor(one_hot_vector)

    return one_hot_vector


def load_edge_list(path, symmetrize=False):
    df = pd.read_csv(path, header=None, sep='\t', engine='c')
    df.dropna(inplace=True)

    if symmetrize:
        rev = df.copy().rename(columns={0: 1, 1: 0})
        df = pd.concat([df, rev])
    idx, objects = pd.factorize(df[[0, 1]].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    IC = df[2].astype('float')

    return idx, objects.tolist(), IC.tolist()


def load_node_list(path):
    with open(path, "r") as read_line:
        go_id = []
        for line in read_line:
            splitted_line = line.strip().split('\t')
            tmp = [int(splitted_line[0]), int(splitted_line[1])]
            go_id.append(tmp)
    return go_id


def normalize(adj):
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_adj):
    sparse_adj = sparse_adj.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_adj.row, sparse_adj.col)).astype(np.int64))
    values = torch.from_numpy(sparse_adj.data)
    shape = torch.Size(sparse_adj.shape)
    return torch.sparse.LongTensor(indices, values, shape)


def build_adj(idx, IC, idx_map):
    adj_new = []
    for i in range(len(idx_map)):
        tmp = [0] * (len(idx_map))
        adj_new.append(tmp)

    for i in range(len(idx)):
        adj_new[idx[i][0]][idx[i][1]] = IC[i]
        adj_new[idx[i][1]][idx[i][0]] = IC[i]
    adj_new = np.array(adj_new)
    return adj_new


def build_graph():
    data_dir = Path("dataset/BPO")
    edge_list_dir = data_dir / "all_go_bpo_IC.tsv"
    node_list_dir = data_dir / "all_go_bpo_only_num.tsv"
    one_hot_dir = data_dir / "all_go_bpo_parents_only_num.tsv"

    idx, objects, IC = load_edge_list(edge_list_dir)
    idx_2d = idx

    edges = np.array(idx)
    labels = np.array(objects)
    idx = idx.reshape(-1)

    # convert to one-dimensional array
    go_id = load_node_list(node_list_dir)
    go_id = np.array(go_id)
    go_id = go_id.reshape(-1)

    # remove duplicate values
    go_id = pd.unique(go_id).tolist()
    tmp = []
    idx = pd.unique(idx).tolist()
    for i in range(len(go_id)):
        go_id[i] = format(go_id[i], '07')
        tmp.append("GO:%07d" % (int(go_id[i])))

    # making dictionary
    idx_map = dict(zip(go_id, idx))
    label_map = dict(zip(tmp, idx))
    ldx_map_ivs = dict(zip(idx, go_id))
    label_map_ivs = dict(zip(idx, tmp))

    # build symmetric adjacency matrix
    adj = build_adj(idx_2d, IC, idx_map)
    adj = sp.coo_matrix(adj)
    adj = adj + np.multiply(adj.T, adj.T > adj) - np.multiply(adj, (adj.T > adj))
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    one_hot_node = one_hot_encoding(one_hot_dir, idx_map, ldx_map_ivs)

    return adj, one_hot_node, label_map, label_map_ivs

