import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

@torch.no_grad()
def bipartite_perfect_matching(source, destination, weight):
    # compute weight of the edge e_mn: weight_m(source_m-destination_n)^2
    edge_weights = torch.zeros([len(source),len(destination)]).to(device)
    for i in range(len(source)):
        for j in range(len(destination)):
            edge_weights[i,j] = weight[i].item()*((source[i].item()-destination[j].item())**2)
    # build graph
    G = nx.Graph()
    top_nodes = range(len(source))
    bottom_nodes = range(len(source), len(source)+len(destination))
    G.add_nodes_from(top_nodes, bipartite=0)
    G.add_nodes_from(bottom_nodes, bipartite=1)
    # add edges
    for i in range(len(source)):
        for j in range(len(destination)):
            G.add_edge(i, len(source)+j, weight=edge_weights[i,j].item())
    # min weight perfect matching
    matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes, 'weight')  # matching is a dict
    # read sequence from matching result
    sequence = torch.zeros(len(source), dtype=int).to(device)
    for i in range(len(source)):
        sequence[i] = matching[i] - len(source)
    return sequence