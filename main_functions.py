import numpy as np
from collections import defaultdict
from tqdm import tqdm
import copy
from helper_functions import *

def import_wiki_vote_data(path):
    edges = []
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                continue
            node1, node2 = map(int, line.strip().split())
            edges.append((node1, node2))
    return edges


def import_lastfm_asia_data(path):
    ...


def girvan_newman_one_level(graph):
    edge_list = get_edges(graph)
    if not edge_list:
        return None, None
    bw = edge_betweeness(graph)
    rem_edges= max_edges(bw, delta = 0.0001)
    new_graph = remove_edge(graph, rem_edges)
    new_comm = assign_community(new_graph)
    print('Current Partition:', new_comm, '\n')   
    return new_graph, new_comm

def girvan_newman(graph):
    og_graph = graph
    curr = copy.deepcopy(og_graph)
    community_matrix = dict_to_np(assign_community(curr))
    curr_comm = assign_community(curr)
    curr_mod = modularity(og_graph, curr_comm)
    
    while True:   
        next_graph, next_comm = girvan_newman_one_level(curr)
        next_mod = modularity(og_graph, next_comm)
        if not next_graph:
            break
        elif next_mod + 0.0001 < curr_mod:
            break
        else:
            curr = next_graph
            curr_mod = next_mod
            curr_comm = next_comm
            community_matrix = np.hstack((community_matrix, dict_to_np(curr_comm)))
    
    print('Community Matrix: ', community_matrix )
    return curr, community_matrix


def visualise_dendogram(community_mat):
    Z = calculate_Z(community_mat) 
    plt.figure(figsize=(10, 7))
    dendrogram(Z, labels=np.arange(community_mat.shape[0]))
    
    plt.title("Dendrogram of Girvan-Newman Communities")
    plt.xlabel("Node ID")
    plt.ylabel("Distance")
    plt.savefig("dendrogram.png")
    plt.show()


def louvain_one_iter(graph):
    with tqdm(total=4, desc="Louvain Algorithm Progress") as pbar:
        pbar.set_description("First Phase")
        partition = louvain_phase_one(graph)
        pbar.update(1)

        pbar.set_description("Coalescing Graph")
        coalesced_graph, node_mapping = coalesce_graph(graph, partition)
        pbar.update(1)

        pbar.set_description("Second Phase")
        coalesced_partition = louvain_phase_one(coalesced_graph)
        pbar.update(1)

        pbar.set_description("Finalizing Partition")
        final_partition = {}
        for coalesced_node, new_community in coalesced_partition.items():
            for original_node in node_mapping[coalesced_node]:
                final_partition[original_node] = new_community
        
        n = len(graph)
        graph_partition = np.zeros(n, dtype=int)
        for node, community in final_partition.items():
            graph_partition[node] = community
        pbar.update(1)
    print('Current Partition:', graph_partition, '\n')
    return graph_partition