import gzip
from collections import deque, defaultdict
from tqdm import tqdm
import numpy as np

def import_wiki_vote_data(path):
    edges = []
    with gzip.open(path, 'rt', encoding='utf-8') as file:
        for line in file:
            if line.startswith('#'):
                continue
            node1, node2 = map(int, line.strip().split())
            edges.append((node1, node2))
    return edges


def build_wiki_graph(edge_list):
    uniq_nodes = set()
    for a,b in edge_list:
        uniq_nodes.add(a)
        uniq_nodes.add(b)
    graph = {node: [] for node in uniq_nodes}
    for a, b in edge_list:
        graph[a].append(b)
    return graph

# -------------------------------- Louvain Functions ---------------------------------------------

def initialize_partition(graph):
    return {node: node for node in graph}

def calculate_modularity(graph, partition):
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    q = 0
    for i in graph:
        for j in graph[i]:
            if partition[i] == partition[j]:
                q += 1 - (len(graph[i]) * len(graph[j])) / (2 * m)
    return q / (2 * m)

def calculate_modularity_gain(node, community, graph, partition, m):
    k_i = len(graph[node])
    k_i_in = sum(1 for neighbor in graph[node] if partition[neighbor] == community)
    sigma_tot = sum(len(graph[n]) for n in graph if partition[n] == community)
    return (k_i_in - sigma_tot * k_i / (2 * m))

def find_best_community(node, graph, partition, m):
    current_community = partition[node]
    best_community = current_community
    best_gain = 0
    for neighbor in graph[node]:
        neighbor_community = partition[neighbor]
        if neighbor_community != current_community:
            gain = calculate_modularity_gain(node, neighbor_community, graph, partition, m)
            if gain > best_gain:
                best_gain = gain
                best_community = neighbor_community
    return best_community

def assign_smallest_node_id(partition):
    community_to_nodes = defaultdict(list)
    for node, community in partition.items():
        community_to_nodes[community].append(node)
    
    new_partition = {}
    for community, nodes in community_to_nodes.items():
        new_community_id = min(nodes)
        for node in nodes:
            new_partition[node] = new_community_id
    
    return new_partition

def louvain_phase_one(graph):
    partition = initialize_partition(graph)
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    improvement = True
    
    while improvement:
        improvement = False
        for node in tqdm(graph):
            best_community = find_best_community(node, graph, partition, m)
            if best_community != partition[node]:
                old_partition = partition.copy()
                partition[node] = best_community
                if calculate_modularity(graph, partition) > calculate_modularity(graph, old_partition):
                    improvement = True
                else:
                    partition = old_partition
    
    return assign_smallest_node_id(partition)

def coalesce_graph(graph, partition):
    new_graph = defaultdict(lambda: defaultdict(int))
    node_to_community = defaultdict(list)
    
    for node, community in partition.items():
        node_to_community[community].append(node)
    
    for node, neighbors in graph.items():
        comm_i = partition[node]
        for neighbor in neighbors:
            comm_j = partition[neighbor]
            new_graph[comm_i][comm_j] += 1
    
    return dict(new_graph), node_to_community


# -------------------------------- Dendogram Functions ---------------------------------------------


def hamming_distance(x, y):
    return np.sum(x != y) / len(x)

def calculate_Z(community_mat):
    n = community_mat.shape[0]
    clusters = {i: [i] for i in range(n)}
    distances = np.full((n, n), np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            distances[i, j] = hamming_distance(community_mat[i], community_mat[j])
            distances[j, i] = distances[i, j]
    
    Z = []
    next_cluster = n

    while len(clusters) > 1:
        min_dist = np.inf
        for i in clusters:
            for j in clusters:
                if i < j and distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    merge_pair = (i, j)
        
        i, j = merge_pair
        Z.append([i, j, min_dist, len(clusters[i]) + len(clusters[j])])
        clusters[next_cluster] = clusters[i] + clusters[j]
        new_distances = np.full((next_cluster + 1, next_cluster + 1), np.inf)
        new_distances[:next_cluster, :next_cluster] = distances
        distances = new_distances 
        del clusters[i]
        del clusters[j]
        
        for k in clusters:
            if k != next_cluster:
                dist = min(distances[i, k], distances[j, k])
                distances[next_cluster, k] = dist
                distances[k, next_cluster] = dist
        
        next_cluster += 1
    return np.array(Z)