import gzip
from collections import deque, defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


#------------------------DATASET AND GRAPH HANDELING--------------------------------------------

def build_graph(edge_list):
    graph = {}
    added_edges = set()
    for u, v in edge_list:
        edge = tuple(sorted((u, v)))
        
        if edge not in added_edges:
            added_edges.add(edge)
            
            if u not in graph:
                graph[u] = []
            if v not in graph:
                graph[v] = []
            
            graph[u].append(v)
            graph[v].append(u)
    return graph

def map_node_ids(graph):
    unique_ids = sorted(graph.keys())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    new_graph = {}
    for old_id, neighbors in graph.items():
        new_id = id_mapping[old_id]
        new_graph[new_id] = [id_mapping[neighbor] for neighbor in neighbors]
    
    return new_graph, id_mapping

def revert_node_ids(new_graph, original_id_mapping):
    
    reverse_mapping = {new_id: old_id for old_id, new_id in original_id_mapping.items()}
    original_graph = {}
    
    for new_id, neighbors in new_graph.items():
        old_id = reverse_mapping[new_id]
        original_graph[old_id] = [reverse_mapping[neighbor] for neighbor in neighbors]
    
    return original_graph


# --------------------------------GIRVAN FUNCTIONS---------------------------------------------

def get_edges(g):
    edge_list = []
    for node in g:
        for neighbor in g[node]:
            if node < neighbor:
                edge_list.append((node, neighbor))
    return edge_list


def asq_comp(graph, start_node, visited, comm):
     q = deque([start_node])
     while q:
          node = q.popleft()
          if not visited[node]:
               visited[node] = True
               for v in graph[node]:
                    if not visited[v]:
                         q.append(v)
                    comm[v] = min(v, start_node)

def assign_community(graph):
     visited = {node: False for node in graph}
     comm = {node: node for node in graph}
     for node in graph:
          if not visited[node]:
               asq_comp(graph, node, visited, comm)
     return comm
 
 
                     
def bfs(graph, start_node, bw, undir=True):
     order = []
     parent = {node: set() for node in graph} 
     visited = {node: False for node in graph} 
     dis = {node: float('inf') for node in graph} 
     num_paths = {node: 0 for node in graph} 
     info = {node: 1 for node in graph}
     dis[start_node] = 0; parent[start_node] = -1; num_paths[start_node] = 1
     q = deque([start_node])
     while q:
          node = q.popleft()
          if not visited[node]:
               visited[node] = True
               order.append(node)

          for v in graph[node]:
               if dis[v] == float('inf'):  
                    dis[v] = dis[node] + 1
                    q.append(v)
               
               if dis[v] == dis[node] + 1:  
                    num_paths[v] += num_paths[node]
                    parent[v].add(node)
     
     for node in reversed(order):
          if parent[node] != -1:
               for p in parent[node]:
                    v1 = min(node, p); v2 = max(node, p)
                    if not undir:
                         bw[(v1, v2)] += info[node] * (num_paths[p]/num_paths[node])
                    else:
                         bw[(v1, v2)] += (info[node] * (num_paths[p]/num_paths[node])/2)
                    info[p] += info[node] * (num_paths[p]/num_paths[node])

def edge_betweeness(graph):
    edges = get_edges(graph)
    bw = {e: 0 for e in edges}
    for node in tqdm(graph, desc="Processing nodes"):
        bfs(graph, node, bw)
    return bw


def max_edges(betweeness, delta = 0.01):
    max_edges = []
    max_betweeness = max(betweeness.values())
    for edge in betweeness:
        if abs(betweeness[edge] - max_betweeness) <= delta:
            max_edges.append(edge)
    return max_edges


def remove_edge(graph, edgelist):
    new_graph = {node: neighbors[:] for node, neighbors in graph.items()}
    for edge in edgelist:
        u, v = edge
        if u in new_graph and v in new_graph[u]:
            new_graph[u].remove(v)
        if v in new_graph and u in new_graph[v]:
            new_graph[v].remove(u)
    return new_graph


def modularity(graph, communities):
    m = sum(len(neighbors) for neighbors in graph.values()) / 2
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}
    modularity = 0

    for node_i in graph:
        for node_j in graph:
            if communities[node_i] == communities[node_j]:
                A_ij = 1 if node_j in graph[node_i] else 0
                modularity += A_ij - (degrees[node_i] * degrees[node_j] / (2 * m))
    
    modularity /= (2 * m)
    return modularity


def dict_to_np(dictionary):
    n = len(dictionary)
    numpy_array = np.zeros((n, 1))
    for key, value in dictionary.items():
        numpy_array[key] = value
    return numpy_array

# -------------------------------- LOUVAIN FUNCTIONS---------------------------------------------

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


# --------------------------------DENDOGRAM FUNCTIONS---------------------------------------------


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