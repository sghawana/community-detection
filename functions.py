import gzip
from collections import deque
from tqdm import tqdm

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
                    
                    
def edge_betweeness(graph, edges):
    bw = {e: 0 for e in edges}
    for node in tqdm(graph, desc="Processing nodes"):
        bfs(graph, node, bw)
    return bw


