import gzip

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

def edge_betweeness(graph):
    ...

