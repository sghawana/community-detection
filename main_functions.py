import numpy as np
from collections import defaultdict
from tqdm import tqdm

from helper_functions import *

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

    return graph_partition