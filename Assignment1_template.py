
from main_functions import  *

#### ----> Instead of edge list all functions use graph adjacency lists as input<---- ###
if __name__ == "__main__":

    ############ Answer qn 1-4 for wiki-vote data #################################################
    # Import wiki-vote.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_wiki = import_wiki_vote_data("./data/wiki-Vote.txt.gz")
    
    # This functions build a undirected graph from edge list
    graph_wiki = build_graph(nodes_connectivity_list_wiki)
    
    # Since the graph node ids are not in range 0 to n-1 this function remaps the node ids 
    new_graph_wiki, node_id_map_wiki = map_node_ids(graph_wiki)
    
    # To revert back to old graph ids use following command
    '''
    graph_wiki = revert_node_ids(new_graph_wiki, node_id_map_wiki)
    '''
    
    # This is for question no. 1
    # graph_partition: graph_partitition is a nx1 numpy array where the rows corresponds to nodes in the network (0 to n-1) and
    #                  the elements of the array are the community ids of the corressponding nodes.
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    
    _, graph_partition_wiki  = girvan_newman_one_level(new_graph_wiki, True)
    graph_partition_wiki = dict_to_np(graph_partition_wiki)
    # First parameter is new graph with removed edges(if required)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # community_mat is a n x m matrix, where m is the number of levels of Girvan-Newmann algorithm and n is the number of nodes in the network.
    # Columns of the matrix corresponds to the graph_partition which is a nx1 numpy array, as before, corresponding to each level of the algorithm. 
    community_mat_wiki = girvan_newman(new_graph_wiki)

    # This is for question no. 3
    # Visualise dendogram for the communities obtained in question no. 2.
    # Save the dendogram as a .png file in the current directory.
    visualise_dendogram(community_mat_wiki, 'wiki')

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before. Show the resulting communities after one iteration of the algorithm.
    #graph_partition_louvain_wiki = louvain_one_iter(new_graph_wiki)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import lastfm_asia_edges.csv
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_lastfm = import_lastfm_asia_data("data/lasftm_asia/lastfm_asia_edges.csv")
    
    graph_lastfm = build_graph(nodes_connectivity_list_lastfm)
    
    new_graph_lastfm, node_id_map_lastfm = map_node_ids(graph_lastfm)
    # To revert back to old graph ids use following command
    '''
    graph_wiki = revert_node_ids(new_graph_wiki, node_id_map_wiki)
    '''

    # Question 1
    _, graph_partition_lastfm = girvan_newman_one_level(new_graph_lastfm, True)
    graph_partition_lastfm = dict_to_np(graph_partition_lastfm)
    # First parameter is new graph with removed edges(if required)

    # Question 2
    community_mat_lastfm = girvan_newman(new_graph_lastfm)
    

    # Question 3
    visualise_dendogram(community_mat_lastfm, 'lastfm')

    # Question 4
    #graph_partition_louvain_lastfm = louvain_one_iter(new_graph_lastfm)

