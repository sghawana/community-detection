import numpy as np
from helper_functions import  import_wiki_vote_data, Girvan_Newman_one_level, Girvan_Newman, visualise_dendogram, louvain_one_iter, import_lastfm_asia_data

if __name__ == "__main__":

    ############ Answer qn 1-4 for wiki-vote data #################################################
    # Import wiki-vote.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_wiki = import_wiki_vote_data("../data/Wiki-Vote.txt")

    # This is for question no. 1
    # graph_partition: graph_partitition is a nx1 numpy array where the rows corresponds to nodes in the network (0 to n-1) and
    #                  the elements of the array are the community ids of the corressponding nodes.
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    graph_partition_wiki  = Girvan_Newman_one_level(nodes_connectivity_list_wiki)

    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # community_mat is a n x m matrix, where m is the number of levels of Girvan-Newmann algorithm and n is the number of nodes in the network.
    # Columns of the matrix corresponds to the graph_partition which is a nx1 numpy array, as before, corresponding to each level of the algorithm. 
    community_mat_wiki = Girvan_Newman(nodes_connectivity_list_wiki)

    # This is for question no. 3
    # Visualise dendogram for the communities obtained in question no. 2.
    # Save the dendogram as a .png file in the current directory.
    visualise_dendogram(community_mat_wiki)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before. Show the resulting communities after one iteration of the algorithm.
    graph_partition_louvain_wiki = louvain_one_iter(nodes_connectivity_list_wiki)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import lastfm_asia_edges.csv
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is an edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_lastfm = import_lastfm_asia_data("../data/lastfm_asia_edges.csv")

    # Question 1
    graph_partition_lastfm = Girvan_Newman_one_level(nodes_connectivity_list_lastfm)

    # Question 2
    community_mat_lastfm = Girvan_Newman(nodes_connectivity_list_lastfm)

    # Question 3
    visualise_dendogram(community_mat_lastfm)

    # Question 4
    graph_partition_louvain_lastfm = louvain_one_iter(nodes_connectivity_list_lastfm)

