import numpy as np
from scipy.spatial import distance
from scipy.stats.mstats import gmean
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


def operator_norm_pruning(weight):
    c_out, c_in, h, w = weight.shape
    weight = weight.reshape(c_out, c_in, -1)
    C_M = []
    mean_vec = []
    for c in range(np.shape(weight)[1]):
        A = weight[:, c, :]
        A_mean = np.mean(A, 0)
        e = np.tile(A_mean, (np.shape(A)[0], 1))
        A_centred = A - e
        mean_vec.append(A_mean)
        u, q, v = np.linalg.svd(A_centred)
        u1 = np.reshape(u[:, 0], (np.shape(A)[0], 1))
        v1 = np.reshape(v[0, :], (np.shape(A)[1], 1))
        c_1 = np.matmul(u1, v1.T)
        c_1_norm = c_1[0, :] / np.linalg.norm(c_1[0, :])
        C_M.append(c_1_norm)
    Score = []
    for Nf in range(np.shape(weight)[0]):
        Score.append(np.trace(
            np.matmul(
                (weight[Nf, :, :] - np.array(mean_vec)),
                np.array(C_M).T
                )
            )
        )
    Mse_score = (np.array(Score)) ** 2
    Mse_score_norm = Mse_score / np.max(Mse_score)
    idx = np.argsort(Mse_score_norm)
    return idx


def rank1_apporx(data):
    u, w, v = np.linalg.svd(data)
    M = np.matmul(np.reshape(u[:, 0], (-1, 1)), np.reshape(v[0, :], (1, -1)))
    M_prototype = M[:, 0] / np.linalg.norm(M[:, 0], 2)
    return M_prototype


def cs_interspeech(Z):
    d, c, a, b = np.shape(Z)
    # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    A = np.reshape(Z, (d, c, -1)).transpose(2, 1, 0) # reshape filters
    N = np.zeros((a * b, d))

    for i in range(d):
        data = A[:, :, i]
        N[:,i] = rank1_apporx(data)

    W = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            W[i, j] = W[i, j] + distance.cosine(N[:, i], N[:, j])

    Q = []
    S = []
    for i in range(np.shape(W)[0]):
        n = np.argsort(W[i,:])[1]
        Q.append([i, n, W[i, n]])  # store closest pairs with their distance.
        S.append(W[i, n])   # store closest distance for each filter (ordered pairwise distance)

    Q_sort = []
    q = list(np.argsort(S)) # save the indexes of filters with closest pairwise distance.

    for i in q:
        Q_sort.append(Q[i]) # sort closest filter pairs.

    imp_list = []
    red_list = []

    for i in range(np.shape(W)[0]):
        index_imp = Q_sort[i][0]
        index_red = Q_sort[i][1]
        if index_imp not in red_list:
            imp_list.append(index_imp)
            red_list.append(index_red)

    return imp_list

#% entry-wise l_1 norm based scores
def iclr_l1(W):
    Score = []
    for Nf in range(np.shape(W)[0]):
        Score.append(np.sum(np.abs(W[Nf, :, 0])))
    score = Score / np.max(Score)
    return np.argsort(score)

#% Geometric median based scores
def iclr_gm(W):
    G_GM = gmean(np.abs(W.flatten()))
    Diff = []
    for Nf in range(np.shape(W)[0]):
        F_GM = gmean(np.abs(W[Nf, :, :]).flatten())
        Diff.append((G_GM - F_GM) ** 2)
    score = Diff / np.max(Diff)
    return np.argsort(score)


def filter_pruning_using_ranked_weighted_degree(filters,
                                                ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)
    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))

    for i in range(num_filters):
        for j in range(i + 1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])

    # Compute weighted degree centrality
    weighted_degree_centrality = {node: sum([G.edges[node, neighbor]['weight'] for neighbor in G.neighbors(node)]) for node in G.nodes()}

    # Rank the nodes
    ranked_nodes = sorted(weighted_degree_centrality.items(), key=lambda x: x[1], reverse=not ascending)

    idxs = [node for node, centrality in ranked_nodes[::-1]]

    return idxs


def filter_pruning_using_ranked_betweenness(filters, ascending=False):
    num_filters, channel_size = filters.shape
    cosine_similarities = cosine_similarity(filters)

    # Create a graph
    G = nx.Graph()
    G.add_nodes_from(range(num_filters))

    for i in range(num_filters):
        for j in range(i + 1, num_filters):
            G.add_edge(i, j, weight=cosine_similarities[i, j])

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

    # Rank the nodes
    ranked_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=not ascending)

    # TODO whether to keep the highest or lowest ranked nodes
    idxs = [node for node, centrality in ranked_nodes[::-1]]

    return idxs


def cs_waspaa(Z):
    d, c, a, b = np.shape(Z) # "d" is number of filters, "c" is number of channels, "a" and "b" represent length of the filter.
    A = np.reshape(Z, (d, c, -1)).transpose(2, 1, 0) # reshape filters
    N = np.zeros((a * b, d))
    for i in range(d):
        data = A[:, :, i]
        N[:, i] = rank1_apporx(data)
    # It expects filling W or Z directly.
    idxs = filter_pruning_using_ranked_weighted_degree(N.T, d)
    return idxs
