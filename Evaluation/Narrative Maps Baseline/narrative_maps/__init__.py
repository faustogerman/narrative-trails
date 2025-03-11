import math
from math import ceil, exp, pi, sqrt
from time import time
from urllib.parse import urlparse

import networkx as nx
import numpy as np
import pandas as pd
from pulp import *


# Computing Similarity Tables
def compute_temp_distance_table(query, dataset):
    n = len(query.index)
    filename = dataset + '_temp.npy'
    if os.path.isfile(filename):
        # print("Temporal distance table already exists.")
        temporal_distance_table = np.load(filename)
        if n == temporal_distance_table.shape[0]:  # Must match, otherwise regenerate.
            return temporal_distance_table

    temporal_distance_table = np.zeros((n, n))
    for i in range(0, n - 1):
        for j in range(i, n):
            # We are using days as the basic temporal unit.
            temporal_distance_table[i, j] = (query.iloc[j]['date'] - query.iloc[i]['date']).days
    temporal_distance_table = np.maximum(temporal_distance_table, temporal_distance_table.T)

    np.save(filename, temporal_distance_table)
    return temporal_distance_table


# Linear Program Construction
# This has a lot of parameters, some of them ended up unused.
def create_LP(query, sim_table, start_time, window_i_j, window_j_i, membership_vectors, clust_sim_table, exp_temp_table, ent_table, numclust, relevance_table,
              K, mincover, sigma_t, credibility=[], bias=[], operations=[],
              has_start=True, has_end=False, window_time=None, cluster_list=[], start_nodes=[], end_nodes=[],
              verbose=True, force_cluster=True, previous_varsdict=None):
    n = len(query.index)  # We can cut out everything after the end.
    # Variable names and indices
    var_i = []
    var_ij = []
    var_k = [str(k) for k in range(0, numclust)]

    for i in range(0, n):  # This goes up from 0 to n-1.
        var_i.append(str(i))
        for j in window_i_j[i]:
            if i == j:
                print("ERROR IN WINDOW - BASE")
            var_ij.append(str(i) + "_" + str(j))

    # Linear program variable declaration.
    minedge = LpVariable("minedge", lowBound=0, upBound=1)
    node_act_vars = LpVariable.dicts("node_act", var_i, lowBound=0, upBound=1)
    node_next_vars = LpVariable.dicts("node_next", var_ij, lowBound=0,  upBound=1)
    # clust_active_vars = LpVariable.dicts("clust_active", var_k, lowBound=0, upBound=1)

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("StoryChainProblem", LpMaximize)
    # The objective function is added to 'prob' first
    prob += minedge, "WeakestLink"

    # Chain restrictions
    if has_start:
        num_starts = len(start_nodes)
        if verbose:
            print("Start node(s):")
            print(start_nodes)
        if num_starts == 0:  # This is the default when no list is given and it has a start.
            prob += node_act_vars[str(0)] == 1, 'InitialNode'
        else:
            if verbose:
                print("Added start node(s)")
                print("--- %s seconds ---" % (time() - start_time))
            initial_energy = 1.0 / num_starts
            earliest_start = min(start_nodes)
            for node in start_nodes:
                prob += node_act_vars[str(node)] == initial_energy, 'InitialNode' + str(node)
            for node in range(0, earliest_start):
                prob += node_act_vars[str(node)] == 0, 'BeforeStart' + str(node)
    if has_end:
        num_ends = len(end_nodes)
        if verbose:
            print("End node(s):")
            print(end_nodes)
        if num_ends == 0:  # This is the default when no list is given and it has a start.
            prob += node_act_vars[str(n - 1)] == 1, 'FinalNode'
        else:
            if verbose:
                print("Added end node(s)")
                print("--- %s seconds ---" % (time() - start_time))
            final_energy = 1.0 / num_ends
            latest_end = min(end_nodes)
            for node in end_nodes:
                prob += node_act_vars[str(node)] == final_energy, 'FinalNode' + str(node)
            for node in range(latest_end + 1, n):
                prob += node_act_vars[str(node)] == 0, 'AfterEnd' + str(node)

    if verbose:
        print("Chain constraints created.")
        print("--- %s seconds ---" % (time() - start_time))
    prob += lpSum([node_act_vars[i] for i in var_i]) == K, 'KNodes'

    if verbose:
        print("Expected length constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    if has_start:
        if verbose:
            print("Equality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for j in range(1, n):
            if j not in start_nodes:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for i in window_j_i[j]]) == node_act_vars[str(j)], 'InEdgeReq' + str(j)
            else:
                if verbose:
                    print("Generating specific starting node constraints.")
                    print("--- %s seconds ---" % (time() - start_time))
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for i in window_j_i[j]]) == 0, 'InEdgeReq' + str(j)
    else:
        if verbose:
            print("Inequality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for j in range(1, n):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                          for i in window_j_i[j]]) <= node_act_vars[str(j)], 'InEdgeReq' + str(j)
    if verbose:
        print("In-degree constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    if has_end:
        if verbose:
            print("Equality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for i in range(0, n - 1):
            if i not in end_nodes:
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for j in window_i_j[i]]) == node_act_vars[str(i)], 'OutEdgeReq' + str(i)
            else:
                if verbose:
                    print("Generating specific starting node constraints.")
                    print("--- %s seconds ---" % (time() - start_time))
                prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                              for j in window_i_j[i]]) == 0, 'OutEdgeReq' + str(i)
    else:
        if verbose:
            print("Inequality constraints.")
            print("--- %s seconds ---" % (time() - start_time))
        for i in range(0, n - 1):
            prob += lpSum([node_next_vars[str(i) + "_" + str(j)]
                          for j in window_i_j[i]]) <= node_act_vars[str(i)], 'OutEdgeReq' + str(i)
    if verbose:
        print("Out-degree constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    # Objective
    for i in range(0, n):
        for j in window_i_j[i]:
            coherence_weights = [0.5, 0.5]
            # Five or more entities in common means double the connection strength.
            entity_multiplier = min(1 + ent_table[i, j], 2)
            # Geometric mean the relevances, multiply based on how far it is from 0.5.
            relevance_multiplier = (relevance_table[i] * relevance_table[j]) ** 0.5
            coherence = (sim_table[i, j] ** coherence_weights[0]) * \
                (clust_sim_table[i, j] ** coherence_weights[1])
            weighted_coherence = min(coherence * entity_multiplier * relevance_multiplier, 1.0)
            prob += minedge <= 1 - node_next_vars[str(i) + "_" + str(j)] + \
                weighted_coherence, "Objective" + str(i) + "_" + str(j)
    if verbose:
        print("Objective constraints created.")
        print("--- %s seconds ---" % (time() - start_time))

    if previous_varsdict:
        current_names = [v.name for v in prob.variables() if "node_act" in v.name]
        if verbose:
            print("Generated list of names.")
            print("--- %s seconds ---" % (time() - start_time))
        for k, v in previous_varsdict.items():
            if "node_act" in k and k in current_names:
                node_act_vars[k.replace("node_act_", "")].setInitialValue(v)

    if verbose:
        if previous_varsdict:
            print("Used previous solution as starting point.")
            print("--- %s seconds ---" % (time() - start_time))
        else:
            print("No previous solution available.")
            print("--- %s seconds ---" % (time() - start_time))
    # The problem data is written to an .lp file
    return prob


# Useful function to build the graph from the LP output
def extract_varsdict(prob):
    # We get all the node_next variables in a dict.
    varsdict = {}
    for v in prob.variables():
        if "node_next" in v.name or "node_act" in v.name:
            varsdict[v.name] = np.clip(v.varValue, 0, 1)  # Just to avoid negative rounding errors.
    return varsdict


# Building the Graph Data Frame
def build_graph_df_multiple_starts(query, varsdict, window_i_j, prune=None, threshold=0.01, cluster_dict={}, start_nodes=[]):
    n = len(query)
    # This has some leftover stuff that is not really useful now.
    if 'bias' in query.columns:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights',
                                'date', 'publication', 'title', 'text', 'url', 'bias', 'coherence'])
    else:
        graph_df = pd.DataFrame(columns=['id', 'adj_list', 'adj_weights',
                                'date', 'publication', 'title', 'text', 'url', 'coherence'])

    already_in = []
    for i in range(0, n):
        prob = []
        coherence = varsdict["node_act_" + str(i)]
        if coherence <= threshold:
            continue
        coherence_list = []
        index_list = []
        for j in window_i_j[i]:
            name = "node_next_" + str(i) + "_" + str(j)
            prob.append(varsdict[name])
            coherence_list.append(varsdict["node_act_" + str(j)])
        idx_list = [window_i_j[i][idx] for idx, e in enumerate(prob) if round(
            e, 8) != 0 and e > threshold and coherence_list[idx] > threshold]  # idx + i + 1
        nz_prob = [e for idx, e in enumerate(prob) if round(
            e, 8) != 0 and e > threshold and coherence_list[idx] > threshold]
        if prune:
            if len(idx_list) > prune:
                top_prob_idx = sorted(range(len(nz_prob)), key=lambda k: nz_prob[k])[-prune:]
                idx_list = [idx_list[j] for j in top_prob_idx]
                nz_prob = [nz_prob[idx] for idx in top_prob_idx]
        sum_nz = sum(nz_prob)
        nz_prob = [nz_prob[j] / sum_nz for j in range(0, len(nz_prob))]
        # If we haven't checked this one before we add it to the graph.
        url = str(query.iloc[i]['url'])
        if i in already_in or sum_nz > 0:
            if len(url) > 0:
                url = urlparse(url).netloc
            if not (graph_df['id'] == i).any():
                title = query.iloc[i]['title']
                for key, value in cluster_dict.items():
                    if str(i) in value:
                        title = "[" + str(key) + "] " + title
                outgoing_edges = [idx_temp for idx_temp in idx_list]
                # coherence = varsdict["node_act_" + str(i)]
                if 'bias' in query.columns:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], query.iloc[i]['bias'], coherence]
                else:
                    graph_df.loc[len(graph_df)] = [i, outgoing_edges, nz_prob, query.iloc[i]['date'], query.iloc[i]['publication'],
                                                   title, '', query.iloc[i]['url'], coherence]

            already_in += [i] + idx_list
    return graph_df


# Building NetworkX graph.
def build_graph(graph_df):
    G = nx.DiGraph()
    for index, row in graph_df.iterrows():
        G.add_node(str(row['id']), coherence=max(-math.log(row['coherence']), 0))
        for idx, adj in enumerate(row['adj_list']):
            G.add_edge(str(row['id']), str(adj), weight=max(-math.log(row['adj_weights'][idx]), 0))
    return G


# Recursively Extract Storylines
def get_shortest_path(G):
    sources = [node for node, in_degree in G.in_degree() if in_degree == 0]
    targets = [node for node, out_degree in G.out_degree() if out_degree == 0]
    best_st = (sources[0], targets[0])
    try:
        best_val = nx.shortest_path_length(
            G, best_st[0], best_st[1], weight='weight') + G.nodes[best_st[0]]['coherence']  # Check? + vs *
    except nx.NetworkXNoPath:
        best_val = 100000
    for s in sources:
        for t in targets:
            try:
                current_val = nx.shortest_path_length(G, s, t, weight='weight') + G.nodes[s]['coherence']
            except nx.NetworkXNoPath:
                current_val = 100000
            if current_val < best_val:
                best_st = (s, t)
                best_val = current_val
    sp = nx.shortest_path(G, best_st[0], best_st[1], weight='weight')
    return sp


def normalize_graph(G):
    for node in G.nodes():
        llhs = [edge[2]['weight'] for edge in G.out_edges(node, data=True)]
        probabilities = [exp(-llh) for llh in llhs]
        sum_prob = sum(probabilities)
        probs = [prob / sum_prob for prob in probabilities]
        for idx, edge in enumerate(G.out_edges(node)):
            attrs = {edge: {'weigth': llhs[idx], 'prob': probs[idx]}}
            nx.set_edge_attributes(G, attrs)
    return G


def graph_stories(G, start_nodes=[], end_nodes=[]):
    # Base case, return the nodes if there is 1 or fewer nodes left.
    if len(G.nodes()) == 0:
        return []
    if len(G.nodes()) == 1:
        return [list(G.nodes())]
    # Base case, return node singletons if there are no edges left.
    if len(G.edges()) == 0:
        return [[node] for node in G.nodes()]
    # Main case.
    # Get maximum likelihood chain.
    if len(start_nodes) > 0 and len(end_nodes) > 0:  # First, special case when there is a start and end node.
        if nx.has_path(G, str(start_nodes[0]), str(end_nodes[0])):
            mlc = nx.shortest_path(G, str(start_nodes[0]), str(end_nodes[0]), weight='weight')
        else:
            mlc = get_shortest_path(G)  # Case where no paths exist.
    else:
        mlc = get_shortest_path(G)
    # Remove all nodes and adjacent edges to the maximum likelihood chain.
    H = G.copy()
    for node in mlc:
        H.remove_node(node)

    # Normalize outgoing edges to sum up to 1.
    H = normalize_graph(H)

    # Recursive call (special case if multiple connected components)
    # if nx.is_connected(H):
    # print("Connected. No issues.")
    return [mlc] + graph_stories(H)


# Extract Representative Landmarks
def get_representative_landmarks(G, storylines, query, mode="ranked"):
    antichain = []
    # first is default.
    if mode == "last":
        antichain = [story[-1] for story in storylines]  # get the last element in the story
    elif mode == "degree":
        degree_story = [[v for k, v in G.degree(story)] for story in storylines]
        max_degree_idx_list = [degrees.index(max(degrees)) for degrees in degree_story]
        antichain = [storylines[idx][max_degree_idx]
                     for idx, max_degree_idx in enumerate(max_degree_idx_list)]
    elif mode == "centrality":
        explanation = []
        antichain = []
        # Count non-singleton storylines.
        num_landmarks = len([story for story in storylines if len(story) > 1])
        # g_distance_dict = {(e1, e2): 1 / weight for e1, e2, weight in G.edges(data='weight')}
        # nx.set_edge_attributes(g, g_distance_dict, 'distance')
        # centrality = nx.closeness_centrality(G, distance='weight')
        centrality = nx.degree_centrality(G)
        centrality_df = pd.DataFrame.from_dict(
            {'node': list(centrality.keys()), 'centrality': list(centrality.values())})
        centrality_df = centrality_df.sort_values('centrality', ascending=False)
        for idx in range(num_landmarks):  # Get the top N landmarks based on centrality, where N = num_landmarks
            antichain.append(centrality_df.iloc[idx]['node'])
            explanation.append(
                "This event was marked as important due to its position as a relevant -hub- in the map.")
    elif mode == "centroid":
        explanation = []
        antichain = []
        for story in storylines:
            if len(story) > 1:  # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                idx_closest_node = np.argmin(l1_distance)
                antichain.append(story[idx_closest_node])  # need the ID, not the index
                explanation.append(
                    "This event was marked as important due to its -content- being representative of its corresponding storyline.")
    elif mode == "ranked":
        centrality = nx.betweenness_centrality(G, weight='weight')
        explanation = []
        for story in storylines:
            if len(story) > 1:  # We exclude singleton storyline (no relevant landmarks)
                node_embedding_list = [query.loc[query['id'] == node]['embed'].item() for node in story]
                node_embeddings = np.stack(node_embedding_list)
                centroid = node_embeddings.mean(axis=0)
                l1_distance = np.linalg.norm(node_embeddings - centroid, axis=1)
                temp = l1_distance.argsort()
                ranks_dist = np.empty_like(temp)
                ranks_dist[temp] = np.arange(len(l1_distance))
                # idx_closest_node = np.argmin(l1_distance)
                # antichain.append(story[idx_closest_node])
                centrality_array = np.array([centrality[node] for node in story])
                temp = centrality_array.argsort()
                ranks_centrality = np.empty_like(centrality_array)
                ranks_centrality[temp] = np.arange(len(centrality_array))

                average_ranks = np.average([ranks_dist, ranks_centrality], axis=0)
                idx_min = np.where(average_ranks == average_ranks.min())[0]
                lowest_dist_rank_idx = idx_min[0]
                lowest_dist_rank = ranks_dist[lowest_dist_rank_idx]
                # This doesn't matter if there's only one min node.
                # If there are many we give priority to dist to break ties.
                for idx in idx_min:
                    if lowest_dist_rank > ranks_dist[idx]:
                        lowest_dist_rank_idx = idx
                        lowest_dist_rank = ranks_dist[idx]
                antichain.append(story[lowest_dist_rank_idx])
                if ranks_dist[lowest_dist_rank_idx] < ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append(
                        "This event was marked as important due to its -content- being representative of its corresponding storyline.")
                elif ranks_dist[lowest_dist_rank_idx] == ranks_centrality[lowest_dist_rank_idx]:
                    explanation.append(
                        "This event was marked as important based on its -content- being representative of its corresponding storyline and acting as a relevant -hub- in the map.")
                else:
                    explanation.append(
                        "This event was marked as important due to its position as a relevant -hub- in the map.")
    else:  # first
        antichain = [story[0] for story in storylines]  # get the first element in the story
    return antichain

# start_time = None
# window_i_j = {}
# window_j_i = {}


def solve_LP(
    query,
    membership_vectors,
    temporal_distance_table,
    sim_table,
    clust_sim_table,
    numclust,
    K=6,
    mincover=0.20,
    sigma_t=30,
    start_nodes=[],
    end_nodes=[],
    verbose=True,
    force_cluster=True,
    use_temporal=True,
):
    # global start_time
    # global window_i_j
    # global window_j_i

    start_time = None
    window_i_j = {}
    window_j_i = {}

    start_time = time()

    n = len(query.index)

    if sigma_t != 0 and use_temporal:
        exp_temp_table = np.exp(-temporal_distance_table / sigma_t)
    else:
        exp_temp_table = np.ones(temporal_distance_table.shape)

    if verbose:
        print("Computed temporal distance table.")
        print("--- %s seconds ---" % (time() - start_time))

    window_time = None
    if sigma_t != 0 and use_temporal:
        window_time = sigma_t * 3  # Days

    if window_time is None:
        for i in range(0, n):
            window_i_j[i] = list(range(i + 1, n))
        for j in range(0, n):
            window_j_i[j] = list(range(0, j))
    else:
        for j in range(0, n):
            window_j_i[j] = []
        for i in range(0, n):
            window_i_j[i] = []
        for i in range(0, n - 1):
            window = 0
            for j in range(i + 1, n):
                if temporal_distance_table[i, j] <= window_time:
                    window += 1
            window = max(min(5, n - i), window)
            window_i_j[i] = list(range(i + 1, min(i + window, n)))
            for j in window_i_j[i]:
                window_j_i[j].append(i)

    if verbose:
        print("Computed temporal windows.")
        print("--- %s seconds ---" % (time() - start_time))

    if verbose:
        print("Computed entity similarities.")
        print("--- %s seconds ---" % (time() - start_time))
    ent_table = np.zeros((n, n))  # Fill entity information with zeros by default.
    actual_ent_table = ent_table
    ent_doc_list = None

    # Deprecated relevance table computation
    relevance_table = [1.0] * membership_vectors.shape[0]  # Create a vector full of 1s

    has_start = False
    if start_nodes is not None:
        has_start = (len(start_nodes) > 0)
    if end_nodes is not None:
        has_end = (len(end_nodes) > 0)
    if verbose:
        print("Creating LP...")

    # Read previous solution and feed to LP. If none there is no previous solution.
    previous_varsdict = None
    # if os.path.isfile(varsdict_filename):
    #     with open(varsdict_filename, 'rb') as handle:
    #         previous_varsdict = pickle.load(handle)

    prob = create_LP(
        query,
        sim_table,
        start_time,
        window_i_j,
        window_j_i,
        membership_vectors,
        clust_sim_table,
        exp_temp_table,
        actual_ent_table,
        numclust,
        relevance_table,
        K=K,
        mincover=mincover,
        sigma_t=sigma_t,
        has_start=has_start,
        has_end=has_end,
        start_nodes=start_nodes,
        end_nodes=end_nodes,
        verbose=verbose,
        force_cluster=force_cluster,
        previous_varsdict=previous_varsdict
    )

    # if verbose:
    #     print("Saving model...")
    #     print("--- %s seconds ---" % (time() - start_time))

    # prob.writeLP("left_story.lp")

    if verbose:
        print("Solving model...")
        print("--- %s seconds ---" % (time() - start_time))

    # (GLPK_CMD(path = 'C:\\glpk-4.65\\w64\\glpsol.exe', options = ["--tmlim", "180"]))

    prob.solve(PULP_CBC_CMD(mip=False, warmStart=True))

    varsdict = extract_varsdict(prob)

    # Overwrite last solution.
    # with open(varsdict_filename, 'wb') as handle:
    #     pickle.dump(varsdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graph_df = build_graph_df_multiple_starts(
        query,
        varsdict,
        window_i_j,
        prune=ceil(sqrt(K)),
        threshold=0.1/K,
        cluster_dict={}
    )

    if verbose:
        print("Graph data frame construction...")
        print("--- %s seconds ---" % (time() - start_time))

    if verbose:
        print("Graph clean up...")
        print("--- %s seconds ---" % (time() - start_time))

    return [graph_df, (numclust, LpStatus[prob.status]), sim_table, clust_sim_table, ent_table, ent_doc_list]
