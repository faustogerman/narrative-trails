import heapq

import umap
import hdbscan
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import KDTree, distance

from Library.helper import plot_embedding_space, plot_topic_count


class NarrativeLandscape():
    def __init__(
        self,
        # UMAP Params
        n_neighbors=32,
        n_components=48,
        # HDBSCAN Params
        min_cluster_size=8,
        cluster_selection_method="eom",
        # Sparse Coherence Parameters
        threshold: float = 1,
        # Constraints Parameters
        impose_date_constraint=False,
        # Other
        verbose=False
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_method = cluster_selection_method
        self.threshold = threshold
        self.impose_date_constraint = impose_date_constraint
        self.verbose = verbose

        self.low_dim_embeds = None
        self.cluster_labels = None
        self.cluster_label_probs = None
        self.nx_graph: nx.DiGraph = None

    def fit(
        self,
        embeds: np.ndarray,
        ids: np.array = None,
        dates: pd.Series = None,
        constraints=None,
        with_coherence_graph=True,
        node_ranks=None
    ) -> None:
        self.embeds = embeds

        if self.verbose:
            print("Step 1/3: Constructing Projection Space with UMAP")

        umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0,
            metric="cosine",
            random_state=42,
            n_jobs=1,
            low_memory=True
        )

        low_dim_mapper = umap_model.fit(embeds)
        self.low_dim_embeds = low_dim_mapper.embedding_
        # Center the embeddings around the mean
        self.low_dim_embeds = self.low_dim_embeds - np.mean(self.low_dim_embeds, axis=0)

        if self.verbose:
            print("Step 2/3: Discovering topics with HDBSCAN")

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        ).fit(self.low_dim_embeds)

        self.cluster_label_probs = hdbscan.prediction.all_points_membership_vectors(
            hdbscan_model
        )

        # Assign outliers to the cluster with the highest probability.
        # Uncomment the next line to keep outliers as a separate cluster.
        self.cluster_labels = self.cluster_label_probs.argmax(1)
        # self.cluster_labels = hdbscan_model.labels_

        if self.verbose:
            print(f"\t>>> Discovered {len(np.unique(self.cluster_labels))} Topics")

        if with_coherence_graph:  # this is only to be able to do parameter sensitivity tests
            print("Step 3/3: Constructing Coherence Graph")
            self.__build_coherence_graph(ids, dates, constraints, node_ranks)

    def extract_narrative(self, s, t, hidden_nodes=None):
        return NarrativeLandscape.maximum_capacity_path(self.nx_graph, s, t, hidden_nodes)

    def __build_coherence_graph(self, ids: np.array = None, dates: pd.Series = None, constraints: np.array = None, node_ranks=None):
        full_constraints = (dates[:, None] <= dates).astype(int) if self.impose_date_constraint else None

        if constraints is not None:
            full_constraints = (full_constraints is not None if full_constraints else 1) * constraints

        if self.verbose:
            print("\t >>> Computing base coherence")

        self.base_coherence, self.ang_sim, self.topic_sim = NarrativeLandscape.compute_base_coherence_matrix(
            embeds=self.embeds,
            cluster_probs=self.cluster_label_probs,
            node_ranks=node_ranks
        )

        if self.verbose:
            print("\t >>> Computing sparse coherence")

        self.sparse_coherence, self.critical_coherence, self.mst = NarrativeLandscape.compute_sparse_coherence(
            self.base_coherence,
            constraints=full_constraints,
            threshold=self.threshold,
            verbose=self.verbose
        )

        if self.verbose:
            print("\t >>> Building NetworkX graph")

        node_ids = ids if ids is not None else range(len(self.embeds))
        self.nx_graph = NarrativeLandscape.build_nx_graph(node_ids, self.sparse_coherence)

    def compute_base_coherence_matrix(embeds, cluster_probs: np.ndarray, node_ranks=None):
        # Compute cosine similarity and fix rounding errors
        # Here we use the dot product, since the high-dimensional embeddings are normalized
        cos_sim = np.clip(embeds @ embeds.T, -1, 1)

        # Compute angular similarity
        ang_sim = 1 - np.arccos(cos_sim) / np.pi

        # Diagonals may sometimes be NaN. Probably from rounding errors
        # We set them to 0 here since we're not interested in self-loops.
        np.fill_diagonal(ang_sim, 0)

        # Compute topic similarity
        topic_sim = 1 - distance.cdist(
            XA=cluster_probs,
            XB=cluster_probs,
            metric='jensenshannon'
        )

        coherence_matrix = (ang_sim * topic_sim) ** (1 / 2)

        if node_ranks is not None:
            coherence_matrix = node_ranks * coherence_matrix

        # Warn user about NaN values.
        if np.isnan(cos_sim).any():
            print("WARNING: Cosine Similarity matrix contains NaN values.")
        if np.isnan(ang_sim).any():
            print("WARNING: Angular Similarity matrix contains NaN values.")
        if np.isnan(topic_sim).any():
            print("WARNING: Topic Similarity matrix contains NaN values.")
        if np.isnan(coherence_matrix).any():
            print("WARNING: Coherence matrix contains NaN values.")

        return coherence_matrix, ang_sim, topic_sim

    def compute_sparse_coherence(base_coherence: np.ndarray, threshold: float, constraints: np.ndarray = None, verbose=False):
        if verbose:
            print("\t\t>>> Creating Undirected Graph")

        # The graph is undirected at this point, so we only consider to the upper-triangular part
        bcg = nx.from_numpy_array(np.triu(base_coherence), create_using=nx.Graph)

        if verbose:
            print("\t\t>>> Finding Maximum Spanning Tree")

        max_span_tree = nx.maximum_spanning_tree(bcg)

        if verbose:
            print("\t\t>>> Getting Min Weight")

        edges = sorted(max_span_tree.edges(data=True), key=lambda edge: edge[2].get('weight', 1))
        critical_weight = edges[0][2]["weight"]

        if verbose:
            print("\t\t----- BEFORE MST -----")
            print("\t\tCritical Coherence:", critical_weight)
            print("\t\tNum Edges:", len(bcg.edges()))
            print("\t\tIs Connected:", nx.is_connected(bcg))

        # NOTE: The graph should always be connected after the threshold.
        sparse_coherence = base_coherence.copy()
        sparse_coherence[sparse_coherence < (critical_weight * threshold)] = 0

        if verbose:
            scg2 = nx.from_numpy_array(sparse_coherence, create_using=nx.Graph)
            print("\t\t----- AFTER MST -----")
            print("\t\tNum Edges:", len(scg2.edges()))
            print("\t\tIs Connected:", nx.is_connected(scg2))

        # NOTE: After constraints, there is no guarantee that the graph will be connected.
        if constraints is not None:
            sparse_coherence = sparse_coherence * constraints

        if verbose:
            scg3 = nx.from_numpy_array(sparse_coherence, create_using=nx.Graph)
            print("\t\t----- AFTER Constraints -----")
            print("\t\tNum Edges:", len(scg3.edges()))
            print("\t\tIs Connected:", nx.is_connected(scg3))

        return sparse_coherence, critical_weight, max_span_tree

    def build_nx_graph(node_ids, sparse_coherence):
        nx_graph = nx.from_numpy_array(sparse_coherence, create_using=nx.DiGraph())

        # Only relabel nodes if the mapping (node_ids) doesn't equal the range itself.
        # Otherwise, NX will throw an error.
        if sorted(node_ids) != list(range(len(node_ids))):
            nx.relabel_nodes(nx_graph, {idx: node_id for idx, node_id in enumerate(node_ids)}, copy=False)

        return nx_graph

    def maximum_capacity_path(G, s, t, hidden=None):
        # Initialize the priority queue with the start node.
        # Use negative infinity to simulate the highest capacity.
        queue = [(-float('inf'), s)]
        best_min_capacity = {s: float('inf')}
        parent = {s: None}

        while queue:
            # Pop the node with the highest minimal capacity.
            neg_min_capacity, node = heapq.heappop(queue)
            min_capacity = -neg_min_capacity

            # If the goal is reached, reconstruct and return the path.
            if node == t:
                path = []
                current_node = t
                while current_node is not None:
                    path.append(current_node)
                    current_node = parent[current_node]
                path.reverse()  # Reverse the path to get it from start to goal.
                return path, min_capacity  # Return the path and the minimal capacity.

            # Explore neighbors.
            for neighbor in G.neighbors(node):
                if hidden and neighbor in hidden:
                    continue

                # Get the capacity (weight) of the edge.
                edge_data = G.get_edge_data(node, neighbor)
                capacity = edge_data.get('weight', 0)

                # Calculate the minimal capacity along the path to the neighbor.
                path_min_capacity = min(min_capacity, capacity)

                # If this path offers a better minimal capacity, update and push to queue.
                if neighbor not in best_min_capacity or path_min_capacity > best_min_capacity[neighbor]:
                    best_min_capacity[neighbor] = path_min_capacity
                    parent[neighbor] = node  # Update the parent to reconstruct the path.
                    heapq.heappush(queue, (-path_min_capacity, neighbor))

        # If the goal is not reachable, return None
        return None, None

    def plot_edge_connections(self):
        A = nx.adjacency_matrix(
            G=self.nx_graph,
            nodelist=np.arange(len(self.sparse_coherence)),
            weight='weight'
        ).todense()

        print("min Sparse Coherence:    ", -np.max(A))
        print("max Sparse Coherence:    ", -np.min(A[A > 0]))
        print("Symmetric matrix:        ", np.isclose(A, A.T, rtol=1e-10).all())
        print("Is connected:            ", nx.is_strongly_connected(self.nx_graph))
        print("Is complete:             ", self.nx_graph.number_of_edges() == len(A) * (len(A) - 1))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))

        img = Image.fromarray((A * 255 / np.max(A)).astype('uint8'))
        ax.set_title("Sparse Coherence Matrix")
        ax.imshow(img)

        return fig

    def plot_2d(self, data, summary_column):
        vis_umap_model = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=2,
            min_dist=0.25,
            metric="cosine",
            random_state=42,
            n_jobs=1,
        )

        embed_2d_mapper = vis_umap_model.fit(self.embeds)
        embed_2d = embed_2d_mapper.embedding_

        scatter_plot, selected_topic, _ = plot_embedding_space(
            data,
            embed_2d,
            self.cluster_labels,
            width=600,
            height=500,
            summary_col=summary_column
        )

        bar_chart = plot_topic_count(self.cluster_labels, True, selected_topic).properties(
            title="Topic Counts",
            width=520,
            height=500,
        )

        return scatter_plot | bar_chart
