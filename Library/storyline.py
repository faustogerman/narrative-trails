import copy

import networkx as nx
import numpy as np
import pandas as pd

from .narrative_landscape import NarrativeLandscape


class Storyline:
    def __init__(self, landscape: NarrativeLandscape, chain: list) -> None:
        self.landscape = landscape
        self.chain = chain

    def path_base_coherence(self):
        return np.array([
            self.landscape.base_coherence[src][tgt]  # Note that this is different than the "sparse" coherence
            for src, tgt in zip(self.chain[:-1], self.chain[1:])
        ])

    def bottleneck_weight(self):
        return self.path_base_coherence().min()

    def reliability(self):
        coherence = self.path_base_coherence()
        return coherence.prod() ** (1 / len(coherence))

    def reduce_redundancy(self, delta=0.2, inplace=False):
        path = copy.deepcopy(self.chain)

        idx = 0
        while idx < len(path) - 2:
            A = path[idx + 0]
            B = path[idx + 1]
            C = path[idx + 2]

            if (
                self.landscape.nx_graph.has_edge(A, C)
                and self.landscape.nx_graph.has_edge(A, B)
                and self.landscape.nx_graph.has_edge(B, C)
            ):
                AB = self.landscape.nx_graph.get_edge_data(A, B)["weight"]
                BC = self.landscape.nx_graph.get_edge_data(B, C)["weight"]
                AC = self.landscape.nx_graph.get_edge_data(A, C)["weight"]
                rel = (AB * BC) ** 0.5

                if AC >= rel - delta:
                    del path[idx + 1]

            idx += 1

        if inplace:
            old_chain = copy.deepcopy(self.chain)
            self.chain = path
            return old_chain
        else:
            return path

    def print_narrative_path(data, cluster_labels, path, config):
        print(
            "idx   ",
            "Topic  ",
            "Date".ljust(16),
            config.summary_column
        )
        print("-" * 64)

        if "date" in data.columns:
            path_docs = data.loc[path].reset_index()[
                ["date", config.summary_column]
            ].values
        else:
            path_docs = data.loc[path].reset_index()[
                [config.summary_column]
            ].values

        clusters = cluster_labels[path]
        for idx, text, cluster in zip(path, path_docs, clusters):

            if "date" in data.columns:
                print(
                    str(idx).ljust(6),
                    str(cluster).ljust(7),
                    str(text[0].strftime("%b %d, %Y")).ljust(16),
                    text[1]
                )
            else:
                print(
                    str(idx).ljust(6),
                    str(cluster).ljust(7),
                    text
                )

