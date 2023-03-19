from datetime import datetime
import random
from typing import List, Tuple
import networkx as nx
from tqdm import tqdm_notebook as tqdm


def spanning_trees(G: nx.Graph):
    def build_tree(H: nx.Graph, edges):
        if nx.is_connected(H):
            yield H
        else:
            for i in range(len(edges)):
                if edges[i][1] not in nx.algorithms.descendants(H, edges[i][0]):
                    H1 = nx.Graph(H)
                    H1.add_edge(*edges[i])
                    for H2 in build_tree(H1, edges[i+1:]):
                        yield H2

    E = nx.Graph()
    E.add_nodes_from(G)
    return build_tree(E, [e for e in G.edges])


class EdgeCode:

    def __init__(self, code=0) -> None:
        self.code = code

    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int]] = [], nodes: int = 0) -> "EdgeCode":
        code = 0
        for edge in edges:
            code += cls.transform_edge_to_code(edge, nodes)
        return cls(code)

    @classmethod
    def transform_edge_to_code(cls, edge: Tuple[int, int], nodes: int) -> int:
        return 2**(edge[0]*nodes + edge[1])

    def zero_intersection(self, edge_code: "EdgeCode") -> bool:
        return (edge_code.code & self.code) == 0

    def add(self, edge_code: "EdgeCode") -> "EdgeCode":
        return EdgeCode(self.code | edge_code.code)

    # TODO
    # def to_edges(self, nodes) -> List[Tuple[int, int]]:


class Problem:

    def __init__(self, nodes: int, forest_size: int) -> None:
        self.nodes = nodes
        self.forest_size = forest_size
        self.nb_edges = (self.nodes-1)*self.forest_size
        if self.nb_edges > (self.nodes*(self.nodes-1)/2):
            raise ValueError(
                f"{self.nb_edges=} > {(self.nodes*(self.nodes-1)/2)}, "
                "the maximum number of edges for {self.nodes=}")

    def choose_edges(self, nb_edges) -> List[Tuple[int, int]]:
        possible_edges = []

        for i in range(self.nodes):
            for j in range(i+1, self.nodes):
                possible_edges.append((i, j))

        return random.sample(possible_edges, nb_edges)

    def create_random_graph(self) -> None:
        self.graph = nx.Graph()
        self.graph.add_nodes_from([i for i in range(self.nodes)])
        self.edges = self.choose_edges(self.nb_edges)
        self.graph.add_edges_from(self.edges)
        min_degree = min(list(zip(*self.graph.degree))[1])
        if min_degree < self.forest_size:
            self.create_random_graph()

    def create_random_coloring(self) -> None:
        shuffled_edges = random.sample(self.graph.edges, self.nb_edges)
        self.colors = []
        for color in range(self.nodes - 1):
            colored_edges = shuffled_edges[self.forest_size*color:
                                           self.forest_size*(color+1)]
            self.colors.append(EdgeCode.from_edges(colored_edges, self.nodes))
            for edge in colored_edges:
                self.graph[edge[0]][edge[1]]['color'] = color

    def create_spanning_trees(self) -> None:
        self.spanning_trees = [EdgeCode.from_edges(tree.edges, self.nodes)
                               for tree in tqdm(spanning_trees(self.graph))]

    def is_colored(self, tree: EdgeCode) -> bool:
        for color in self.colors:
            if tree.zero_intersection(color):
                return False
        return True

    def _has_covering_forest(
        self,
        trees: List[EdgeCode],
        excluded_elems: EdgeCode = EdgeCode(),
        iteration=1
    ) -> bool:
        for i, tree in enumerate(trees):
            if tree.zero_intersection(excluded_elems):
                if iteration == self.forest_size:
                    return True
                else:
                    if self._has_covering_forest(
                        trees[i+1:],
                        excluded_elems=tree.add(excluded_elems),
                        iteration=iteration+1
                    ):
                        return True
        return False
    
    def has_covering_forest(
        self,
        trees: List[EdgeCode],
    ) -> bool:
        return self._has_covering_forest(trees)

    def is_counter_example(self) -> bool:
        # This supposes that the graph is generated, colored and spanning trees
        # have been found
        # print("Finding colored trees...")
        self.colored_trees = [
            tree for tree in self.spanning_trees if self.is_colored(tree)
        ]

        # print("Finding colored covering forest...")
        return not self.has_covering_forest(self.colored_trees)

    def find_counter_example(
        self,
        number_of_graphs,
        number_of_colorings
    ) -> nx.Graph:
        # TODO: keep more info in order to recalculate less

        for _ in tqdm(range(number_of_graphs)):
            self.create_random_graph()
            self.create_spanning_trees()
            for _ in tqdm(range(number_of_colorings)):
                self.create_random_coloring()
                if self.is_counter_example():
                    nx.write_gml(
                        self.graph,
                        f"{self.nodes}-{self.forest_size}-{datetime.now().microsecond}")
                    if self.has_covering_forest(self.spanning_trees):
                        return self.graph
                    else:
                        print("No covering forests.")
                        break


def draw_colored_graph(graph: nx.Graph) -> None:
    nx.draw(graph, edge_color=[graph[u][v]['color'] for u, v in graph.edges()])
