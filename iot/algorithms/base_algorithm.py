from abc import ABC, abstractmethod
from graph_class import Graph

class BaseAlgo(ABC):
    def __init__(self, name: str, graph: Graph, prob: float):
        self.graph = graph
        self.change_prob = prob
        self.name = name

    @abstractmethod
    def _algorithm(self):
        pass

    def run(self):
        self.graph.create_graph()
        self._algorithm()