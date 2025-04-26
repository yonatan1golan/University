from abc import ABC, abstractmethod
from graph_class import Graph

class BaseAlgo(ABC):
    def __init__(self, name: str, graph: Graph, prob: float):
        self.graph = graph
        self.change_prob = prob
        self.name = name

    @abstractmethod
    def _algorithm(self) -> list:
        pass

    def run(self) -> list:
        """
        retrun a dictionary of the algorithm's results {iteration: cost}
        """
        self.graph.create_graph()
        run_results = self._algorithm()
        # self.graph.visualize_graph()
        return run_results