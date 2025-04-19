from algorithms.base_algorithm import BaseAlgo

class MGM(BaseAlgo):
    def __init__(self, prob: float, graph):
        super().__init__('MGM', graph, prob)

    def _algorithm(self):
        pass