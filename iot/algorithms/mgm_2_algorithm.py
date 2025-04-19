from algorithms.base_algorithm import BaseAlgo

class MGM2(BaseAlgo):
    def __init__(self, prob: float, graph):
        super().__init__('MGM-2', graph, prob)

    def _algorithm(self):
        pass