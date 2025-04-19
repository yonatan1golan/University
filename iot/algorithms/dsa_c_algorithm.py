from algorithms.base_algorithm import BaseAlgo

class DSA_C(BaseAlgo):
    def __init__(self, prob: float, graph):
        super().__init__('DSA_C', graph, prob)
        self.random = graph.env.random

    def _algorithm(self):
        # initialize the first agent
        init_node = self.random.choice(self.graph.nodes)
        init_node.change_current_assign(value=self.random.choice(self.graph.domain))

        # run the algorithm until the stopping criteria is met
        for _ in range(self.graph.env.iterations):
            node = self.random.choice(self.graph.nodes)
            node.change_current_assign(prob=self.change_prob)