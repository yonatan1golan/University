from algorithms.base_algorithm import BaseAlgo

class DSA_C(BaseAlgo):
    def __init__(self, prob: float, graph):
        super().__init__('DSA_C', graph=graph)
        self.random = graph.env.random
        self.change_prob = prob

    def _algorithm(self) -> list:
        results = {}
        for iteration in range(0, self.graph.env.iterations):
            if iteration == 0:
                for node in self.graph.nodes:
                    node.change_current_assign(value=self.graph.env.random.choice(self.graph.domain))
            else:
                for node in self.graph.nodes:
                    node.notify_neighbors(iteration=iteration-1)
                
                self.graph.post_office.deliver_messages()
                
                for node in self.graph.nodes:
                    node.change_current_assign(prob=self.change_prob, iteration=iteration-1)
            
            results[iteration] = self.graph.calculate_global_price() 
        return results