from algorithms.base_algorithm import BaseAlgo
import json

class MGM_K(BaseAlgo):
    def __init__(self, graph, prob: float, k: int = 1):
        super().__init__(f'MGM-{k}', graph, prob)
        self.k = k
        self.share_prob = prob

    def _algorithm(self):
        results = []
        for iteration in range(0, self.graph.env.iterations):
            if iteration == 0:
                for node in self.graph.nodes:
                    node.change_current_assign(value=self.graph.env.random.choice(self.graph.domain))
            else:
                # phase 1
                for node in self.graph.nodes:
                    node.notify_neighbors(iteration)

                # phase 2
                for node in self.graph.nodes:
                    node.extend_my_knowledge(extend_knowledge = self.share_prob, k = self.k, iteration=iteration)

                self.graph.post_office.deliver_messages()

                for node in self.graph.nodes:
                    got_extended_knowledge = node.did_get_extended_knowledge(iteration)
                    if got_extended_knowledge:
                        best_offer = node.select_best_offer(iteration)
                        if best_offer is not None and node.is_offer_better_than_current(best_offer):
                            pass
            results.append({iteration: self.graph.calculate_global_price()})
        return results