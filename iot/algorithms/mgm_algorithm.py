from algorithms.base_algorithm import BaseAlgo

class MGM_K(BaseAlgo):
    def __init__(self, graph, k: int = 1):
        super().__init__(f'MGM-{k}', graph)
        self.k = k
        self.share_prob = 0.5

    def _algorithm(self):
        results = {}
        logical_iterations = self.graph.env.iterations
        total_iterations = logical_iterations * self.k  # phase 1 + phase 2 per iteration

        for iteration in range(total_iterations):
            scaled_step = (iteration ) / self.k

            if iteration == 0:
                for node in self.graph.nodes:
                    initial_value = self.graph.env.random.choice(self.graph.domain)
                    node.change_current_assign(value=initial_value)
                results[0.0] = self.graph.calculate_global_price()

            elif iteration % 2 == 1:
                # phase 1
                for node in self.graph.nodes:
                    node.notify_neighbors(iteration=iteration - 1)
                self.graph.post_office.deliver_messages()

                for node in self.graph.nodes:
                    node.consider_change_current_assign(iteration=iteration - 1)

                for node in self.graph.nodes:
                    node.notify_k_neighbors_about_gain(
                        iteration=iteration,
                        k=self.k,
                        prob=self.share_prob
                    )

                results[scaled_step] = self.graph.calculate_global_price()

            else:
                # phase 2
                self.graph.post_office.deliver_messages()
                for node in self.graph.nodes:
                    node.decide_and_commit(iteration=iteration - 1)

                results[scaled_step] = self.graph.calculate_global_price()

        return results
