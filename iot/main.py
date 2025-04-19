from enviroment_class import Environment
from graph_class import Graph
from config import CONFIG

# algorithms
from algorithms.dsa_c_algorithm import DSA_C
from algorithms.mgm_algorithm import MGM
from algorithms.mgm_2_algorithm import MGM2


if __name__ == '__main__':
    env = Environment(num_agent=CONFIG.NUM_AGENTS, iterations=CONFIG.NUM_ITERATIONS, runs=CONFIG.NUM_RUNS)
    uniform_graph_k1 = Graph(CONFIG.NUM_DOMAIN, CONFIG.NEIGHBOR_PROB[0], env)    

    # for node in uniform_graph_k1.nodes:
    #     print(f"\n🧠 Agent {node.id}")
    #     print(f"Neighbors: {[neighbor.id for neighbor in node.neighbors]}")

    #     for neighbor_id, cost_matrix in node.costs_to_neighbors.items():
    #         print(f"\n  📌 Cost to Neighbor {neighbor_id}:")
    #         for row in cost_matrix:
    #             formatted_row = '  '.join(f"{val:6.2f}" for val in row)
    #             print(f"    {formatted_row}")

    #   uniform_graph_k1.visualize_graph()

    dsa_k1 = DSA_C(prob=CONFIG.PROB_DSA[0], graph=uniform_graph_k1)
    dsa_k1.run()


