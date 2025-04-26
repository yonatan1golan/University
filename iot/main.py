from enviroment_class import Environment
from graph_class import Graph
from config import CONFIG

import matplotlib.pyplot as plt
from collections import defaultdict

# algorithms
from algorithms.dsa_c_algorithm import DSA_C
from algorithms.mgm_algorithm import MGM_K
from algorithms.mgm_2_algorithm import MGM2

def plot_results(results, title):
    iterations = list(results.keys())
    cost_values = list(results.values())
    
    plt.plot(iterations, cost_values, marker='o')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    env = Environment(num_agent=CONFIG.NUM_AGENTS, iterations=CONFIG.NUM_ITERATIONS, runs=CONFIG.NUM_RUNS)

    iteration_grouped = defaultdict(list)
    for i in range(env.runs):
        graph = Graph(CONFIG.NUM_DOMAIN, CONFIG.DENSITY['LOW'], env)
        algorithm = MGM_K(graph = graph, k = 1, prob = CONFIG.PROB_DSA[1]) # DSA_C(prob=CONFIG.PROB_DSA[0], graph=graph)
        results_run = algorithm.run()
        for iteration_key, iteration_value in enumerate(results_run):
            iteration_grouped[iteration_key].append(iteration_value)

    averaged_results = {}
    for iteration_num, list_of_dicts in iteration_grouped.items():
        values = [list(d.values())[0] for d in list_of_dicts]
        averaged_results[iteration_num] = sum(values) / len(values)

    # plot_results(averaged_results, 'tt')

