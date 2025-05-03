from enviroment_class import Environment
from graph_class import Graph
from config import CONFIG

import matplotlib.pyplot as plt
from collections import defaultdict

# algorithms
from algorithms.dsa_c_algorithm import DSA_C
from algorithms.mgm_algorithm import MGM_K

import matplotlib.pyplot as plt

def plot_results(results, title):
    plt.figure(figsize=(10, 6))

    for algo_name, iteration_data in results.items():
        iterations = sorted(iteration_data.keys(), key=lambda x: float(x))
        costs = [iteration_data[i] for i in iterations]
        plt.plot(iterations, costs, marker='.', label=algo_name)

    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    env = Environment(
        num_agent=CONFIG.NUM_AGENTS,
        iterations=CONFIG.NUM_ITERATIONS,
        runs=CONFIG.NUM_RUNS
    )

    graph_settings = {
        'K=0.25': (CONFIG.NUM_DOMAIN, CONFIG.DENSITY['K1']),
        'K=0.75': (CONFIG.NUM_DOMAIN, CONFIG.DENSITY['K2']),
        'Colors': (CONFIG.COLOR_DOMAIN, CONFIG.DENSITY['K3'])
    }

    algorithms = {
        'DSA_C_LOW_CHANGE': (DSA_C, CONFIG.CHANGE_PROB['LOW']),
        'DSA_C_MEDIUM_CHANGE': (DSA_C, CONFIG.CHANGE_PROB['MEDIUM']),
        'DSA_C_HIGH_CHANGE': (DSA_C, CONFIG.CHANGE_PROB['HIGH']),
        'MGM_1': (MGM_K, 1),
        'MGM_2': (MGM_K, 2)
    }

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for graph_key, (domain, density) in graph_settings.items():
        for algo_name, algo_def in algorithms.items():
            for i in range(env.runs):
                graph = Graph(domain, density, env)
                if 'DSA_C' in algo_name:
                    AlgoClass, prob_value = algo_def
                    algorithm = AlgoClass(graph=graph, prob=prob_value)
                else: 
                    AlgoClass, k_value = algo_def
                    algorithm = AlgoClass(graph=graph, k=k_value)

                results_run = algorithm.run()
                for iteration_key, cost in results_run.items():
                    results[graph_key][algo_name][iteration_key].append(cost)

    for graph_key in graph_settings.keys():
        averaged_results = {}
        for algo_name in algorithms.keys():
            averaged_results[algo_name] = {}
            for iteration_key, cost_list in results[graph_key][algo_name].items():
                averaged_results[algo_name][iteration_key] = sum(cost_list) / len(cost_list)

        plot_results(averaged_results, title=f"Graph: {graph_key}")