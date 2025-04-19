from enviroment_class import Environment
from config import CONFIG
from agent_class import Agent

import networkx as nx
import matplotlib.pyplot as plt 

class Graph:
    def __init__(self, domain: list, K: float, env: Environment):
        self.env = env
        self.domain = domain
        self.density_prob = K
        self.nodes = []
        self.costs = {}

    def _add_node(self, agent: Agent):
        self.nodes.append(agent)

    def _create_neighborhoods(self):
        for node_i in self.nodes:
            for node_j in self.nodes:
                if node_i == node_j:
                    continue
                if self.env.random.random() <= self.density_prob:
                    if node_j not in node_i.neighbors:
                        node_i.add_neighbor(node_j)
                    if node_i not in node_j.neighbors:
                        node_j.add_neighbor(node_i)

    def _create_edges(self):
        n = len(self.domain)
        for node_i in self.nodes:
            for neighbor in node_i.neighbors:
                id_i, id_j = node_i.id, neighbor.id
                edge_key = tuple(sorted((id_i, id_j)))
                if edge_key in self.costs:
                    continue
                if n == len(CONFIG.NUM_DOMAIN):
                    cost_matrix = [[round(self.env.random.uniform(CONFIG.COST_LB, CONFIG.COST_UB), 2) for _ in range(n)] for _ in range(n)]
                elif n == len(CONFIG.COLOR_DOMAIN):
                    cost_matrix = [[round(self.env.random.uniform(CONFIG.COST_LB, CONFIG.COST_UB), 2) if i==j else 0 for j in range(n)] for i in range(n)]
                self.costs[edge_key] = cost_matrix
                node_i.costs_to_neighbors[neighbor.id] = cost_matrix
                neighbor.costs_to_neighbors[node_i.id] = cost_matrix

    def create_graph(self):
        for i in range(self.env.num_agent):
            self._add_node(Agent(i, self.domain, self.env))

        self._create_neighborhoods()
        self._create_edges()

    def visualize_graph(self):
        G = nx.Graph()

        # Add nodes
        for agent in self.nodes:
            G.add_node(agent.id)

        # Add edges (avoid duplicates)
        added_edges = set()
        for agent in self.nodes:
            for neighbor in agent.neighbors:
                edge = tuple(sorted((agent.id, neighbor.id)))
                if edge not in added_edges:
                    G.add_edge(*edge)
                    added_edges.add(edge)

        # Draw the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 7))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
        plt.title("Agent Graph")
        plt.show()

    def calculate_global_price(self):
        pass

    def print_result(self):
        pass