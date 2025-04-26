from enviroment_class import Environment
from post_office import PostOffice
import random

class Agent:
    def __init__(self, id, domain, env: Environment):
        self.id = id
        self.current_assign = None
        self.neighbors = set()
        self.mailbox = {} # json that looks like {iteration: {agent_id: [message]}, ...}
        self.costs_to_neighbors = {}
        self.domain = domain
        self.env = env
        self.post_office = None

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Agent) and self.id == other.id

    def add_neighbor(self, agent_id: int, cost_matrix=None):
        if agent_id in self.neighbors or agent_id == self.id:
            return
        self.neighbors.add(agent_id)
        self.costs_to_neighbors[agent_id] = cost_matrix if cost_matrix is not None else {}

    def receive_message(self, agent_id: int, message:str, iteration=int):
        # print(f"Iteration {iteration}: Agent {self.id} received message from Agent {agent_id}: {message}")
        if iteration not in self.mailbox:
            self.mailbox[iteration] = {}
        self.mailbox[iteration][agent_id] = message

    def notify_neighbors(self, iteration:int):
        # print(f"Current Iteration: {iteration}")
        for neighbor_id in self.neighbors:
            self.post_office.add_message({
                'sender': self.id,
                'recipient': neighbor_id,
                'content': self.current_assign,
                'iteration': iteration 
            })

    def add_post_office(self, post_office: PostOffice):
        self.post_office = post_office

    def _calculate_cost(self, current_assign: int, iteration:int):
        total_cost = 0
        for neighbor_id in self.neighbors:
            cost_matrix = self.costs_to_neighbors.get(neighbor_id)
            neighbor_value = self.mailbox[iteration].get(neighbor_id)
            i = self.domain.index(current_assign) # my value
            j = self.domain.index(neighbor_value) # neighbor's value
            total_cost += cost_matrix[i][j]
        return total_cost
    
    def _choose_new_assign(self, prob: float, iteration: int):
        current_cost = self._calculate_cost(self.current_assign, iteration)
        best_value = self.current_assign
        best_cost = current_cost

        for value in self.domain:
            if value == self.current_assign:
                continue
            new_cost = self._calculate_cost(value, iteration=iteration)
            if new_cost < best_cost:
                best_value = value
                best_cost = new_cost
        
        if best_cost < current_cost and self.env.random.random() < prob:
            return best_value
        return self.current_assign

    def change_current_assign(self, value=None, prob=None, iteration=None):
        if value is not None: # for the first iteration
            self.current_assign = value
        elif prob is not None:
            new_assign = self._choose_new_assign(prob, iteration)
            if new_assign != self.current_assign:
                self.current_assign = new_assign