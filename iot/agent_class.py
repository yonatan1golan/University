from enviroment_class import Environment
class Agent:
    def __init__(self, id, domain, env: Environment):
        self.id = id
        self.current_assign = None
        self.neighbors = set()
        self.mailbox = {}
        self.costs_to_neighbors = {}
        self.domain = domain
        self.env = env

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Agent) and self.id == other.id

    def add_neighbor(self, agent: 'Agent', cost_matrix=None):
        if agent in self.neighbors or agent.id == self.id:
            return
        self.neighbors.add(agent)
        self.mailbox[agent.id] = []
        self.costs_to_neighbors[agent.id] = cost_matrix if cost_matrix is not None else {}

    def add_message_to_mailbox(self, agent: 'Agent', message):
        self.mailbox[agent.id].append(message)

    def notify_neighbors(self):
        for neighbor in self.neighbors:
            neighbor.add_message_to_mailbox(agent = self, message = self.current_assign)

    def _calculate_cost(self, value):
        total_cost = 0
        for neighbor in self.neighbors:
            messages = self.mailbox.get(neighbor.id, [])
            if not messages:
                continue  # no message yet from this neighbor

            neighbor_value = messages[-1]  # latest message
            if neighbor_value is None or value is None: # incase of first iteration
                continue

            cost_matrix = self.costs_to_neighbors.get(neighbor.id)
            if cost_matrix is None:
                continue  # safety check

            i = self.domain.index(value)
            j = self.domain.index(neighbor_value)
            total_cost += cost_matrix[i][j]
        return total_cost
    
    def _choose_new_assign(self, prob: float):
        current_cost = self._calculate_cost(self.current_assign)
        best_value = self.current_assign
        best_cost = current_cost

        for value in self.domain:
            if value == self.current_assign:
                continue
            new_cost = self._calculate_cost(value)
            if new_cost < best_cost:
                best_value = value
                best_cost = new_cost
        if best_cost < current_cost and self.env.random.random() <= prob:
            return best_value
        return best_value

    def change_current_assign(self, value=None, prob=None):
        if value is not None:
            self.current_assign = value
        elif prob is not None:
            new_assign = self._choose_new_assign(prob)
            self.current_assign = new_assign
        self.notify_neighbors()