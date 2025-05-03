from enviroment_class import Environment
from post_office import PostOffice

class Agent:
    def __init__(self, id, domain, env: Environment):
        self.id = id
        self.current_assign = None
        self.neighbors = set()
        self.mailbox = {}  # {iteration: {agent_id: message, ...}, ...}
        self.costs_to_neighbors = {}
        self.domain = domain
        self.env = env
        self.post_office = None

        # MGM
        self.proposed_gain = float('inf')
        self.proposed_value = None
        self.last_committed_iteration = -1

        # MGM-k matching
        self.proposals_sent = set()
        self.current_partner = None
        self.matched = False
        self.partner_id = None

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Agent) and self.id == other.id

    def add_neighbor(self, agent_id: int, cost_matrix=None):
        if agent_id in self.neighbors or agent_id == self.id:
            return
        self.neighbors.add(agent_id)
        self.costs_to_neighbors[agent_id] = cost_matrix if cost_matrix is not None else {}

    def receive_message(self, agent_id: int, message: str, iteration: int):
        if iteration not in self.mailbox:
            self.mailbox[iteration] = {}
        self.mailbox[iteration][agent_id] = message

    def notify_neighbors(self, iteration: int):
        for neighbor_id in self.neighbors:
            self.post_office.add_message({
                'sender': self.id,
                'recipient': neighbor_id,
                'content': self.current_assign,
                'iteration': iteration 
            })

    def notify_k_neighbors_about_gain(self, iteration: int, k: int, prob: float):
        self.proposals_sent = set()
        notified_count = 0
        for neighbor_id in sorted(self.neighbors):
            if notified_count >= k:
                break
            if self.env.random.random() >= prob:
                self.post_office.add_message({
                    'sender': self.id,
                    'recipient': neighbor_id,
                    'content': {
                        'proposal': True,
                        'gain': self.proposed_gain,
                        'id': self.id
                    },
                    'iteration': iteration
                })
                self.proposals_sent.add(neighbor_id)
                notified_count += 1

    def handle_incoming_proposals(self, iteration: int):
        messages = self.mailbox.get(iteration, {})
        proposals = [
            (agent_id, msg) for agent_id, msg in messages.items()
            if isinstance(msg, dict) and msg.get('proposal')
        ]
        if not proposals:
            return

        chosen_neighbor_id, chosen_msg = self.env.random.choice(proposals)

        self.post_office.add_message({
            'sender': self.id,
            'recipient': chosen_neighbor_id,
            'content': {
                'response': True,
                'accepted': True,
                'id': self.id,
                'gain': self.proposed_gain
            },
            'iteration': iteration
        })
        self.current_partner = chosen_neighbor_id

    def finalize_mutual_match(self, iteration: int):
        messages = self.mailbox.get(iteration, {})
        for agent_id, msg in messages.items():
            if (
                isinstance(msg, dict) and
                msg.get('response') and
                msg.get('accepted') and
                agent_id in self.proposals_sent
            ):
                self.matched = True
                self.partner_id = agent_id
                return
        self.matched = False
        self.partner_id = None

    def add_post_office(self, post_office: PostOffice):
        self.post_office = post_office

    def _calculate_cost(self, current_assign: int, iteration: int):
        total_cost = 0
        for neighbor_id in self.neighbors:
            cost_matrix = self.costs_to_neighbors.get(neighbor_id)
            neighbor_value = self.mailbox[iteration].get(neighbor_id)
            if neighbor_value is None:
                continue
            i = self.domain.index(current_assign)
            j = self.domain.index(neighbor_value)
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
            return best_value, best_cost
        return self.current_assign, current_cost

    def change_current_assign(self, value=None, prob=None, iteration=None):
        if value is not None:
            self.current_assign = value
        elif prob is not None:
            new_assign, best_cost = self._choose_new_assign(prob, iteration)
            if new_assign != self.current_assign:
                self.current_assign = new_assign

    def consider_change_current_assign(self, iteration: int):
        current_cost = self._calculate_cost(self.current_assign, iteration)
        best_value, best_cost = self._choose_new_assign(prob=1, iteration=iteration)
        self.proposed_value = best_value
        self.proposed_gain = current_cost - best_cost

    def decide_and_commit(self, iteration: int):
        mailbox = self.mailbox.get(iteration, {})
        can_commit = True

        for message in mailbox.values():
            if isinstance(message, dict) and 'gain' in message:
                neighbor_gain = message['gain']
                if neighbor_gain <= 0:
                    continue
                neighbor_id = message['id']

                if neighbor_gain > self.proposed_gain:
                    can_commit = False
                elif neighbor_gain == self.proposed_gain and neighbor_id < self.id:
                    can_commit = False

        if can_commit and self.proposed_value != self.current_assign and self.proposed_gain > 0:
            if iteration - self.last_committed_iteration > 1:
                self.current_assign = self.proposed_value
                self.last_committed_iteration = iteration

        self.proposed_gain = float('inf')
        self.proposed_value = None
