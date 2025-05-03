from config import CONFIG
import random

class Environment:
    def __init__(self, num_agent: int, iterations: int, runs: int):
        self.iterations = iterations
        self.runs = runs
        self.num_agent = num_agent
        self.random = random.Random(CONFIG.RANDOM_SEED)