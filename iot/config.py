class CONFIG:
    # graphs
    NUM_DOMAIN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    COLOR_DOMAIN = ['R', 'G', 'B']
    NEIGHBOR_PROB = [0.25, 0.75, 0.1]
    COST_LB = 100
    COST_UB = 200

    # algoritems
    PROB_DSA = [0.2, 0.7, 1.0]

    # env
    NUM_AGENTS = 4 #30
    NUM_ITERATIONS = 5#100
    NUM_RUNS = 30
    RANDOM_SEED = 42