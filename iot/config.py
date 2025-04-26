class CONFIG:
    # graphs
    NUM_DOMAIN = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4
    }
    COLOR_DOMAIN = {
        'R': 0,
        'G': 1,
        'B': 2
    }
    DENSITY = {
        'LOW': 0.2,
        'MEDIUM': 0.75,
        'HIGH': 1
    }
    COST_LB = 100
    COST_UB = 200

    # algoritems
    PROB_DSA = [0.2, 0.7, 1.0]

    # env
    NUM_AGENTS = 3 # 30
    NUM_ITERATIONS = 3 # 100
    NUM_RUNS = 1 # 30
    RANDOM_SEED = 42