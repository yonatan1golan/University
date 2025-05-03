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
        'K1': 0.25,
        'K2': 0.75,
        'K3': 0.1
    }
    COST_LB = 100
    COST_UB = 200

    # algoritems
    CHANGE_PROB = {
        'LOW': 0.2,
        'MEDIUM': 0.7,
        'HIGH': 1.0
    }

    # env
    NUM_AGENTS = 30
    NUM_ITERATIONS = 100
    NUM_RUNS = 30
    RANDOM_SEED = 42