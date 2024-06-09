import geopy.distance
import pandas as pd
import regex as re
import requests
import os.path
import random
import math

# defaults
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
adjacency_path = os.path.join(parent_directory, "adjacency.csv")

# states-coordinates dictionary instead of using the API for lowering the run time 
state_coords = {
    "AL": (32.31823, -86.902298),
    "AK": (66.160507, -153.369141),
    "AR": (34.799999, -92.199997),
    "AZ": (34.048927, -111.093735),
    "CA": (36.778259, -119.417931),
    "CO": (39.113014, -105.358887),
    "CT": (41.599998, -72.699997),
    "DE": (39, -75.5),
    "DC": (39.2037, 76.861),
    "FL": (27.994402, -81.760254),
    "GA": (33.247875, -83.441162),
    "HI": (19.741755, -155.844437),
    "ID": (44.068203, -114.742043),
    "IL": (40, -89),
    "IN": (40.273502, -86.126976),
    "IA": (42.032974, -93.581543),
    "KS": (38.5, -98),
    "KY": (37.839333, -84.27002),
    "LA": (30.39183, -92.329102),
    "ME": (45.367584, -68.972168),
    "MD": (39.045753, -76.641273),
    "MA": (42.407211, -71.382439),
    "MI": (44.182205, -84.506836),
    "MN": (46.39241, -94.63623),
    "MS": (33, -90),
    "MO": (38.573936, -92.60376),
    "MT": (46.96526, -109.533691),
    "NE": (41.5, -100),
    "NV": (39.876019, -117.224121),
    "NH": (44, -71.5),
    "NJ": (39.833851, -74.871826),
    "NM": (34.307144, -106.018066),
    "NY": (43, -75),
    "NC": (35.782169, -80.793457),
    "ND": (47.650589, -100.437012),
    "OH": (40.367474, -82.996216),
    "OK": (36.084621, -96.921387),
    "OR": (44, -120.5),
    "PA": (41.203323, -77.194527),
    "RI": (41.742325, -71.742332),
    "SC": (33.836082, -81.163727),
    "SD": (44.5, -100),
    "TN": (35.860119, -86.660156),
    "TX": (31, -100),
    "UT": (39.41922, -111.950684),
    "VT": (44, -72.699997),
    "VA": (37.926868, -78.024902),
    "WA": (47.751076, -120.740135),
    "WV": (39, -80.5),
    "WI": (44.5, -89.5),
    "WY": (43.07597, -107.290283),
}

# global vars
counties = {}
detailed = False

# input
start_locations = ['Red, Washington County, UT' , 'Blue, Fairfield County, CT'] #  
goal_locations = ['Red, Orange County, CA' , 'Blue, Suffolk County, NY'] #  

### classes   
class County:
    def __init__(self, txt:str):
        self.name = txt.split(",")[0]
        self.state = txt.split(",")[1]
        self.id = txt
        self.neighbors = []
        self.parent = None
        self.visited = False
        self.color = ''

        self.g = float('inf') # distance from start
        self.h = float('inf') # heuristic distance from the goal
        self.f = 0
    
    def add_neighbor(self, neighbor):
        if f"{self.name}, {self.state}" != f"{neighbor.name}, {neighbor.state}":
            self.neighbors.append(neighbor)

    def update_f(self, value = None):
        if value is None:
            self.f = self.g + self. h
        else: self.f = value
### end block classes

### help functions
def read_neighbors_file(file_name: os) -> pd.DataFrame: # reads the csv file and converts it into a dataframe
    return pd.read_csv(file_name)

def get_unique_list(df: pd.DataFrame, col_name: str) -> list: # returns the unique values in a df[col_name]
    return list(set(df[col_name]))

def make_object_list(lst: list) -> list: # gets a list of text and returns a list of County objects
    return [County(c) for c in lst]

def preparing_objects(raw_df: pd.DataFrame) -> dict: # making the dataframe into objects and adding their neighbors
    unique_counties = get_unique_list(raw_df, 'countyname')
    county_objects = make_object_list(unique_counties) 
    counties_dict = {county.name + "," + county.state: county for county in county_objects}
    for _, record in raw_df.iterrows():
        county = record['countyname']
        neighbor = record['neighborname']
        cnty_object = counties_dict[county]
        neighbor_object = counties_dict[neighbor]
        cnty_object.add_neighbor(neighbor_object)
    return counties_dict

def find_path(starting_locations, goal_locations, search_method, detail_output): # finds the shortest path from the starting locations to the goal location using a search method
    global detailed 
    detailed = detail_output
    pathes = []
    infos = []
    for start in starting_locations:
        info = None
        if search_method == 1: # A*
            path = a_star(start, goal_locations)
        elif search_method == 2: # Hill Climbing
            path = hill_climbing(start, goal_locations)
        elif search_method == 3: # Simulated Annealing
            tempature = 100
            path, info = simulated_annealing(start,goal_locations, (tempature if tempature <=100 else 100)) # need to do (path, considers)
        elif search_method == 4: # Lical K-Beam
            path, info = local_beam(start, goal_locations) # need to do (path, bags)
        elif search_method == 5: # Genetic Algorithm
            population_size = 10
            max_generations = 10
            path, info = genetic_search(start, goal_locations, population_size, max_generations) # need to do (path, new population)
        
        if not path:
            pathes.append('No path found.')
        else:
            pathes.append(path)
            infos.append(info)
    return pathes, infos

def get_county_coordinates(county_name): # returns the minimum distance between a start county and one of the goals
    url = f"https://nominatim.openstreetmap.org/search?q={county_name}&format=json"
    headers = {
        'User-Agent': 'YourAppName/1.0 (your.email@example.com)'
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            raise Exception(f"No data found for {county_name}")
    else:
        raise Exception(f"Problem with the API. Status code: {response.status_code}")

def heuristic_calc(start, goal): # returns the distance between two state coordinates in km
    start_state = start.split(', ')[1]
    goal_state = goal.split(', ')[1]
    start_cord = state_coords[start_state]
    goal_cord = state_coords[goal_state]
    distance = geopy.distance.geodesic(start_cord, goal_cord).km
    return distance

def retracePath(c, path=None): # retraces path from the last node to the start
    if path is None:
        path = []
    if c.parent:
        retracePath(c.parent, path)
    path.append(c.id)
    c.visited = True
    return path

def get_list_per_color(lst, pattern): # returns a list for the same color
    return [loc.replace('Red, ', '').replace('Blue, ', '') for loc in lst if re.search(pattern, loc)]

def assigning_colors_to_counties(starts, goals): # assigning color to each input location
    global start_locations, goal_locations
    for location in [*starts, *goals]:
        color, county_name, state = location.split(", ")
        county_id = f"{county_name}, {state}"
        if county_id in counties:
            counties[county_id].color = color

    # re-arranging the input lists to be without the colors
    start_locations_no_color = [location.split(', ', 1)[1] for location in starts] # temp
    goal_locations_no_color = [location.split(', ', 1)[1] for location in goals] # temp
    start_locations = start_locations_no_color
    goal_locations = goal_locations_no_color

def print_paths(paths, search_method, info = []): # prints paths according to the search method
    all_lists = [
        [f"{county} (R)" if counties[county].color == 'Red' else f"{county} (B)" for county in path] if path != "No path found." else ["No path found."]
        for path in paths
    ]

    # Determine the maximum length of the lists
    max_length = max(len(lst) for lst in all_lists)
    previous_step = None
    for i in range(max_length):
        step_elements = []
        for lst in all_lists:
            if i < len(lst):
                step_elements.append(lst[i])
            else:
                step_elements.append(lst[-1])  # Repeats the last element if the list is shorter
        print(f"{{{' ; '.join(step_elements)}}}")
        if detailed == 1 and i == 1:
            detail_infos = []
            for current, prev in zip(step_elements, previous_step):
                if "No path found." in [current, prev]:
                    detail_infos.append("N/A")
                else:
                    current_loc = current.split(' (')[0]
                    prev_loc = prev.split(' (')[0]
                    if search_method in (1, 2): # a* or hill climbing
                        heuristic_value = heuristic_calc(prev_loc, current_loc)
                        detail_infos.append(f"{heuristic_value:.2f}")
                    elif search_method == 3:
                        sa_info = next((item for sublist in infos for item in sublist if item['current'] == current_loc), None)
                        if sa_info:
                            neighbors_info = ', '.join([f"{n_id}: {prob:.2f}" for n_id, prob in sa_info['neighbors'].items()])
                            detail_infos.append(f"{neighbors_info}")
                    elif search_method == 4:
                        for info in infos:
                            detail_infos.append(f"{[c.id for c in info]}")
                    elif search_method == 5:
                        for sublist in infos:
                            for item in sublist:
                                detail_infos.append(f"{item}")
            print(f"\ndetailed:\n{{{' ; '.join(detail_infos)}}}\n")
        previous_step = step_elements

def pre_process(): # pre process the counties to the object dict format
    global counties
    raw_df = read_neighbors_file(adjacency_path)
    counties = preparing_objects(raw_df) # dict: {county.name, county.state: county object}. this is the same dict as neighbors so it is enough for one of them    
    assigning_colors_to_counties(start_locations, goal_locations)
### end block help functions 

### search algorithms
# 1
def a_star(start, goals): # performs A* search from a starting location to one of the ending locations in the goal list
    frontier = [] # have been visited but not expanded 
    explored = set() # visited and expanded
    path = []
    start_county = counties[start]
    start_county.g = 0
    start_county.h = min(heuristic_calc(start_county.id, counties[goal].id) for goal in goals)
    start_county.update_f()
    frontier.append(counties[start])

    while frontier:
        current = min(frontier, key = lambda county: county.f) # takes the county with the minimum f
        if current.id in goals and not current.visited and current.color == start_county.color : # reached an unvisited goal node
            path = retracePath(current)
            return path
        frontier.remove(current)
        current.visited = True
        explored.add(current)
        for neighbor in current.neighbors:
            if neighbor in explored: continue
            tentative_g = current.g + heuristic_calc(current.id, neighbor.id)
            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = min(heuristic_calc(neighbor.id, counties[goal].id) for goal in goals)
                neighbor.update_f()

            if neighbor not in frontier:
                frontier.append(neighbor)
    return path

# 2
def hill_climbing(start, goals, max_iteration = 5): # performs a hill climbing search from a start point to one of the goals
    def check_valid_solution(current):
        return (current.id in goals) and (not current.visited) and (current.color == counties[start].color)

    current = counties[start]
    for _ in range(max_iteration):
        while True:
            if check_valid_solution(current):  # reached an unvisited goal node
                return retracePath(current)
            
            unvisited_neighbors = [neighbor for neighbor in current.neighbors if not neighbor.visited]
            if not unvisited_neighbors:
                break

            goal_neighbors = [neighbor for neighbor in unvisited_neighbors if (neighbor.id in goals)]
            if goal_neighbors:
                valid_goal_neighbors = [goal for goal in goal_neighbors if check_valid_solution(goal)]
                if valid_goal_neighbors:
                    goal = random.choice(valid_goal_neighbors)
                    goal.parent = current
                    return retracePath(goal)
            
            closest_neighbor_to_goal = min(unvisited_neighbors, key=lambda neighbor: min(heuristic_calc(neighbor.id, goal) for goal in goals))
            closest_neighbor_to_goal.visited = (True if closest_neighbor_to_goal.id not in goals else False)
            closest_heuristic = min(heuristic_calc(closest_neighbor_to_goal.id, goal) for goal in goals)
            if closest_heuristic > min(heuristic_calc(current.id, goal) for goal in goals):
                current = closest_neighbor_to_goal
                break
            
            closest_neighbor_to_goal.parent = current
            current = closest_neighbor_to_goal

    return None

# 3
def simulated_annealing(start, goals, max_temp): # performs a simulated annealing search from a start point to one of the goals
    def check_valid_solution(current):
        return (not current.visited) and (current.color == counties[start].color)

    def choose_random_neighbor(c):
        valid_neighbors = [neighbor for neighbor in c.neighbors if not neighbor.visited]
        if not valid_neighbors:
            return None
        neighbor = random.choice(valid_neighbors)
        return neighbor 

    def calculate_probabilities(neighbors, T):
        heuristic_values = [min(heuristic_calc(neighbor.id, goal) for goal in goals) for neighbor in neighbors]
        min_heuristic = min(heuristic_values)
        deltas = [hv - min_heuristic for hv in heuristic_values]
        probabilities = [math.exp(-delta / T) if delta > 0 else 1 for delta in deltas]
        return dict(zip(neighbors, probabilities))
    
    current = counties[start]
    min_temp = 0
    alpha = 0.95
    T = max_temp / 100
    path = []
    info = []
    first_iteration = True

    while T >= min_temp:
        if current.id in goals and check_valid_solution(current):
            return retracePath(current), info
        neighbors = current.neighbors
        neighbor_chances = calculate_probabilities(neighbors, T)
        if first_iteration:
            info.append({'current': current.id, 'neighbors': {n.id: neighbor_chances[n] for n in neighbors}})
            first_iteration = False
        next_neighbor = choose_random_neighbor(current)
        if next_neighbor is None:
            break
        delta = (min(heuristic_calc(next_neighbor.id, goal) for goal in goals) - 
                 min(heuristic_calc(current.id, goal) for goal in goals))
        if delta <= 0 or math.exp(-delta / T) > random.uniform(0, 1):
            next_neighbor.parent = current
            current.visited = True
            current = next_neighbor
            info.append({'current': current.id, 'neighbors': {n.id: neighbor_chances[n] for n in neighbors}})
        T *= alpha
    return path, info

# 4
def local_beam(start, goals, k=3): # performs a local k beam search from a start point to one of the goal points
    def get_k_most_close_nodes(current_node):
        current_neighbors = current_node.neighbors
        for neighbor in current_neighbors:
            neighbor.update_f(min(heuristic_calc(neighbor.id, goal) for goal in goals))
        return sorted(current_neighbors, key=lambda x: x.f)[:k]  # sorted by distance from goal and only the top k (closest)
    
    current = counties[start]  # initialize the current node
    current.g = 0  # initialize g value for the start node
    current.h = min(heuristic_calc(current.id, goal) for goal in goals)
    current.update_f()
    explored = set()  # visited and expanded
    frontier = [current]  # visited but not expanded
    first_iteration = True
    info = []
    while frontier:
        frontier.sort(key=lambda x: x.f)
        next = frontier.pop(0)
        if next.id in goals and not next.visited and next.color == counties[start].color:  # reached an unvisited goal node
            return retracePath(next), info
        next.visited = True
        next_neighbors = get_k_most_close_nodes(next)
        if first_iteration:
            info = next_neighbors.copy()
            first_iteration = False
        for neighbor in next_neighbors:
            if neighbor in explored:
                continue
            tentative_g = next.g + heuristic_calc(next.id, neighbor.id)
            if tentative_g < neighbor.g:
                neighbor.parent = next
                neighbor.g = tentative_g
                neighbor.h = min(heuristic_calc(neighbor.id, goal) for goal in goals)
                neighbor.update_f()
            if neighbor not in frontier:
                frontier.append(neighbor)
        explored.add(next)
    return None, None

# 5
def genetic_search(start, goals, population_size, max_generations): # performs a genetic search from a start point to one of the goal points
    def generate_random_path(start, goals, max_generations): # generates a random path from start to somewhere that is not a goal, in a maximum length of <max_generations>
        current = counties[start]
        path = [current]
        while current.id not in goals and len(path) < max_generations:
            current.visited = True
            unvisited_path_neighbors = [n for n in current.neighbors if n not in path]
            if not unvisited_path_neighbors: break
            next = random.choice(unvisited_path_neighbors)
            next.parent = current
            path.append(next)
            current = next
        return path

    def reset_visited(p): # resets visited for making random paths
        for county in p:
            county.visited = False

    def initiallize_population(start, goals, population_size): # initiallize a paths population in a length of <population_size>
        population = []
        for _ in range(population_size):
            path = generate_random_path(start, goals, max_generations)
            population.append(path)
            reset_visited(path)
        return population
    
    def check_valid_solution(node): # check whether the goal end solution is a valid goal
        return (not node.visited) and (node.color == counties[start].color)
    
    def fitness(path): # determines the fitness of a path, preferring shorter paths
        last_node = path[-1]
        if last_node.id in goals and check_valid_solution(last_node):
            return max_generations
        return max_generations - len(path)

    def select_parents(population, fitnesses): # select 2 parents paths based on their fitness as a probability
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            return random.choices(population, k = 2)
        probs = [fit / total_fitness for fit in fitnesses]
        parents = random.choices(population, probs, k = 2)
        return parents
    
    def crossover(p1, p2): # crossover two parents for making a new path child
        cross_index = random.randint(1, min(len(p1), len(p2) - 1))
        child = p1[:cross_index] + p2[cross_index:]
        for i in range(1, len(child)):
            child[i].parent = child[i-1]
        return child

    def mutate(p, rate = 0.1): # mutates the child path in a p(0.1)
        if random.random() < rate:
            mutate_index = random.randint(1, len(p) - 1)
            mutated_county = p[mutate_index]
            unvisited = [n for n in mutated_county.neighbors if n not in p]
            if unvisited:
                new_node = random.choice(unvisited)
                p[mutate_index] = new_node
                new_node.parent = p[mutate_index - 1] if mutate_index > 0 else None
        return p

    # main function of the genetic algorithm
    population = initiallize_population(start, goals, population_size)
    info = [[c.id for c in individual] for individual in population] # for printing on detailed_output the first population
    for _ in range(max_generations):
        fitnesses = [fitness(path) for path in population]
        if max(fitnesses) == max_generations:
            best_path = population[fitnesses.index(max(fitnesses))]
            return retracePath(best_path[-1]), info
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            new_population.extend([mutate(child1), mutate(child2)])
            for c in [child1, child2]:
                reset_visited(c)

    best_fitness_index = fitnesses.index(max(fitnesses))
    best_path = population[best_fitness_index]
    return None, None
### end block search algorithms

if __name__ == "__main__":
    pre_process() # pre process the adjacency file into county objects and into a dictionary

     # search methods
     # {
        # 1: A*,
        # 2: hill climbing,
        # 3: simulated annealing,
        # 4: local k beam,
        # 5: genetic algorithm
     # }

    search_method = 3
    detailed_output = 0
    pathes, infos = find_path(start_locations, goal_locations, search_method, detailed_output)
    print_paths(pathes, search_method, infos)