import geopy.distance
import pandas as pd
import regex as re
import requests
import os

# defaults
adjacency_path = os.path.join("adjacency.csv")

# global vars
counties = {}

# input
start_locations = ['Blue, Washington County, UT', 'Blue, Chicot County, AR', 'Red, Fairfield County, CT'] 
goal_locations = ['Blue, San Diego County, CA', 'Blue, Bienville Parish, LA', 'Red, Rensselaer County, NY'] 

### classes   
class County:
    def __init__(self, txt:str):
        self.name = txt.split(",")[0]
        self.state = txt.split(",")[1]
        self.id = txt
        self.neighbors = []
        self.parent = None
        self.visited = False
        self.color = None

        self.g = float('inf') # distance from start
        self.h = float('inf') # heuristic distance from the goal
        self.f = 0
    
    def add_neighbor(self, neighbor):
        if f"{self.name}, {self.state}" != f"{neighbor.name}, {neighbor.state}":
            self.neighbors.append(neighbor)

    def update_f(self):
        self.f = self.g + self. h
### end block classes

### functions
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
    if search_method == 1:
        pathes = []
        for start in starting_locations:
            path = a_star(start, goal_locations)
            pathes.append(path)
    else:
        pass
    return pathes, detail_output

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

def heuristic_calc(start, goal): # returns the distance between two coordinates in km
    start_cord = get_county_coordinates(start)
    goal_cord = get_county_coordinates(goal)
    distance = geopy.distance.geodesic(start_cord, goal_cord).km
    return distance

def a_star(start, goals): # performs A* search from a starting location to one of the ending locations in the goal list
    frontier = [] # have been visited but not expanded 
    explored = set() # visited and expanded
    path = []

    def retracePath(c):
            if c.parent:
                retracePath(c.parent)
            path.append(c.id)

    # initiallizing the start
    start_county = counties[start]
    start_county.g = 0
    start_county.h = min(heuristic_calc(start_county.id, counties[goal].id) for goal in goals)
    start_county.update_f()
    frontier.append(counties[start])

    while frontier:
        current = min(frontier, key = lambda county:county.f) # takes the county with the minimum f
        if current.id in goals and not current.visited : # reached an unvisited goal node
            retracePath(current)
            return path
        
        frontier.remove(current)
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
    
def get_list_per_color(lst, pattern): # returns a list for the same color
    return [loc.replace('Red, ', '').replace('Blue, ', '') for loc in lst if re.search(pattern, loc)]

def print_pathes(reds, blues, detailed): # prints the pathes accordingly the detailed choice
    # adding the party color for each track
    red_paths = [[f"{county} (R)" for county in path] for path in reds]
    blue_paths = [[f"{county} (B)" for county in path] for path in blues]

    # building the print
    all_lists = red_paths + blue_paths
    max_length = max(len(lst) for lst in all_lists) # maximum numbers of rows
    previous_step = None
    
    for i in range(max_length):
        step_elements = []
        for lst in all_lists:
            if i < len(lst):
                step_elements.append(lst[i])
            else:
                step_elements.append(lst[-1])  # repeats the last element if the list is shorter
        print(f"{{{' ; '.join(step_elements)}}}")
        
        # calculates and prints heuristic if detailed is set to 1 and the row index is 1
        if detailed == 1 and previous_step is not None and i == 1:
            heuristics = []
            for current, prev in zip(step_elements, previous_step):
                current_loc = current.split(' (')[0]
                prev_loc = prev.split(' (')[0]
                heuristic_value = heuristic_calc(prev_loc, current_loc)
                heuristics.append(f"{heuristic_value:.2f}")
            print(f"Heuristic: {{{' ; '.join(heuristics)}}}")
        previous_step = step_elements
### functions 
if __name__ == "__main__":
    # initiallizing
    raw_df = read_neighbors_file(adjacency_path)
    counties = preparing_objects(raw_df) # dict: {county.name, county.state: county object}. this is the same dict as neighbors so it is enough for one of them    

    # dividing the starting locations and goal locations into different lists according to their colors
    red_starts = get_list_per_color(start_locations, r'Red,')
    red_goals = get_list_per_color(goal_locations, r'Red,')
    blue_starts = get_list_per_color(start_locations, r'Blue,')
    blue_goals = get_list_per_color(goal_locations, r'Blue,')

    # getting the pathes
    red_paths, detailed = find_path(red_starts, red_goals, 1, 1)
    blue_pathes, detailed = find_path(blue_starts, blue_goals, 1, 1)

    # printing
    ## i assume that detailed is equals in both pathes
    print_pathes(red_paths, blue_pathes, detailed) # it usually takes around 4 minutes to find all of the pathes and only then it prints them