import geopy.distance
import pandas as pd
import regex as re
import requests
import os.path

# defaults
adjacency_path = os.path.join("test_adjacency.csv")

# global vars
counties = {}
detailed = False

# input
start_locations = ['Red, Tel Aviv, DN', 'Blue, Imperial County, CA'] # , 'Red, Fairfield County, CT'
goal_locations = ['Red, San Diego County, CA' , 'Blue, Mohave County, AZ'] # , 'Red, Rensselaer County, NY'

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
    detailed = detail_output
    if search_method == 1:
        pathes = []
        for start in starting_locations:
            path = a_star(start, goal_locations)
            if not path:
                pathes.append('No path found.')
            else: pathes.append(path)
    else:
        pass
    return pathes

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
        current = min(frontier, key = lambda county: county.f) # takes the county with the minimum f
        if current.id in goals and not current.visited and current.color == start_county.color : # reached an unvisited goal node
            retracePath(current)
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
    
def get_list_per_color(lst, pattern): # returns a list for the same color
    return [loc.replace('Red, ', '').replace('Blue, ', '') for loc in lst if re.search(pattern, loc)]

def print_paths(paths):
    # Format the paths with color indications
    all_lists = [
        [f"{county} (R)" if counties[county].color == 'Red' else f"{county} (B)" for county in path] if path != "No path found." else ["No path found."]
        for path in paths
    ]

    # Insert start locations as the first step
    start_locations_colored = [f"{county} (R)" if counties[county].color == 'Red' else f"{county} (B)" for county in start_locations]
    for lst in all_lists:
        lst.insert(0, start_locations_colored[all_lists.index(lst)])

    # Determine the maximum length of the lists
    max_length = max(len(lst) for lst in all_lists)
    previous_step = None

    # Iterate through the steps and print accordingly
    for i in range(max_length):
        step_elements = []
        for lst in all_lists:
            if i < len(lst):
                step_elements.append(lst[i])
            else:
                step_elements.append(lst[-1])  # Repeats the last element if the list is shorter

        if i == 0:
            print(f"{{{' ; '.join(step_elements)}}}")
        elif i == 1 and detailed == 1:
            heuristics = []
            for current, prev in zip(step_elements, previous_step):
                if "No path found." in any [current, prev]:
                    heuristics.append("N/A")
                else:
                    current_loc = current.split(' (')[0]
                    prev_loc = prev.split(' (')[0]
                    heuristic_value = heuristic_calc(prev_loc, current_loc)
                    heuristics.append(f"{heuristic_value:.2f}")
            print(f"Heuristic: {{{' ; '.join(heuristics)}}}")
            previous_step = step_elements
        else:
            if detailed == 1 or i > 1:
                print(f"{{{' ; '.join(step_elements)}}}")
            previous_step = step_elements

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

### functions 
if __name__ == "__main__":
    raw_df = read_neighbors_file(adjacency_path)
    counties = preparing_objects(raw_df) # dict: {county.name, county.state: county object}. this is the same dict as neighbors so it is enough for one of them    
    assigning_colors_to_counties(start_locations, goal_locations)
    pathes = find_path(start_locations, goal_locations, 1, 1)
    print_paths(pathes) # 