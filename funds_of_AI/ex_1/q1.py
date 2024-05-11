import pandas as pd
import numpy as np
import os

# defaults
neighbors = os.path.join("adjacency.csv")

# classes
class County:
    def __init__(self, txt:str):
        self.name = txt.split(",")[0]
        self.state = txt.split(",")[1]
        self.neighbors = []

        self.g = np.inf
        self.h = np.inf
        self.f = self.g + self.h
    
    def add_neighbor(self, neighbor):
        if self.name != neighbor.name and self.state != neighbor.state:
            self.neighbors.append(neighbor)


# reads the csv file and converts it into a dataframe
def read_neighborgs(file_name: os) -> pd.DataFrame:
    return pd.read_csv(file_name)

# returns the unique values in a df[col_name]
def get_unique_list(df: pd.DataFrame, col_name: str) -> list:
    return list(set(df[col_name]))

# gets a list of text and returns a list of County objects
def make_object_list(lst: list) -> list:
    return [County(c) for c in lst]

# finds the shortest path from the starting locations to the goal location
# using a search method - an integer
# details_output = {F: not showing the first iteration, T: showing the first iteration}
def find_path(starting_locations ,goal_locations ,search_method: int, detail_output: bool):
    pass

if __name__ == "__main__":
    raw_df = read_neighborgs(neighbors)
    (unique_counties, unique_neighbors) = (get_unique_list(raw_df, 'countyname'), get_unique_list(raw_df, 'neighborname')) # to hold the unique text values of counties and neighbors
    (county_objects, neighbor_objects) = (make_object_list(unique_counties), make_object_list(unique_neighbors)) # transforming the text lists into a County objects list

    # having a dict that represents the unique counties.
    # key == county.name
    # value = county object
    counties_dict = {county.name: county for county in county_objects}  