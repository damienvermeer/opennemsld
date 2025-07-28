import osmnx as ox
from typing import Tuple
import pandas as pd
from pyvis.network import Network
import utm
import copy

net = Network()


def get_id(row):
    return int(row.Index[1]) if hasattr(row, "Index") else int(row.index[0][1])


def lat_long_to_x_y(point: "Point") -> Tuple[float, float]:
    x, y = utm.from_latlon(point.y, point.x)[0:2]
    return x, -y


def find_substations_near_point(
    lat: float, lon: float, distance: int = 20000
) -> pd.DataFrame:
    """
    Find substations near a point

    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        distance (int, optional): Distance in meters. Defaults to 25.

    Returns:
        pd.DataFrame: DataFrame of substations and dataframe of lines
    """
    all_subs = ox.features.features_from_point(
        (lat, lon), {"power": True}, dist=distance
    )
    # add the substation as a node, use its ID as the node ID and its name as the label
    for sub in all_subs[all_subs["power"] == "substation"].itertuples():
        # check if the substation is already in the network
        if get_id(sub) not in net.get_nodes():
            x, y = lat_long_to_x_y(sub.geometry.centroid)
            net.add_node(get_id(sub), label=sub.name, x=x, y=y)
    # return all substations and all lines
    return (
        all_subs[all_subs["power"] == "substation"],
        all_subs[all_subs["power"] == "line"],
    )


def find_substations_at_end_of_a_line(
    source_sub: pd.DataFrame,
    line: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Find substations at the end of a line

    Args:
        line (pd.DataFrame): DataFrame of a line

    Returns:
        pd.DataFrame: DataFrame of substations
    """
    # extract the first and last coords
    first_coord = line.geometry.coords[0][::-1]
    last_coord = line.geometry.coords[-1][::-1]
    # Get substations at both ends of the line
    first_substations, _ = find_substations_near_point(*first_coord)
    last_substations, _ = find_substations_near_point(*last_coord)
    # connect the substations to the line
    # Merge the two dataframes, dropping duplicates
    combined_substations = pd.concat(
        [first_substations, last_substations]
    ).drop_duplicates()
    # and remove any subs in the results which are the source substation
    if source_sub.empty:
        return combined_substations

    # This assumes source_sub has one row.
    source_sub_id = source_sub.index[0]
    combined_substations = combined_substations[
        combined_substations.index != source_sub_id
    ]
    # now add an edge from the source substation to each of the substations
    for sub in combined_substations.itertuples():
        net.add_edge(get_id(source_sub), get_id(sub))
    return combined_substations


# test start with bairnsdale ZSS
substations = []
subs_at_loc, lines_at_loc = find_substations_near_point(-37.808466, 147.647892)
substations.append(subs_at_loc)
for line in lines_at_loc.itertuples():
    print(line)
    subs = find_substations_at_end_of_a_line(subs_at_loc, line)
    substations.append(subs)


# normalise each node
for node in net.nodes:
    node["x"] /= 100
    node["y"] /= 100
net.toggle_physics(False)
net.show("test.html", notebook=False)
