import osmnx as ox
from typing import Tuple
import pandas as pd
from pyvis.network import Network
import utm
import copy
from typing import List
import networkx as nx

pd.options.mode.chained_assignment = None  # default='warn'
net = Network(directed=True)
net.set_edge_smooth("dynamic")

# gippsland "R4246268"
# victoria "R2316741"


def get_lines(osmid="R4246268"):
    source_poly = ox.geocode_to_gdf(query=osmid, by_osmid=True)
    lines = ox.features.features_from_polygon(
        source_poly.geometry.iloc[0], {"power": "line"}
    )
    return lines


def get_substations(osmid="R2316741"):
    source_poly = ox.geocode_to_gdf(query=osmid, by_osmid=True)
    lines = ox.features.features_from_polygon(
        source_poly.geometry.iloc[0], {"power": "substation"}
    )
    return lines


def find_relations_from_way(way_id: int) -> List:
    import requests
    import json

    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way({way_id});
    <;
    out meta;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})

    if response.status_code == 200:
        # get a list of all members
        elements = response.json()["elements"]
        if len(elements) == 0:
            return []
        members = elements[0]["members"]
        return [x["ref"] for x in members if x["type"] == "way"]
    else:
        print(f"Error: {response.status_code}")
        return []


def lat_long_to_x_y(point: "Point") -> Tuple[float, float]:
    x, y = utm.from_latlon(point.y, point.x)[0:2]
    return x, -y


def get_id(row):
    try:
        return int(row.index)
    except Exception as e:
        return int(row.Index) if hasattr(row, "Index") else int(row.index[0])


def find_closest_substation_to_point(
    lat: float, lon: float, threshold: int = 20
) -> pd.DataFrame:
    """
    Find substations near a point

    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point
        distance (int, optional): Distance in meters. Defaults to 25.

    Returns:
        pd.DataFrame: DataFrame of substations
    """
    all_subs = ox.features.features_from_point((lat, lon), {"power": True}, dist=500)
    substations = all_subs[all_subs["power"] == "substation"]

    # If no substations were found, return empty DataFrame
    if substations.empty:
        return pd.DataFrame()

    substations.index = substations.index.droplevel(0)

    # Create a point object from lat/lon
    from shapely.geometry import Point

    point = Point(lon, lat)  # Note: shapely uses (x,y) which is (lon,lat)

    # Calculate distance for each substation and add as column
    substations["distance"] = substations.geometry.apply(
        lambda geom: geom.distance(point)
    )

    # Sort by distance and return closest
    closest = substations.sort_values(by="distance").iloc[0]
    print(f"Found closest substation at distance {closest['distance']:.6f}")

    if closest["distance"] > threshold:
        return pd.DataFrame()

    # Return the closest substation as a single-row DataFrame
    return pd.DataFrame([closest])


def add_sub_as_node(sub: pd.DataFrame) -> None:
    if get_id(sub) not in net.get_nodes():
        x, y = lat_long_to_x_y(sub.geometry.centroid)
        net.add_node(get_id(sub), label=sub.name, x=x, y=y)


import requests


def get_first_last_lat_lon(way_id):
    """
    Given the ID of an OpenStreetMap way, this function returns the latitude and
    longitude of the first and last nodes of that way.

    Args:
        way_id (int): The ID of the OpenStreetMap way.

    Returns:
        tuple: A tuple containing two tuples. The first inner tuple contains the
               latitude and longitude of the first node, and the second inner
               tuple contains the latitude and longitude of the last node.
               Returns (None, None) if the way is not found or an error occurs.
    """
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way({way_id});
    out geom;
    """

    try:
        response = requests.get(overpass_url, params={"data": overpass_query})
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()

        if "elements" in data and len(data["elements"]) > 0:
            way_element = data["elements"][0]
            if "geometry" in way_element and len(way_element["geometry"]) > 0:
                first_node = way_element["geometry"][0]
                last_node = way_element["geometry"][-1]

                first_lat_lon = (first_node["lat"], first_node["lon"])
                last_lat_lon = (last_node["lat"], last_node["lon"])

                return first_lat_lon, last_lat_lon
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return None, None


def process_line_result_into_edge(line_way: pd.DataFrame) -> None:
    print(f"Processing line {get_id(line_way)}")
    # first check if both ends of the line_way are the same sub
    near_coord = line.geometry.coords[0][::-1]
    far_coord = line.geometry.coords[-1][::-1]
    # Get substations at both ends of the line
    near_sub = find_closest_substation_to_point(*near_coord)
    far_sub = find_closest_substation_to_point(*far_coord)

    # check for type 0 = near and far sub are the same (is like
    # ... a busbar or some internal to sub line segment)
    if not near_sub.empty and not far_sub.empty and near_sub.equals(far_sub):
        print(
            f"Line/way ID {get_id(line)} is type 0 - a busbar or internal to sub line segment"
        )
        return

    # else check for type 1 = there is one near sub, one far sub BUT
    # ... they are not the same sub
    if (
        not near_sub.empty
        and not far_sub.empty
        and len(near_sub) == 1
        and len(far_sub) == 1
    ):
        print(f"Line/way ID {get_id(line)} is type 1 - a line between two substations")
        near_sub_id = get_id(near_sub)
        far_sub_id = get_id(far_sub)
        if near_sub_id not in net.get_nodes():
            net.add_node(near_sub_id)
        if far_sub_id not in net.get_nodes():
            net.add_node(far_sub_id)
        net.add_edge(near_sub_id, far_sub_id)
        return

    # else check for type 2 = this line might be part of a relation which describes the
    # ... line
    if not near_sub.empty or not far_sub.empty:
        # lat/long pairs for searching
        latlongs = [near_coord, far_coord]
        # see if this belongs to any parent relation
        way_ids_related_to_this_line = find_relations_from_way(get_id(line_way))
        for way_id in [
            x for x in way_ids_related_to_this_line if x is not get_id(line_way)
        ]:
            lat1lon1, lat2lon2 = get_first_last_lat_lon(way_id)
            if lat1lon1 is not None and lat2lon2 is not None:
                latlongs.append(lat1lon1)
                latlongs.append(lat2lon2)
            else:
                print(f"Could not find lat/long for way ID {way_id}")

        # remove duplicates
        latlongs = list(set(latlongs))

        # now find the closest substation to each lat/long pair and combine them
        # into a single dataframe
        subs = []
        for latlong in latlongs:
            sub = find_closest_substation_to_point(*latlong)
            if not sub.empty:
                subs.append(sub)
        # now check for type 3s
        if len(subs) == 2:
            print(
                f"Line/way ID {get_id(line)} is type 3 - a line between two substations (across multiple ways)"
            )
            if get_id(subs[0]) not in net.get_nodes():
                net.add_node(get_id(subs[0]))
            if get_id(subs[1]) not in net.get_nodes():
                net.add_node(get_id(subs[1]))
            net.add_edge(get_id(subs[0]), get_id(subs[1]))
            return

    print(f"Line/way ID {get_id(line)} is type ? - ????")


# change the multi-index into the 2nd index entry for each

all_lines = get_lines()
all_lines.index = all_lines.index.droplevel(0)
all_subs = get_substations()
all_subs.index = all_subs.index.droplevel(0)

for sub in all_subs.itertuples():
    print(f"Adding substation {get_id(sub)}")
    add_sub_as_node(sub)


# # dump to csv after sorting via index id
# all_lines.sort_index().to_csv("victoria_lines.csv")
# all_subs.sort_index().to_csv("victoria_subs.csv")
maxx = 200
for line in all_lines.itertuples():
    process_line_result_into_edge(line)
    maxx -= 1
    if maxx < 0:
        break

# # normalise each node
for node in net.nodes:
    if "x" in node:
        node["x"] /= 100
    if "y" in node:
        node["y"] /= 100
net.toggle_physics(False)
net.show("victoria.html", notebook=False)


# for line in all_lines.itertuples():
#     # check if part of a parent relation
#     for x in find_relations_from_way(get_id(line)):
#         print(x)
#         print(ox.geocode_to_gdf(query=f"W{x}", by_osmid=True))
#     x = 0


#
