import osmnx as ox
from typing import Tuple
import pandas as pd
from pyvis.network import Network
import utm
import copy

net = Network()


def get_id(row):
    return int(row.Index[1]) if hasattr(row, "Index") else int(row.index[0][1])


def get_lines():
    print("Starting get_lines()...")
    # victoria_australia_state_polygon = ox.geocode_to_gdf("2316741")
    print("Getting Gippsland polygon...")
    gippsland_vic_polygon = ox.geocode_to_gdf(query="R4246268", by_osmid=True)
    print("Getting power lines within polygon...")
    lines = ox.features.features_from_polygon(
        gippsland_vic_polygon.geometry.iloc[0], {"power": "line"}
    )
    print(f"Found {len(lines)} lines")
    return lines


def find_substations_near_point(lat: float, lon: float) -> pd.DataFrame:
    """
    Find the closest substation to a point

    Args:
        lat (float): Latitude of the point
        lon (float): Longitude of the point

    Returns:
        pd.DataFrame: DataFrame of the closest substation
    """
    print(f"Searching for closest substation to lat={lat:.6f}, lon={lon:.6f}")
    all_subs = ox.features.features_from_point((lat, lon), {"power": True}, dist=500)
    substations = all_subs[all_subs["power"] == "substation"]
    print(f"Found {len(substations)} substations near the point")

    # If no substations were found, return empty DataFrame
    if substations.empty:
        print("No substations found, returning empty DataFrame")
        return pd.DataFrame()

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

    # Return the closest substation as a single-row DataFrame
    return pd.DataFrame([closest])


def store_substation_as_node(
    substation: pd.DataFrame,
):
    if substation.empty:
        return

    # Handle DataFrame with multiple rows
    if len(substation) > 1:
        print(f"Processing multiple rows: {len(substation)}")
        # Process each row individually
        for i, (idx, row) in enumerate(substation.iterrows()):
            print(f"Processing row {i+1}/{len(substation)} with index {idx}")
            # Create a new DataFrame with a single row, preserving the index
            row_df = pd.DataFrame([row.to_dict()], index=[idx])
            store_substation_as_node(row_df)
        return

    # At this point we have a single row DataFrame
    # Check if node already exists
    if get_id(substation) in net.get_nodes():
        return

    # Project to a suitable projected CRS before calculating centroid
    # UTM zone 55S for Victoria, Australia
    projected_geom = (
        substation.geometry.iloc[0].to_crs(epsg=32755)
        if hasattr(substation.geometry.iloc[0], "to_crs")
        else substation.geometry.iloc[0]
    )

    # Get centroid coordinates
    x, y = utm.from_latlon(projected_geom.centroid.y, projected_geom.centroid.x)[0:2]
    y *= -1
    print(x, y)

    # Add node
    net.add_node(get_id(substation), label=substation.name.iloc[0], x=x, y=y)


def find_substations_at_end_of_a_line(
    line: pd.DataFrame,
) -> None:
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
    first_substations = find_substations_near_point(*first_coord)
    last_substations = find_substations_near_point(*last_coord)
    # connect the substations to the line
    # Merge the two dataframes, dropping duplicates
    combined_substations = pd.concat(
        [first_substations, last_substations]
    ).drop_duplicates()
    # and remove any subs in the results which are the source substation
    if combined_substations.empty:
        return

    # add nodes
    store_substation_as_node(first_substations)
    store_substation_as_node(last_substations)

    for sub1, sub2 in zip(
        first_substations.itertuples(), last_substations.itertuples()
    ):
        net.add_edge(get_id(sub1), get_id(sub2))

    return combined_substations


count = 0
for line in get_lines().itertuples():
    print(line)
    find_substations_at_end_of_a_line(line)
    count += 1
    if count > 20:
        break

# normalise each node
for node in net.nodes:
    node["x"] /= 100
    node["y"] /= 100
net.toggle_physics(False)
net.show("test.html", notebook=False)
