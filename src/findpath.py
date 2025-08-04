"""
This module provides grid-based pathfinding functionality using the A* algorithm
from the NetworkX library. It is designed to find the shortest path on a 2D grid
where cells can have different traversal costs (weights).
"""

import time
from typing import List, Tuple

import networkx as nx

# --- Graph Creation ---


def create_graph_from_grid(grid: List[List[int]]) -> nx.Graph:
    """
    Converts a 2D grid into a NetworkX graph suitable for pathfinding.

    Each grid cell becomes a node, and edges are created between adjacent
    (non-diagonal) cells. The weight of an edge is the average of the
    traversal costs of the two nodes it connects.

    Args:
        grid: A 2D list representing the grid. Each cell's value is its
              traversal cost (e.g., 1 for open, 25 for high-penalty).

    Returns:
        A NetworkX Graph object representing the grid.
    """
    G = nx.Graph()
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            node_id = (r, c)
            G.add_node(node_id, weight=grid[r][c])

            # Add edge to the right neighbor
            if c + 1 < cols:
                right_neighbor_id = (r, c + 1)
                edge_weight = (grid[r][c] + grid[r][c + 1]) / 2
                G.add_edge(node_id, right_neighbor_id, weight=edge_weight)

            # Add edge to the bottom neighbor
            if r + 1 < rows:
                bottom_neighbor_id = (r + 1, c)
                edge_weight = (grid[r][c] + grid[r + 1][c]) / 2
                G.add_edge(node_id, bottom_neighbor_id, weight=edge_weight)
    return G


# --- Heuristic for A* ---


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """
    Calculates the Manhattan distance heuristic for A* search.
    This is an admissible heuristic for a grid where movement is restricted
    to horizontal and vertical steps.

    Args:
        a: The first node (row, col).
        b: The second node (row, col).

    Returns:
        The Manhattan distance between the two nodes.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Pathfinding Execution ---


def find_shortest_path(
    graph: nx.Graph, start_node: Tuple[int, int], end_node: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """
    Finds the shortest path in a graph using the A* algorithm.

    Args:
        graph: The NetworkX graph to search.
        start_node: The starting node (row, col).
        end_node: The target node (row, col).

    Returns:
        A list of nodes representing the shortest path, or an empty list
        if no path is found.
    """
    try:
        start_time = time.time()
        path = nx.astar_path(
            graph, start_node, end_node, weight="weight", heuristic=manhattan_distance
        )
        end_time = time.time()
        # print(f"Path found in: {end_time - start_time:.4f} seconds.")
        return path
    except nx.NetworkXNoPath:
        # print(f"No path could be found from {start_node} to {end_node}.")
        return []


# --- Main Orchestration ---


def run_gridsearch(
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    points: List[List[int]],
    path_weight: int = 10,
) -> Tuple[List[Tuple[int, int]], List[List[int]], nx.Graph]:
    """
    Orchestrates the grid-based pathfinding process.

    This function takes a grid, converts it to a graph, finds the shortest
    path between two points, and updates the grid to mark the path as used.

    Args:
        start_node: The starting coordinate (row, col).
        end_node: The ending coordinate (row, col).
        points: The 2D grid with traversal costs.
        path_weight: The weight to assign to cells in the found path.

    Returns:
        A tuple containing:
        - The found path as a list of coordinates.
        - The updated grid with the path marked as high-penalty.
        - The graph used for pathfinding (note: it will be stale after points are updated).
    """

    # --- 2. Create Graph ---
    # Always create a new graph from the potentially updated points grid to ensure
    # pathfinding uses the latest weights.
    # print("Creating new graph from grid for pathfinding...")
    start_time = time.time()
    graph = create_graph_from_grid(points)
    end_time = time.time()
    # print(f"Graph creation took: {end_time - start_time:.4f} seconds.")

    # --- 3. Find Path ---
    # print(f"\nFinding shortest path from {start_node} to {end_node} using A*...")
    path = find_shortest_path(graph, start_node, end_node)

    # --- 4. Update Grid and Return ---
    if path:
        # print(f"Path length: {len(path) - 1} steps.")
        # print("Updating 'points' grid with the new path...")
        for r, c in path:
            points[r][c] = path_weight  # Mark path as used in the grid

    return path, points, graph
