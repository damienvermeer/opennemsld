"""
This module provides grid-based pathfinding functionality using the A* algorithm
from the NetworkX library. It is designed to find the shortest path on a 2D grid
where cells can have different traversal costs (weights).
"""

from typing import List, Tuple, Union

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
            G.add_node(node_id)

            # Add edge to the right neighbor
            if c + 1 < cols:
                right_neighbor_id = (r, c + 1)
                edge_weight = 1 + (grid[r][c] + grid[r][c + 1]) / 2
                G.add_edge(node_id, right_neighbor_id, weight=edge_weight)

            # Add edge to the bottom neighbor
            if r + 1 < rows:
                bottom_neighbor_id = (r + 1, c)
                edge_weight = 1 + (grid[r][c] + grid[r + 1][c]) / 2
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
    graph: nx.Graph,
    start_node: Tuple[int, int],
    end_node: Tuple[int, int],
    heuristic=None,
) -> List[Tuple[int, int]]:
    """
    Finds the shortest path in a graph using the A* algorithm.

    Args:
        graph: The NetworkX graph to search.
        start_node: The starting node.
        end_node: The target node.
        heuristic: The heuristic function for A*. Defaults to Manhattan distance.

    Returns:
        A list of nodes representing the shortest path, or an empty list
        if no path is found.
    """
    if heuristic is None:
        heuristic = manhattan_distance

    try:
        path = nx.astar_path(
            graph, start_node, end_node, weight="weight", heuristic=heuristic
        )
        return path
    except nx.NetworkXNoPath:
        return []


# --- Pathfinding Helpers ---


def _calculate_congestion_penalty(
    iterations: int, i: int, base_penalty: float
) -> float:
    """Calculates a quadratically scaled congestion penalty.

    The penalty increases with each iteration to encourage convergence.

    Args:
        iterations: The total number of refinement iterations.
        i: The current iteration index.
        base_penalty: The base penalty value to be scaled.

    Returns:
        The calculated penalty for the current iteration.
    """
    if iterations <= 1:
        return base_penalty
    # Use a quadratic scaling factor from 0 to 1.
    scaling_factor = (i / (iterations - 1)) ** 2
    # The max penalty multiplier is set to be aggressive in the final iterations.
    max_penalty_multiplier = 1.0
    return base_penalty * max_penalty_multiplier * scaling_factor


def _get_busbar_edges(
    graph: nx.Graph,
    points: List[List[int]],
    grid_owners: List[List[str]],
    busbar_weight: int,
) -> dict:
    """Identifies all edges in the graph that correspond to busbars.

    Args:
        graph: The main pathfinding graph.
        points: The 2D grid of pathfinding weights.
        grid_owners: The 2D grid of owner IDs for each point.
        busbar_weight: The weight value that identifies a busbar point.

    Returns:
        A dictionary mapping busbar edges (as tuples of nodes) to their
        owner ID.
    """
    busbar_edges = {}
    if busbar_weight is None:
        return busbar_edges
    for u, v in graph.edges():
        if points[u[0]][u[1]] == busbar_weight and points[v[0]][v[1]] == busbar_weight:
            edge = tuple(sorted((u, v)))
            # Assume owner is same for both ends of a busbar segment
            owner = grid_owners[u[0]][u[1]]
            busbar_edges[edge] = owner
    return busbar_edges


def _calculate_busbar_crossing_penalties(
    busbar_edges: dict,
    start_owner: tuple,
    end_owner: tuple,
    busbar_crossing_penalty: int,
) -> dict:
    """Calculates penalties for a path crossing various busbars.

    This function determines the penalty for crossing each busbar edge based
    on the ownership of the path and the busbar.

    Args:
        busbar_edges: A dictionary of busbar edges and their owners.
        start_owner: The owner tuple (substation_name, owner_id) of the
            path's start point.
        end_owner: The owner tuple of the path's end point.
        busbar_crossing_penalty: The high penalty value for an illegal crossing.

    Returns:
        A dictionary mapping busbar edges to their calculated penalty value.
    """
    edge_penalties = {}
    for edge, owner in busbar_edges.items():
        if not owner:
            edge_penalties[edge] = 1
            continue

        bus_sub_name, bus_owner_id = owner

        # An intra-substation connection
        if start_owner[0] == end_owner[0]:
            path_sub_name = start_owner[0]
            # If path is inside one sub, but crosses busbar of another sub
            if bus_sub_name != path_sub_name:
                edge_penalties[edge] = busbar_crossing_penalty
            # If path crosses a busbar within the same sub, but not belonging to start/end owners
            elif bus_owner_id not in (start_owner[1], end_owner[1]):
                edge_penalties[edge] = busbar_crossing_penalty
            else:
                # Crossing its own busbar, small penalty
                edge_penalties[edge] = 1
        # An inter-substation connection
        else:
            path_sub_names = {start_owner[0], end_owner[0]}
            # If it crosses a busbar of an unrelated sub
            if bus_sub_name not in path_sub_names:
                edge_penalties[edge] = busbar_crossing_penalty
    return edge_penalties


def _calculate_congestion_usage(
    all_paths: list, current_path_idx: int
) -> tuple[dict, dict]:
    """Calculates node and edge usage by all paths except the current one.

    Args:
        all_paths: The list of all current paths.
        current_path_idx: The index of the path to be excluded from the
            calculation (the one being rerouted).

    Returns:
        A tuple containing two dictionaries:
        - node_usage: Maps nodes to their usage count.
        - edge_usage: Maps edges to their usage count.
    """
    node_usage = {}
    edge_usage = {}
    for other_req_idx, other_path in enumerate(all_paths):
        if current_path_idx == other_req_idx or not other_path:
            continue

        # Penalize intermediate nodes to discourage paths from crossing.
        for node in other_path[1:-1]:
            node_usage[node] = node_usage.get(node, 0) + 1

        for j in range(len(other_path) - 1):
            u, v = other_path[j], other_path[j + 1]
            edge = tuple(sorted((u, v)))
            edge_usage[edge] = edge_usage.get(edge, 0) + 1
    return node_usage, edge_usage


def _apply_penalties_to_graph(
    graph: nx.Graph,
    edge_usage: dict,
    node_usage: dict,
    current_penalty: float,
    start_node: tuple,
    end_node: tuple,
) -> list:
    """Temporarily adds penalties to graph edges based on usage.

    Args:
        graph: The `nx.Graph` to modify.
        edge_usage: A dictionary mapping edges to their usage count.
        node_usage: A dictionary mapping nodes to their usage count.
        current_penalty: The scaled penalty value for the current iteration.
        start_node: The start node of the path being rerouted.
        end_node: The end node of the path being rerouted.

    Returns:
        A list of (edge, penalty_value) tuples that were applied, so they
        can be reverted later.
    """
    applied_penalties = []
    for edge, count in edge_usage.items():
        penalty = (count**2) * current_penalty
        if graph.has_edge(*edge):
            graph.edges[edge]["weight"] += penalty
            applied_penalties.append((edge, penalty))

    for node, count in node_usage.items():
        if node in (start_node, end_node):
            continue

        if graph.has_node(node):
            penalty = (count**2) * current_penalty
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted((node, neighbor)))
                if graph.has_edge(*edge):
                    graph.edges[edge]["weight"] += penalty
                    applied_penalties.append((edge, penalty))
    return applied_penalties


def _remove_penalties_from_graph(graph: nx.Graph, applied_penalties: list):
    """Removes temporary penalties from graph edges.

    Args:
        graph: The `nx.Graph` to modify.
        applied_penalties: A list of (edge, penalty_value) tuples to revert.
    """
    for edge, penalty in applied_penalties:
        if graph.has_edge(*edge):
            graph.edges[edge]["weight"] -= penalty


def _block_connection_nodes(
    graph: nx.Graph, all_connection_nodes: set, start_node: tuple, end_node: tuple
) -> list:
    """Temporarily blocks access to connection nodes not part of the current path.

    This prevents paths from routing through the connection points of other
    unrelated lines.

    Args:
        graph: The `nx.Graph` to modify.
        all_connection_nodes: A set of all connection nodes in the graph.
        start_node: The start node of the current path, which should not be blocked.
        end_node: The end node of the current path, which should not be blocked.

    Returns:
        A list of (edge, original_weight) tuples for the edges that were
        blocked, so they can be restored.
    """
    blocked_edges = []
    if not all_connection_nodes:
        return blocked_edges

    nodes_to_block = all_connection_nodes - {start_node, end_node}
    for node in nodes_to_block:
        if graph.has_node(node):
            for neighbor in list(graph.neighbors(node)):
                edge_tuple = tuple(sorted((node, neighbor)))
                if graph.has_edge(*edge_tuple):
                    original_weight = graph.edges[edge_tuple]["weight"]
                    blocked_edges.append((edge_tuple, original_weight))
                    graph.edges[edge_tuple]["weight"] = float("inf")
    return blocked_edges


def _unblock_connection_nodes(graph: nx.Graph, blocked_edges: list):
    """Restores access to previously blocked connection nodes.

    Args:
        graph: The `nx.Graph` to modify.
        blocked_edges: A list of (edge, original_weight) tuples to restore.
    """
    for edge, original_weight in blocked_edges:
        if graph.has_edge(*edge):
            graph.edges[edge]["weight"] = original_weight


def _create_out_of_bounds_heuristic(bounds: tuple):
    """Creates an A* heuristic that penalizes paths going outside specified bounds.

    Args:
        bounds: A tuple (min_x, min_y, max_x, max_y) defining the allowed area.

    Returns:
        A heuristic function for use with `nx.astar_path`.
    """
    min_x, min_y, max_x, max_y = bounds

    def out_of_bounds_heuristic(u, v):
        dist = manhattan_distance(u, v)
        # u is the current node in the search. (row, col) -> (y, x)
        y, x = u
        if x < min_x or x > max_x or y < min_y or y > max_y:
            dist += 1000000  # Very large penalty
        return dist

    return out_of_bounds_heuristic


def _try_straighten_one_corner(
    path: list,
    i: int,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> Union[Tuple[int, int], None]:
    """Attempts to straighten a single corner in a path.

    It looks for an "L-shaped" segment and tries to replace it with an
    alternative L-shape if it has the same weight and doesn't cause collisions.

    Args:
        path: The path being modified.
        i: The index of the corner node in the path to check.
        graph: The main pathfinding graph.
        other_occupied_nodes: A set of nodes occupied by other paths.
        other_occupied_edges: A set of edges occupied by other paths.

    Returns:
        The new corner node if a valid straightening was found, otherwise None.
    """
    p_prev, p_curr, p_next = path[i - 1], path[i], path[i + 1]

    # Check for a corner (not collinear).
    if p_prev[0] == p_curr[0] == p_next[0] or p_prev[1] == p_curr[1] == p_next[1]:
        return None

    # Find the alternative node for a rectangular detour.
    p_alt = (
        p_prev[0] + p_next[0] - p_curr[0],
        p_prev[1] + p_next[1] - p_curr[1],
    )

    if not graph.has_node(p_alt) or p_alt in other_occupied_nodes or p_alt in path:
        return None

    edge1_alt = tuple(sorted((p_prev, p_alt)))
    edge2_alt = tuple(sorted((p_alt, p_next)))

    if (
        not graph.has_edge(*edge1_alt)
        or not graph.has_edge(*edge2_alt)
        or edge1_alt in other_occupied_edges
        or edge2_alt in other_occupied_edges
    ):
        return None

    # Check if weights are equal.
    edge1_orig = tuple(sorted((p_prev, p_curr)))
    edge2_orig = tuple(sorted((p_curr, p_next)))
    weight_orig = graph.edges[edge1_orig]["weight"] + graph.edges[edge2_orig]["weight"]
    weight_alt = graph.edges[edge1_alt]["weight"] + graph.edges[edge2_alt]["weight"]

    # Tie-break by preferring the lexicographically smaller node to prevent flipping.
    if abs(weight_orig - weight_alt) < 1e-9 and p_alt < p_curr:
        return p_alt

    return None


# --- Path Straightening ---


def _straighten_paths(
    all_paths: List[List[Tuple[int, int]]], graph: nx.Graph, iterations: int = 3
) -> List[List[Tuple[int, int]]]:
    """
    Post-processes a set of paths to reduce corners by "squaring them off".

    This function iterates through each path and looks for "L-shaped" segments
    (corners). It attempts to replace the corner with an alternative L-shape
    if the new path segment has the same weight and does not collide with any
    other existing paths. This process is repeated to allow for cascading
    improvements.

    Args:
        all_paths: A list of paths to be straightened.
        graph: The graph on which the paths exist, used for weight lookups.
        iterations: The number of times to repeat the straightening process.

    Returns:
        A new list of paths with corners potentially straightened.
    """
    paths_to_modify = [list(p) for p in all_paths]

    for iter_num in range(iterations):
        print(
            f"Step 5.1.4.{iter_num + 1}: Straightening iteration {iter_num + 1}/{iterations}..."
        )
        paths_changed_in_iteration = False

        indexed_paths_to_process = sorted(
            enumerate(paths_to_modify), key=lambda x: len(x[1]), reverse=True
        )

        # Build the set of all occupied nodes and edges once per iteration
        all_occupied_nodes = {node for p in paths_to_modify if p for node in p}
        all_occupied_edges = {
            tuple(sorted((p[i], p[i + 1])))
            for p in paths_to_modify
            if p
            for i in range(len(p) - 1)
        }

        for original_idx, path in indexed_paths_to_process:
            if not path or len(path) < 3:
                continue

            # Loop until no more changes can be made to this path
            while True:
                made_change_in_pass = False

                # Create occupied sets for *other* paths for collision detection.
                current_path_nodes = set(path)
                current_path_edges = {
                    tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)
                }
                other_occupied_nodes = all_occupied_nodes - current_path_nodes
                other_occupied_edges = all_occupied_edges - current_path_edges

                for i in range(1, len(path) - 1):
                    p_alt = _try_straighten_one_corner(
                        path, i, graph, other_occupied_nodes, other_occupied_edges
                    )

                    if p_alt:
                        old_node = path[i]
                        p_prev = path[i - 1]
                        p_next = path[i + 1]
                        path[i] = p_alt
                        made_change_in_pass = True
                        paths_changed_in_iteration = True

                        # Update the global occupied sets for subsequent paths
                        all_occupied_nodes.remove(old_node)
                        all_occupied_nodes.add(p_alt)
                        all_occupied_edges.remove(tuple(sorted((p_prev, old_node))))
                        all_occupied_edges.remove(tuple(sorted((old_node, p_next))))
                        all_occupied_edges.add(tuple(sorted((p_prev, p_alt))))
                        all_occupied_edges.add(tuple(sorted((p_alt, p_next))))

                        # Restart scan on the now-modified path
                        break

                if not made_change_in_pass:
                    break

        if not paths_changed_in_iteration:
            break

    return paths_to_modify


# --- Main Orchestration ---


def run_all_gridsearches(
    path_requests: List[dict],
    points: List[List[int]],
    grid_owners: List[List[str]],
    iterations: int = 5,
    congestion_penalty_increment: float = 2.0,
    all_connection_nodes: set = None,
    busbar_weight: int = None,
    busbar_crossing_penalty: int = 100000,
) -> List[List[Tuple[int, int]]]:
    """
    Finds paths for a series of requests, aiming to minimize total congestion.

    This is an enhanced version of a sequential rip-up and reroute algorithm.
    Key features:
    1.  **Longest Path First**: It prioritizes routing longer paths first, as
        they are typically harder to place.
    2.  **Iterative Refinement**: It iteratively refines paths. In each
        iteration, it reroutes each path one by one on a graph that is
        penalized by the congestion caused by all other paths.
    3.  **Increasing Penalty**: The penalty for congestion increases
        quadratically with each iteration. It starts very low to allow
        for more chaotic path exploration and ramps up aggressively towards
        the end to force convergence on a low-congestion solution.

    Args:
        path_requests: A list of (start_node, end_node) tuples.
        points: The initial 2D grid with traversal costs.
        grid_owners: A 2D grid storing the owner of each cell.
        iterations: The number of times to iterate the pathfinding process.
        congestion_penalty_increment: The base penalty added to a graph edge
                                      for each path crossing it. This value
                                      is scaled up with each iteration.
        all_connection_nodes: A set of all connection nodes to be avoided.
        busbar_weight: The grid value identifying a busbar.
        busbar_crossing_penalty: The penalty for crossing a busbar incorrectly.

    Returns:
        A list of paths, where each path is a list of coordinates, in the
        same order as the input path_requests.
    """
    print("Step 5.1.1: Creating base pathfinding graph...")
    graph = create_graph_from_grid(points)

    # --- Sort requests to route longest paths first ---
    indexed_requests = sorted(
        enumerate(path_requests),
        key=lambda x: manhattan_distance(x[1]["start"], x[1]["end"]),
        reverse=True,
    )
    sorted_requests = [req for i, req in indexed_requests]

    print("Step 5.1.2: Performing initial routing...")
    all_paths = [
        find_shortest_path(graph, req["start"], req["end"]) for req in sorted_requests
    ]

    # --- Iteratively refine paths ---
    busbar_edges = _get_busbar_edges(graph, points, grid_owners, busbar_weight)

    for i in range(iterations):
        print(f"Step 5.1.3: Refining paths (iteration {i + 1}/{iterations})...")
        current_penalty = _calculate_congestion_penalty(
            iterations, i, congestion_penalty_increment
        )

        for req_idx, current_request in enumerate(sorted_requests):
            start_node = current_request["start"]
            end_node = current_request["end"]

            # --- Calculate Penalties ---
            busbar_penalties = _calculate_busbar_crossing_penalties(
                busbar_edges,
                current_request["start_owner"],
                current_request["end_owner"],
                busbar_crossing_penalty,
            )
            node_usage, congestion_usage = _calculate_congestion_usage(
                all_paths, req_idx
            )
            edge_usage = {**congestion_usage}
            for edge, penalty in busbar_penalties.items():
                edge_usage[edge] = edge_usage.get(edge, 0) + penalty

            # --- Apply Penalties and Blockers ---
            applied_penalties = _apply_penalties_to_graph(
                graph, edge_usage, node_usage, current_penalty, start_node, end_node
            )
            blocked_edges = _block_connection_nodes(
                graph, all_connection_nodes, start_node, end_node
            )

            # --- Reroute Path ---
            heuristic = (
                _create_out_of_bounds_heuristic(current_request["bounds"])
                if "bounds" in current_request
                else manhattan_distance
            )
            new_path = find_shortest_path(graph, start_node, end_node, heuristic)
            if new_path:
                all_paths[req_idx] = new_path

            # --- Remove Penalties and Blockers ---
            _unblock_connection_nodes(graph, blocked_edges)
            _remove_penalties_from_graph(graph, applied_penalties)

    # --- Post-process and Finalize ---
    print("Step 5.1.4: Straightening paths...")
    all_paths = _straighten_paths(all_paths, graph, iterations=3)

    # Re-sort paths back to original order
    original_indices = [item[0] for item in indexed_requests]
    final_paths = [path for _, path in sorted(zip(original_indices, all_paths))]

    return final_paths


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
        - The graph used for pathfinding.
    """
    graph = create_graph_from_grid(points)
    path = find_shortest_path(graph, start_node, end_node)

    if path:
        for r, c in path:
            points[r][c] = path_weight  # Mark path as used in the grid

    return path, points, graph
