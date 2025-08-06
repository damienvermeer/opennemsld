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
        start_time = time.time()
        path = nx.astar_path(
            graph, start_node, end_node, weight="weight", heuristic=heuristic
        )
        end_time = time.time()
        # print(f"Path found in: {end_time - start_time:.4f} seconds.")
        return path
    except nx.NetworkXNoPath:
        # print(f"No path could be found from {start_node} to {end_node}.")
        return []


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
        print(f"  Straightening iteration {iter_num + 1}/{iterations}...")
        paths_changed_in_iteration = False

        # Sort paths by length (longest first) for processing, keeping original index
        indexed_paths_to_process = sorted(
            enumerate(paths_to_modify), key=lambda x: len(x[1]), reverse=True
        )

        # Build the set of all occupied nodes and edges once per iteration
        all_occupied_nodes = set()
        all_occupied_edges = set()
        for p in paths_to_modify:
            if not p:
                continue
            for node in p:
                all_occupied_nodes.add(node)
            for i in range(len(p) - 1):
                edge = tuple(sorted((p[i], p[i + 1])))
                all_occupied_edges.add(edge)

        for original_idx, _ in indexed_paths_to_process:
            path = paths_to_modify[original_idx]
            if not path or len(path) < 2:
                continue

            # Loop until no more changes can be made to this path
            while True:
                made_change_in_pass = False

                # Create occupied sets for *other* paths for collision detection.
                current_path_nodes = set(path)
                current_path_edges = set()
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    current_path_edges.add(edge)

                other_occupied_nodes = all_occupied_nodes - current_path_nodes
                other_occupied_edges = all_occupied_edges - current_path_edges

                for i in range(1, len(path) - 1):
                    p_prev, p_curr, p_next = (
                        path[i - 1],
                        path[i],
                        path[i + 1],
                    )

                    # Check for a corner (not collinear).
                    if (
                        p_prev[0] == p_curr[0] == p_next[0]
                        or p_prev[1] == p_curr[1] == p_next[1]
                    ):
                        continue

                    # This is a corner. Find the alternative node for a rectangular detour.
                    p_alt = (
                        p_prev[0] + p_next[0] - p_curr[0],
                        p_prev[1] + p_next[1] - p_curr[1],
                    )

                    if not graph.has_node(p_alt):
                        continue

                    # Check for collisions with other paths or self-intersection.
                    if p_alt in other_occupied_nodes or p_alt in path:
                        continue

                    edge1_alt = tuple(sorted((p_prev, p_alt)))
                    edge2_alt = tuple(sorted((p_alt, p_next)))

                    if not graph.has_edge(*edge1_alt) or not graph.has_edge(*edge2_alt):
                        continue

                    if (
                        edge1_alt in other_occupied_edges
                        or edge2_alt in other_occupied_edges
                    ):
                        continue

                    # Check if weights are equal.
                    edge1_orig = tuple(sorted((p_prev, p_curr)))
                    edge2_orig = tuple(sorted((p_curr, p_next)))

                    weight_orig = (
                        graph.edges[edge1_orig]["weight"]
                        + graph.edges[edge2_orig]["weight"]
                    )
                    weight_alt = (
                        graph.edges[edge1_alt]["weight"]
                        + graph.edges[edge2_alt]["weight"]
                    )

                    # Tie-break by preferring the lexicographically smaller node to prevent flipping.
                    if abs(weight_orig - weight_alt) < 1e-9 and p_alt < p_curr:
                        print(
                            f"    Path {original_idx}: Straightened corner at {p_curr} to {p_alt}"
                        )
                        old_node = path[i]
                        path[i] = p_alt
                        made_change_in_pass = True
                        paths_changed_in_iteration = True

                        # Update the global occupied sets for subsequent paths
                        all_occupied_nodes.remove(old_node)
                        all_occupied_nodes.add(p_alt)
                        all_occupied_edges.remove(edge1_orig)
                        all_occupied_edges.remove(edge2_orig)
                        all_occupied_edges.add(edge1_alt)
                        all_occupied_edges.add(edge2_alt)

                        # Restart scan on the now-modified path
                        break

                if not made_change_in_pass:
                    # No changes in a full pass, so this path is done.
                    break

        if not paths_changed_in_iteration:
            print("  No more paths changed, finishing straightening.")
            break

    return paths_to_modify


# --- Main Orchestration ---


def run_all_gridsearches(
    path_requests: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    points: List[List[int]],
    iterations: int = 5,
    congestion_penalty_increment: float = 2.0,
    all_connection_nodes: set = None,
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
        iterations: The number of times to iterate the pathfinding process.
        congestion_penalty_increment: The base penalty added to a graph edge
                                      for each path crossing it. This value
                                      is scaled up with each iteration.

    Returns:
        A list of paths, where each path is a list of coordinates, in the
        same order as the input path_requests.
    """
    # Create the graph once from the base grid.
    print("Pathfinding: Creating base graph...")
    graph = create_graph_from_grid(points)

    # --- Sort requests to route longest paths first ---
    indexed_requests = [
        (i, req, manhattan_distance(req[0], req[1]))
        for i, req in enumerate(path_requests)
    ]
    indexed_requests.sort(key=lambda x: x[2], reverse=True)
    sorted_requests = [req for i, req, _ in indexed_requests]

    # Initial routing of all paths on the base graph, in sorted order.
    print("Pathfinding: Starting initial routing (longest paths first)...")
    all_paths = [
        find_shortest_path(graph, start, end) for start, end in sorted_requests
    ]
    print("Pathfinding: Initial routing complete.")

    # Iteratively refine paths by re-routing each one on a graph
    # that is penalized by the existence of all other paths.
    for i in range(iterations):
        print(f"Pathfinding: Starting iteration {i + 1}/{iterations}...")
        # The penalty is scaled quadratically. It starts low to allow for more
        # "chaotic" pathfinding and increases sharply in later iterations to
        # force convergence to a low-congestion state.
        if iterations > 1:
            # Use a quadratic scaling factor from 0 to 1.
            scaling_factor = (i / (iterations - 1)) ** 2
            # The max penalty multiplier is set to be aggressive in the final iterations.
            max_penalty_multiplier = 1.0
            current_penalty = (
                congestion_penalty_increment * max_penalty_multiplier * scaling_factor
            )
        else:
            # For a single iteration, use the base penalty.
            current_penalty = congestion_penalty_increment

        for req_idx in range(len(sorted_requests)):
            start_node, end_node = sorted_requests[req_idx]

            # Calculate edge and node usage from all *other* paths.
            edge_usage = {}
            node_usage = {}
            for other_req_idx, other_path in enumerate(all_paths):
                if req_idx == other_req_idx:
                    continue
                if not other_path:
                    continue

                # Penalize intermediate nodes to discourage paths from crossing.
                # We don't penalize the endpoints of paths.
                for node in other_path[1:-1]:
                    node_usage[node] = node_usage.get(node, 0) + 1

                for j in range(len(other_path) - 1):
                    u, v = other_path[j], other_path[j + 1]
                    # Normalize edge to be order-independent for the undirected graph.
                    edge = tuple(sorted((u, v)))
                    edge_usage[edge] = edge_usage.get(edge, 0) + 1

            # Temporarily apply penalties to the graph for shared edges.
            penalized_edges = []
            for edge, count in edge_usage.items():
                penalty = (count**2) * current_penalty
                if graph.has_edge(*edge):
                    graph.edges[edge]["weight"] += penalty
                    penalized_edges.append((edge, penalty))

            # Temporarily apply penalties for shared nodes by penalizing their edges.
            penalized_node_edges = []
            for node, count in node_usage.items():
                # Don't penalize the start/end nodes of the path we are currently routing.
                if node in (start_node, end_node):
                    continue

                if graph.has_node(node):
                    penalty = (count**2) * current_penalty
                    for neighbor in graph.neighbors(node):
                        edge = tuple(sorted((node, neighbor)))
                        if graph.has_edge(*edge):
                            graph.edges[edge]["weight"] += penalty
                            penalized_node_edges.append((edge, penalty))

            # --- Temporarily block other connection points ---
            blocked_edges = []
            if all_connection_nodes:
                nodes_to_block = all_connection_nodes - {start_node, end_node}
                for node in nodes_to_block:
                    if graph.has_node(node):
                        # Block all edges connected to this node
                        for neighbor in list(graph.neighbors(node)):
                            edge_tuple = tuple(sorted((node, neighbor)))
                            if graph.has_edge(*edge_tuple):
                                original_weight = graph.edges[edge_tuple]["weight"]
                                blocked_edges.append((edge_tuple, original_weight))
                                graph.edges[edge_tuple]["weight"] = float("inf")

            # Reroute the current path on the penalized graph.
            new_path = find_shortest_path(graph, start_node, end_node)
            if new_path:  # Only update if a path was found
                all_paths[req_idx] = new_path

            # --- Unblock connection points before removing penalties ---
            for edge, original_weight in blocked_edges:
                if graph.has_edge(*edge):
                    graph.edges[edge]["weight"] = original_weight

            # Remove the temporary penalties to prepare for the next path.
            for edge, penalty in penalized_node_edges:
                if graph.has_edge(*edge):
                    graph.edges[edge]["weight"] -= penalty
            for edge, penalty in penalized_edges:
                if graph.has_edge(*edge):
                    graph.edges[edge]["weight"] -= penalty

        print(f"Pathfinding: Iteration {i + 1} complete.")

    # --- Post-process to straighten paths ---
    print("Pathfinding: Post-processing to straighten paths...")
    all_paths = _straighten_paths(all_paths, graph, iterations=3)
    print("Pathfinding: Straightening complete.")

    # --- Re-sort paths back to original order ---
    # Associate original indices with the final paths
    indexed_paths = list(zip([item[0] for item in indexed_requests], all_paths))
    # Sort by original index
    indexed_paths.sort(key=lambda x: x[0])
    # Extract paths in original order
    final_paths = [path for i, path in indexed_paths]

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

    # --- 2. Create Graph ---
    graph = create_graph_from_grid(points)

    # --- 3. Find Path ---
    path = find_shortest_path(graph, start_node, end_node)

    # --- 4. Update Grid and Return ---
    if path:
        # print(f"Path length: {len(path) - 1} steps.")
        # print("Updating 'points' grid with the new path...")
        for r, c in path:
            points[r][c] = path_weight  # Mark path as used in the grid

    return path, points, graph
