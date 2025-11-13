"""
This module provides grid-based pathfinding functionality using the A* algorithm
from the NetworkX library. It is designed to find the shortest path on a 2D grid
where cells can have different traversal costs (weights).
"""

from corner_minimise import find_alternative_paths, steps_to_segments, segment_to_path
from queue import Queue
from typing import List, Tuple
import networkx as nx

# --- Graph Creation ---


def create_graph_from_grid(grid: List[List[int]]) -> nx.Graph:
    """
    Converts a 2D grid into a NetworkX graph suitable for pathfinding.

    Each grid cell becomes a node, and edges are created between adjacent
    (non-diagonal) cells. The weight of an edge is the average of the
    traversal costs of the two nodes it connects.

    Vertical edges have 0.5% lower weight than horizontal edges to encourage
    north-south routing.

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

            # Add edge to the right neighbor (horizontal edge)
            if c + 1 < cols:
                right_neighbor_id = (r, c + 1)
                edge_weight = 1 + (grid[r][c] + grid[r][c + 1]) / 2
                G.add_edge(node_id, right_neighbor_id, weight=edge_weight)

            # Add edge to the bottom neighbor (vertical edge)
            if r + 1 < rows:
                bottom_neighbor_id = (r + 1, c)
                edge_weight = 1 + (grid[r][c] + grid[r + 1][c]) / 2
                # Apply 0.5% reduction to vertical edges
                edge_weight *= 0.995
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

        # Calculate the total weight of the initial path found by A*.
        # nx.path_weight sums the 'weight' attribute of all edges in the path.
        try:
            weight = nx.path_weight(graph, path, weight="weight")
        except nx.NodeNotFound:
            # This can happen if the path is empty or invalid, though astar_path should prevent this.
            return path

        # --- Path Simplification ---
        # Convert the path to segments to check for simplification opportunities.
        starting_point: tuple[int, int] = path[0]
        normalised_path: list[tuple[int, int]] = [
            (x - starting_point[0], y - starting_point[1]) for x, y in path
        ]
        segment = steps_to_segments(steps=normalised_path)

        # A path needs at least 2 corners (3 segments) to be a candidate for
        # simplification. Paths with 1 corner (2 segments) or fewer are returned.
        if len(segment) < 3:
            return path
        # find alternative paths
        alt_path_queue = Queue()
        alt_paths = find_alternative_paths(segment)
        search_space: list[list[tuple[int, int]]] = []
        for x in alt_paths:
            if x in search_space:
                continue
            alt_path_queue.put(x)
            search_space.append(x)

        best_path_corners = len(segment)
        best_path = path

        while not alt_path_queue.empty():
            alt_path = alt_path_queue.get()
            # convert back to normal path
            path_to_test: list[tuple[int, int]] = segment_to_path(
                alt_path, starting_point=starting_point
            )
            # use nx to find the weight of the path
            try:
                alt_weight = nx.path_weight(graph, path_to_test, weight="weight")
            except nx.NodeNotFound:
                # If the path is not valid in the graph, simply skip it and continue.
                continue
            if alt_weight <= weight and len(alt_path) <= len(segment):
                # This path has potential. Explore its alternatives.
                new_alternatives = find_alternative_paths(alt_path)
                for new_alt in new_alternatives:
                    if new_alt not in search_space:
                        alt_path_queue.put(new_alt)
                        search_space.append(new_alt)

            # A path is "best" if it has fewer corners than the current best
            # ... but the same or lower weight (lower is unlikely)
            if len(alt_path) < best_path_corners and alt_weight <= weight:
                print("Found better path with ", len(alt_path), " corners")
                best_path_corners = len(alt_path)
                best_path = path_to_test
        return best_path
    except nx.NetworkXNoPath:
        return []


# --- Pathfinding Helpers ---


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
    on the ownership of the path and the busbar. It includes hierarchical
    relationships to prevent crossing foreign busbars at any level.

    Args:
        busbar_edges: A dictionary of busbar edges and their owners.
        start_owner: The owner tuple (substation_name, owner_id) of the
            path's start point.
        end_owner: The owner tuple of the path's end point.
        busbar_crossing_penalty: The high penalty value for an illegal crossing.

    Returns:
        A dictionary mapping busbar edges to their calculated penalty value.
    """

    def _get_allowed_owners_for_substation(sub_name: str, path_owner: str) -> set:
        """Get all owner IDs that are allowed for a given substation and path owner."""
        allowed = {path_owner}

        # If path owner is main, allow all children
        if path_owner == "main":
            # Add all possible child owners (we'll be conservative and allow child_0 through child_9)
            for i in range(10):
                allowed.add(f"child_{i}")

        # If path owner is a child, allow main and all other children
        elif path_owner.startswith("child_"):
            allowed.add("main")
            for i in range(10):
                allowed.add(f"child_{i}")

        return allowed

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
            else:
                # Path is within the same substation
                # Check if the busbar owner is related to either start or end owner
                start_allowed_owners = _get_allowed_owners_for_substation(
                    path_sub_name, start_owner[1]
                )
                end_allowed_owners = _get_allowed_owners_for_substation(
                    path_sub_name, end_owner[1]
                )

                if (
                    bus_owner_id in start_allowed_owners
                    or bus_owner_id in end_allowed_owners
                ):
                    # Crossing a related busbar, small penalty
                    edge_penalties[edge] = 1
                else:
                    # Crossing an unrelated busbar within the same substation
                    edge_penalties[edge] = busbar_crossing_penalty

        # An inter-substation connection
        else:
            path_sub_names = {start_owner[0], end_owner[0]}

            # If it crosses a busbar of an unrelated substation
            if bus_sub_name not in path_sub_names:
                edge_penalties[edge] = busbar_crossing_penalty
            else:
                # Busbar belongs to one of the connected substations
                # Check if the busbar owner is related to the appropriate path owner
                if bus_sub_name == start_owner[0]:
                    allowed_owners = _get_allowed_owners_for_substation(
                        bus_sub_name, start_owner[1]
                    )
                elif bus_sub_name == end_owner[0]:
                    allowed_owners = _get_allowed_owners_for_substation(
                        bus_sub_name, end_owner[1]
                    )
                else:
                    allowed_owners = set()

                if bus_owner_id in allowed_owners:
                    # Crossing a related busbar, small penalty
                    edge_penalties[edge] = 1
                else:
                    # Crossing an unrelated busbar within a connected substation
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
        if other_req_idx == current_path_idx or not other_path:
            continue

        path_len = len(other_path)

        # Penalize intermediate nodes - direct slice access
        if path_len > 2:
            for node in other_path[1:-1]:
                node_usage[node] = node_usage.get(node, 0) + 1

        # Optimised edge calculation - avoid min/max calls by using conditional
        for j in range(path_len - 1):
            u, v = other_path[j], other_path[j + 1]
            # Use conditional instead of min/max for better performance
            if u < v:
                edge = (u, v)
            elif u > v:
                edge = (v, u)
            else:
                edge = (u, v)  # Same node (shouldn't happen but handle gracefully)

            edge_usage[edge] = edge_usage.get(edge, 0) + 1

    return node_usage, edge_usage


def _apply_penalties_to_graph(
    graph: nx.Graph,
    edge_usage: dict,
    node_usage: dict,
    current_penalty: float,
    start_node: tuple,
    end_node: tuple,
    # substation_pair: tuple = None,
    # all_paths: list = None,
    # current_path_idx: int = None,
    # substation_pairs: list = None,
) -> list:
    """Temporarily adds penalties to graph edges based on usage.

    Args:
        graph: The `nx.Graph` to modify.
        edge_usage: A dictionary mapping edges to their usage count.
        node_usage: A dictionary mapping nodes to their usage count.
        current_penalty: The scaled penalty value for the current iteration.
        start_node: The start node of the path being rerouted.
        end_node: The end node of the path being rerouted.
        substation_pair: The substation pair for this path (for adjacent routing).
        all_paths: All current paths (for adjacent routing calculations).
        current_path_idx: Index of current path being rerouted.
        substation_pairs: List of all substation pairs.

    Returns:
        A list of (edge, penalty_value) tuples that were applied, so they
        can be reverted later.
    """
    applied_penalties = []

    # Direct access to internal graph data structures for maximum performance
    graph_adj = graph._adj

    # Batch collect all penalty updates first
    penalty_updates = {}

    # Process edge penalties
    for edge, count in edge_usage.items():
        u, v = edge
        if u in graph_adj and v in graph_adj[u]:
            penalty = (count**2) * current_penalty
            penalty_updates[edge] = penalty_updates.get(edge, 0) + penalty

    # Process node penalties
    excluded_nodes = {start_node, end_node}
    for node, count in node_usage.items():
        if node not in excluded_nodes and node in graph_adj:
            penalty = (count**2) * current_penalty
            # Direct access to neighbors via adjacency dict
            for neighbor in graph_adj[node]:
                # Use conditional instead of min/max for better performance
                if node < neighbor:
                    edge = (node, neighbor)
                elif node > neighbor:
                    edge = (neighbor, node)
                else:
                    edge = (node, neighbor)  # Same node (shouldn't happen)
                penalty_updates[edge] = penalty_updates.get(edge, 0) + penalty

    # Apply all penalties in one batch
    for edge, total_penalty in penalty_updates.items():
        u, v = edge
        if u in graph_adj and v in graph_adj[u]:
            graph_adj[u][v]["weight"] += total_penalty
            # NetworkX graphs are undirected, so update both directions
            if v in graph_adj and u in graph_adj[v]:
                graph_adj[v][u]["weight"] += total_penalty
            applied_penalties.append((edge, total_penalty))

    # Simplified adjacent routing (skip for now to focus on core performance)
    # The adjacency bonus was causing additional expensive graph operations

    return applied_penalties


def _remove_penalties_from_graph(graph: nx.Graph, applied_penalties: list):
    """Removes temporary penalties from graph edges.

    Args:
        graph: The `nx.Graph` to modify.
        applied_penalties: A list of (edge, penalty_value) tuples to revert.
    """
    # Direct access to internal graph data structures for maximum performance
    graph_adj = graph._adj

    for edge, penalty in applied_penalties:
        u, v = edge
        if u in graph_adj and v in graph_adj[u]:
            graph_adj[u][v]["weight"] -= penalty
            # NetworkX graphs are undirected, so update both directions
            if v in graph_adj and u in graph_adj[v]:
                graph_adj[v][u]["weight"] -= penalty


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
    graph_adj = graph._adj

    # Batch collect and apply blocking in one pass
    for node in nodes_to_block:
        if node in graph_adj:
            # Direct access to neighbors via adjacency dict
            for neighbor in graph_adj[node]:
                # Use conditional instead of min/max for better performance
                if node < neighbor:
                    edge_tuple = (node, neighbor)
                elif node > neighbor:
                    edge_tuple = (neighbor, node)
                else:
                    edge_tuple = (node, neighbor)  # Same node (shouldn't happen)

                original_weight = graph_adj[node][neighbor]["weight"]
                blocked_edges.append((edge_tuple, original_weight))
                graph_adj[node][neighbor]["weight"] = float("inf")
                # Update both directions for undirected graph
                if neighbor in graph_adj and node in graph_adj[neighbor]:
                    graph_adj[neighbor][node]["weight"] = float("inf")

    return blocked_edges


def _unblock_connection_nodes(graph: nx.Graph, blocked_edges: list):
    """Restores access to previously blocked connection nodes.

    Args:
        graph: The `nx.Graph` to modify.
        blocked_edges: A list of (edge, original_weight) tuples to restore.
    """
    graph_adj = graph._adj
    for edge, original_weight in blocked_edges:
        u, v = edge
        if u in graph_adj and v in graph_adj[u]:
            graph_adj[u][v]["weight"] = original_weight
            # Update both directions for undirected graph
            if v in graph_adj and u in graph_adj[v]:
                graph_adj[v][u]["weight"] = original_weight


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


def run_all_gridsearches(
    path_requests: List[dict],
    points: List[List[int]],
    grid_owners: List[List[str]],
    congestion_penalty_increment: float = 2.0,
    all_connection_nodes: set = None,
    busbar_weight: int = None,
    busbar_crossing_penalty: int = 100000,
    substation_pairs: List[tuple] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Finds paths for a series of requests, aiming to minimize total congestion.

    This is an enhanced version of a sequential rip-up and reroute algorithm.
    Key features:
    1.  **Longest Path First**: It prioritizes routing longer paths first, as
        they are typically harder to place.
    2.  **Adjacent Routing**: Connections between the same substation pairs
        are encouraged to route adjacently for cleaner layouts.

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
        substation_pairs: A list of substation pair tuples for adjacent routing.

    Returns:
        A list of paths, where each path is a list of coordinates, in the
        same order as the input path_requests.
    """
    print("Step 5.1.1: Creating base pathfinding graph...")
    graph = create_graph_from_grid(points)

    # --- Group requests by substation pairs for adjacent routing ---
    pair_groups = {}
    for i, pair in enumerate(substation_pairs):
        if pair not in pair_groups:
            pair_groups[pair] = []
        pair_groups[pair].append(i)

    # Sort groups by the shortest path in each group, then sort within groups
    sorted_group_indices = []
    for pair, indices in pair_groups.items():
        # Sort indices within this group by path length
        group_requests = [(i, path_requests[i]) for i in indices]
        group_requests.sort(
            key=lambda x: manhattan_distance(x[1]["start"], x[1]["end"]),
            reverse=True,
        )

        # Use shortest path in group for overall group sorting
        min_distance = min(
            manhattan_distance(path_requests[i]["start"], path_requests[i]["end"])
            for i in indices
        )
        sorted_group_indices.append((min_distance, [x[0] for x in group_requests]))

    # Sort groups by their minimum distance
    sorted_group_indices.sort(key=lambda x: x[0], reverse=True)

    # Flatten to get final order
    indexed_requests = []
    for _, group_indices in sorted_group_indices:
        for idx in group_indices:
            indexed_requests.append((idx, path_requests[idx]))

    print("Step 5.1.2: Performing routing...")
    busbar_edges = _get_busbar_edges(graph, points, grid_owners, busbar_weight)

    all_paths = []

    for req_idx, (_, current_request) in enumerate(indexed_requests):
        start_node = current_request["start"]
        end_node = current_request["end"]

        # --- Calculate Penalties ---
        busbar_penalties = _calculate_busbar_crossing_penalties(
            busbar_edges,
            current_request["start_owner"],
            current_request["end_owner"],
            busbar_crossing_penalty,
        )
        node_usage, congestion_usage = _calculate_congestion_usage(all_paths, req_idx)
        edge_usage = {**congestion_usage}
        for edge, penalty in busbar_penalties.items():
            edge_usage[edge] = edge_usage.get(edge, 0) + penalty

        # --- Apply Penalties and Blockers ---
        applied_penalties = _apply_penalties_to_graph(
            graph,
            edge_usage,
            node_usage,
            congestion_penalty_increment,
            start_node,
            end_node,
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
        print(f"Path finding {start_node} -> {end_node} with id {req_idx}")
        new_path = find_shortest_path(graph, start_node, end_node, heuristic)
        if new_path:
            all_paths.append(new_path)

        # --- Remove Penalties and Blockers ---
        _unblock_connection_nodes(graph, blocked_edges)
        _remove_penalties_from_graph(graph, applied_penalties)

    # Re-sort paths back to original order
    original_indices = [item[0] for item in indexed_requests]
    final_paths = [path for _, path in sorted(zip(original_indices, all_paths))]

    return final_paths
