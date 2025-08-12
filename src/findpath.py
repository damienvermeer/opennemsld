"""
This module provides grid-based pathfinding functionality using the A* algorithm
from the NetworkX library. It is designed to find the shortest path on a 2D grid
where cells can have different traversal costs (weights).
"""

from typing import List, Literal, Tuple, Union

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
                start_allowed_owners = _get_allowed_owners_for_substation(path_sub_name, start_owner[1])
                end_allowed_owners = _get_allowed_owners_for_substation(path_sub_name, end_owner[1])
                
                if bus_owner_id in start_allowed_owners or bus_owner_id in end_allowed_owners:
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
                    allowed_owners = _get_allowed_owners_for_substation(bus_sub_name, start_owner[1])
                elif bus_sub_name == end_owner[0]:
                    allowed_owners = _get_allowed_owners_for_substation(bus_sub_name, end_owner[1])
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
    substation_pair: tuple = None,
    all_paths: list = None,
    current_path_idx: int = None,
    substation_pairs: list = None,
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
    
    # Apply standard congestion penalties
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
    
    # Apply adjacent routing incentives for same substation pairs
    if (substation_pair and all_paths and current_path_idx is not None and 
        substation_pairs and len(all_paths) == len(substation_pairs)):
        
        # Find other paths from the same substation pair
        same_pair_paths = []
        for i, other_pair in enumerate(substation_pairs):
            if i != current_path_idx and other_pair == substation_pair and all_paths[i]:
                same_pair_paths.append(all_paths[i])
        
        # Apply adjacency bonus (negative penalty) to edges near same-pair paths
        adjacency_bonus = current_penalty * 0.3  # 30% bonus for being adjacent
        
        for same_pair_path in same_pair_paths:
            for path_node in same_pair_path:
                # Apply bonus to edges adjacent to this path
                if graph.has_node(path_node):
                    for neighbor in graph.neighbors(path_node):
                        edge = tuple(sorted((path_node, neighbor)))
                        if graph.has_edge(*edge):
                            # Apply negative penalty (bonus) to encourage adjacency
                            bonus = -adjacency_bonus
                            graph.edges[edge]["weight"] += bonus
                            applied_penalties.append((edge, bonus))
    
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

    def _get_straight_run_length(
            path: List,
            i: int,
            direction: Literal['forward', 'backward'] = 'forward'
    ) -> Tuple[Union[Literal['hor', 'vert'], None], int]:
        """
        Looks for and returns the number of colinear points on the path from index i+/-1
        and onwards/backwards, until the path turns.

        Args:
            path: The path within which we are looking for straight runs
            i: The index of the "L" in the path at which to begin the search
            direction: foward means we look towards increasing indices, backwards means decreasing

        Returns:
            The type of straight run ('vert' or 'hor') or None if only one point in the straight run
            The number of colinear points on the path from index i+/-1 on
        """

        count = 1
        s = 1 if direction == 'forward' else -1

        # If we are looking at the final node in the path, return immediately
        if not i+s*2 in range(len(path)):
            return None, 1

        # figure out if we're looking for a vertical or horizontal run
        search_type = 'vert' if path[i+s][0] == path[i+s*2][0] else 'hor'

        while i+s*(count+1) in range(len(path)):
            p1, p2 = path[i+s*count], path[i+s*(count+1)]
            if search_type == 'vert' and p1[0] == p2[0] or search_type == 'hor' and p1[1] == p2[1]:
                count += 1
            else:
                break

        return search_type, count

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

    # If weights are (relatively) similar, prefer a flip which lengthens the straight run within the path
    if abs(weight_orig - weight_alt) < 100:
        # get straight run lengths forward and backward
        forward_run = _get_straight_run_length(path, i, 'forward')
        backward_run = _get_straight_run_length(path, i, 'backward')

        # determine if max run is forwards or backwards in the path
        max_run = max((forward_run, backward_run), key=lambda run: run[1])
        p_check = p_next if forward_run[1] >= backward_run[1] else p_prev

        # choose the alternative if doing so will lengthen our straight run at this point
        if (max_run[0] == 'vert' and p_alt[0] == p_check[0]
            or max_run[0] == 'hor' and p_alt[1] == p_check[1]):
            return p_alt

    return None


def _smooth_long_sections(
    path: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Smooth long sections by identifying and consolidating parallel segments.

    This function looks for patterns like:
    horizontal -> vertical -> horizontal -> vertical
    and tries to convert them to:
    horizontal -> vertical

    Args:
        path: The current path
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Smoothed path
    """
    if len(path) < 4:
        return path

    smoothed = [path[0]]
    i = 0

    while i < len(path) - 1:
        current = path[i]

        # Look ahead to find long straight sections
        horizontal_segments = []
        vertical_segments = []
        j = i

        # Analyze the next several segments to find patterns
        while j < len(path) - 1:
            curr_node = path[j]
            next_node = path[j + 1]

            # Determine if this segment is horizontal or vertical
            if curr_node[0] == next_node[0]:  # Same row = horizontal
                horizontal_segments.append((j, curr_node, next_node))
            elif curr_node[1] == next_node[1]:  # Same column = vertical
                vertical_segments.append((j, curr_node, next_node))
            else:
                # Diagonal or end of pattern
                break

            j += 1

            # If we've found a pattern worth optimizing
            if len(horizontal_segments) >= 2 and len(vertical_segments) >= 1:
                # Try to create a more direct path
                optimized = _try_optimize_segment_pattern(
                    path, i, j, graph, other_occupied_nodes, other_occupied_edges
                )
                if optimized:
                    smoothed.extend(
                        optimized[1:]
                    )  # Skip the first node (already added)
                    i = j
                    break
            elif len(vertical_segments) >= 2 and len(horizontal_segments) >= 1:
                # Try to create a more direct path
                optimized = _try_optimize_segment_pattern(
                    path, i, j, graph, other_occupied_nodes, other_occupied_edges
                )
                if optimized:
                    smoothed.extend(
                        optimized[1:]
                    )  # Skip the first node (already added)
                    i = j
                    break

        # If no optimization was possible, just add the next node
        if j == i + 1 or i >= len(path) - 1:
            if i + 1 < len(path):
                smoothed.append(path[i + 1])
            i += 1
        # If we optimized, i was already updated in the loop

    return smoothed


def _try_optimize_segment_pattern(
    path: list,
    start_idx: int,
    end_idx: int,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Try to optimize a pattern of segments by creating a more direct route.

    Args:
        path: The current path
        start_idx: Start index of the pattern
        end_idx: End index of the pattern
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Optimized segment if successful, None otherwise
    """
    if end_idx - start_idx < 3:
        return None

    start_node = path[start_idx]
    end_node = path[end_idx]

    # Try a simple L-shaped path: horizontal then vertical, or vertical then horizontal
    intermediate_h = (start_node[0], end_node[1])  # Horizontal first
    intermediate_v = (end_node[0], start_node[1])  # Vertical first

    # Try horizontal-then-vertical path
    if graph.has_edge(start_node, intermediate_h) and graph.has_edge(
        intermediate_h, end_node
    ):
        edge1 = tuple(sorted([start_node, intermediate_h]))
        edge2 = tuple(sorted([intermediate_h, end_node]))

        if (
            edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
            and intermediate_h not in other_occupied_nodes
        ):
            return [start_node, intermediate_h, end_node]

    # Try vertical-then-horizontal path
    if graph.has_edge(start_node, intermediate_v) and graph.has_edge(
        intermediate_v, end_node
    ):
        edge1 = tuple(sorted([start_node, intermediate_v]))
        edge2 = tuple(sorted([intermediate_v, end_node]))

        if (
            edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
            and intermediate_v not in other_occupied_nodes
        ):
            return [start_node, intermediate_v, end_node]

    return None


def _try_multi_segment_straightening(
    path: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Try to straighten multiple segments at once by looking for longer patterns.

    This function identifies sequences like:
    A -> B -> C -> D -> E
    where A to E could be connected more directly.

    Args:
        path: The current path
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Path with multi-segment straightening applied
    """
    if len(path) < 4:
        return path

    straightened = [path[0]]
    i = 0

    while i < len(path) - 1:
        current = path[i]
        best_jump = None
        best_jump_length = 0

        # Look ahead to find the furthest node we can reach directly
        for j in range(i + 2, min(i + 8, len(path))):  # Look up to 7 nodes ahead
            target = path[j]

            # Check if we can create a simple path from current to target
            simple_path = _find_simple_path(
                current, target, graph, other_occupied_nodes, other_occupied_edges
            )

            if simple_path and len(simple_path) < (j - i + 1):
                # This is a valid improvement
                jump_length = j - i
                if jump_length > best_jump_length:
                    best_jump = simple_path
                    best_jump_length = jump_length

        if best_jump:
            # Apply the best jump we found
            straightened.extend(best_jump[1:])  # Skip the first node (already added)
            i += best_jump_length
        else:
            # No jump possible, add the next node
            if i + 1 < len(path):
                straightened.append(path[i + 1])
            i += 1

    return straightened


def _find_simple_path(
    start: tuple,
    end: tuple,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Find a simple (L-shaped or straight) path between two nodes.

    Args:
        start: Starting node
        end: Ending node
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Simple path if found, None otherwise
    """
    # Try direct connection first
    if graph.has_edge(start, end):
        edge = tuple(sorted([start, end]))
        if edge not in other_occupied_edges:
            return [start, end]

    # Try L-shaped paths
    # Horizontal then vertical
    intermediate_h = (start[0], end[1])
    if (
        intermediate_h != start
        and intermediate_h != end
        and graph.has_edge(start, intermediate_h)
        and graph.has_edge(intermediate_h, end)
    ):
        edge1 = tuple(sorted([start, intermediate_h]))
        edge2 = tuple(sorted([intermediate_h, end]))

        if (
            edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
            and intermediate_h not in other_occupied_nodes
        ):
            return [start, intermediate_h, end]

    # Vertical then horizontal
    intermediate_v = (end[0], start[1])
    if (
        intermediate_v != start
        and intermediate_v != end
        and graph.has_edge(start, intermediate_v)
        and graph.has_edge(intermediate_v, end)
    ):
        edge1 = tuple(sorted([start, intermediate_v]))
        edge2 = tuple(sorted([intermediate_v, end]))

        if (
            edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
            and intermediate_v not in other_occupied_nodes
        ):
            return [start, intermediate_v, end]

    return None


def _analyze_path_structure(path: list) -> dict:
    """
    Analyze path structure by reading edges directly to identify straight sections and corners.

    This creates a comprehensive analysis of the path structure including:
    - Straight segments (consecutive edges in same direction)
    - Corner points (where direction changes)
    - Segment types and lengths

    Args:
        path: The path to analyze

    Returns:
        Dictionary containing:
        - 'segments': List of (start_idx, end_idx, direction, length) tuples
        - 'corners': List of corner node indices
        - 'straight_runs': List of long straight segments suitable for optimization
        - 'actual_corners': List of actual corner positions in path
    """
    if len(path) < 2:
        return {
            "segments": [],
            "corners": [],
            "straight_runs": [],
            "actual_corners": [],
        }

    segments = []
    corners = []
    actual_corners = []
    current_direction = None
    segment_start = 0

    for i in range(len(path) - 1):
        curr_node = path[i]
        next_node = path[i + 1]

        # Determine edge direction by examining coordinate differences
        row_diff = next_node[0] - curr_node[0]
        col_diff = next_node[1] - curr_node[1]

        if row_diff == 0 and col_diff != 0:
            direction = "horizontal"
        elif col_diff == 0 and row_diff != 0:
            direction = "vertical"
        else:
            direction = "diagonal"  # Shouldn't happen in grid pathfinding

        # Check for direction change
        if current_direction is not None and direction != current_direction:
            # End current segment
            segment_length = i - segment_start
            segments.append((segment_start, i, current_direction, segment_length))
            corners.append(i)  # Mark the corner point
            actual_corners.append(curr_node)  # Store the actual corner position
            segment_start = i

        current_direction = direction

    # Add final segment
    if segment_start < len(path) - 1:
        segment_length = len(path) - 1 - segment_start
        segments.append(
            (segment_start, len(path) - 1, current_direction, segment_length)
        )

    # Identify long straight runs suitable for optimization
    straight_runs = [
        (start, end, direction)
        for start, end, direction, length in segments
        if length >= 3  # Lowered threshold to catch more opportunities
    ]

    # Also identify potential corner elimination opportunities
    corner_opportunities = []
    for i in range(1, len(path) - 1):
        prev_node = path[i - 1]
        curr_node = path[i]
        next_node = path[i + 1]

        # Check if this forms a corner (not collinear)
        if not (
            (prev_node[0] == curr_node[0] == next_node[0])
            or (prev_node[1] == curr_node[1] == next_node[1])
        ):
            corner_opportunities.append(i)

    return {
        "segments": segments,
        "corners": corners,
        "straight_runs": straight_runs,
        "actual_corners": actual_corners,
        "corner_opportunities": corner_opportunities,
    }


def _square_out_corners(
    path: list,
    path_structure: dict,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Apply aggressive staircase elimination and corner squaring.

    This function specifically targets staircase patterns like:
    horizontal -> vertical -> horizontal -> vertical
    and converts them to simple L-shapes: horizontal -> vertical OR vertical -> horizontal

    Args:
        path: The current path
        path_structure: Analysis from _analyze_path_structure
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Path with staircases eliminated and corners squared out
    """
    if len(path) < 4:
        return path

    segments = path_structure["segments"]

    if len(segments) < 2:
        return path

    squared_path = path[:]

    # Phase 1: Aggressive staircase elimination
    staircase_eliminated = _eliminate_staircases(
        squared_path, segments, graph, other_occupied_nodes, other_occupied_edges
    )

    if len(staircase_eliminated) < len(squared_path):
        squared_path = staircase_eliminated

    # Phase 2: Traditional corner squaring for remaining issues
    changes_made = True
    iteration_count = 0
    max_iterations = 5

    while changes_made and iteration_count < max_iterations:
        changes_made = False
        iteration_count += 1

        # Look for simple 3-4 node patterns that can be simplified
        for i in range(len(squared_path) - 3):
            node_a = squared_path[i]
            node_b = squared_path[i + 1]
            node_c = squared_path[i + 2]

            # Try direct connection A -> C (skip B)
            if _can_connect_directly(
                node_a, node_c, graph, other_occupied_nodes, other_occupied_edges
            ):
                squared_path = squared_path[: i + 1] + [node_c] + squared_path[i + 3 :]
                changes_made = True
                break

        if changes_made:
            continue

        # Look for longer patterns
        for i in range(len(squared_path) - 4):
            start_node = squared_path[i]
            end_idx = min(i + 8, len(squared_path) - 1)

            for j in range(i + 4, end_idx + 1):
                end_node = squared_path[j]

                simple_path = _find_simple_l_path(
                    start_node,
                    end_node,
                    graph,
                    other_occupied_nodes,
                    other_occupied_edges,
                )

                if simple_path and len(simple_path) < (j - i + 1):
                    squared_path = (
                        squared_path[:i] + simple_path + squared_path[j + 1 :]
                    )
                    changes_made = True
                    break

            if changes_made:
                break

    return squared_path


def _eliminate_staircases(
    path: list,
    segments: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Aggressively eliminate staircase patterns by converting them to simple L-shapes.

    Identifies patterns like: H-V-H-V-H or V-H-V-H-V and converts them to H-V or V-H.

    Args:
        path: The current path
        segments: List of (start_idx, end_idx, direction, length) tuples
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Path with staircases eliminated
    """
    if len(segments) < 3:
        return path

    # Look for staircase patterns: alternating horizontal/vertical segments
    staircase_groups = []
    current_group = []

    for i, (start_idx, end_idx, direction, length) in enumerate(segments):
        if not current_group:
            current_group = [i]
        else:
            # Check if this continues the staircase pattern
            prev_seg = segments[current_group[-1]]
            if prev_seg[2] != direction:  # Different direction = continues staircase
                current_group.append(i)
            else:
                # Same direction = end of staircase
                if (
                    len(current_group) >= 3
                ):  # At least 3 segments = staircase worth eliminating
                    staircase_groups.append(current_group)
                current_group = [i]

    # Check final group
    if len(current_group) >= 3:
        staircase_groups.append(current_group)

    if not staircase_groups:
        return path

    # Process staircase groups from end to start to maintain indices
    result_path = path[:]

    for group in reversed(staircase_groups):
        first_seg_idx = group[0]
        last_seg_idx = group[-1]

        first_seg = segments[first_seg_idx]
        last_seg = segments[last_seg_idx]

        # Get the start and end nodes of the entire staircase
        staircase_start_node = result_path[first_seg[0]]
        staircase_end_node = result_path[last_seg[1]]

        # Calculate total displacement
        total_row_diff = staircase_end_node[0] - staircase_start_node[0]
        total_col_diff = staircase_end_node[1] - staircase_start_node[1]

        # Try both L-shaped replacements
        replacement_path = None

        # Option 1: Horizontal first, then vertical
        if total_col_diff != 0 and total_row_diff != 0:
            corner_h = (staircase_start_node[0], staircase_end_node[1])
            h_then_v_path = _try_l_shaped_replacement(
                staircase_start_node,
                corner_h,
                staircase_end_node,
                graph,
                other_occupied_nodes,
                other_occupied_edges,
            )
            if h_then_v_path:
                replacement_path = h_then_v_path

        # Option 2: Vertical first, then horizontal (if option 1 didn't work)
        if not replacement_path and total_row_diff != 0 and total_col_diff != 0:
            corner_v = (staircase_end_node[0], staircase_start_node[1])
            v_then_h_path = _try_l_shaped_replacement(
                staircase_start_node,
                corner_v,
                staircase_end_node,
                graph,
                other_occupied_nodes,
                other_occupied_edges,
            )
            if v_then_h_path:
                replacement_path = v_then_h_path

        # Option 3: Direct connection (if it's just horizontal or vertical)
        if not replacement_path:
            if (
                total_row_diff == 0 or total_col_diff == 0
            ):  # Pure horizontal or vertical
                if _can_connect_directly(
                    staircase_start_node,
                    staircase_end_node,
                    graph,
                    other_occupied_nodes,
                    other_occupied_edges,
                ):
                    replacement_path = [staircase_start_node, staircase_end_node]

        # Apply the replacement if we found one
        if replacement_path and len(replacement_path) < (
            last_seg[1] - first_seg[0] + 1
        ):
            result_path = (
                result_path[: first_seg[0]]
                + replacement_path
                + result_path[last_seg[1] + 1 :]
            )

            # Recalculate segments for remaining processing
            new_structure = _analyze_path_structure(result_path)
            segments = new_structure["segments"]

    return result_path


def _try_l_shaped_replacement(
    start_node: tuple,
    corner_node: tuple,
    end_node: tuple,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Try to create an L-shaped path through a specific corner node.

    Args:
        start_node: Starting node
        corner_node: Intermediate corner node
        end_node: Ending node
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        L-shaped path if successful, None otherwise
    """
    if (
        corner_node == start_node
        or corner_node == end_node
        or corner_node in other_occupied_nodes
        or not graph.has_node(corner_node)
    ):
        return None

    # Check if we can create the path: start -> corner -> end
    edge1 = tuple(sorted([start_node, corner_node]))
    edge2 = tuple(sorted([corner_node, end_node]))

    if (
        graph.has_edge(*edge1)
        and graph.has_edge(*edge2)
        and edge1 not in other_occupied_edges
        and edge2 not in other_occupied_edges
    ):
        return [start_node, corner_node, end_node]

    return None


def _can_connect_directly(
    node1: tuple,
    node2: tuple,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> bool:
    """Check if two nodes can be connected directly without conflicts."""
    if not graph.has_edge(node1, node2):
        return False

    edge = tuple(sorted([node1, node2]))
    return edge not in other_occupied_edges


def _find_simple_l_path(
    start: tuple,
    end: tuple,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Find a simple L-shaped path between two nodes.

    Args:
        start: Starting node
        end: Ending node
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Simple L-shaped path if found, None otherwise
    """
    # Try horizontal-then-vertical
    corner_h = (start[0], end[1])
    if (
        corner_h != start
        and corner_h != end
        and corner_h not in other_occupied_nodes
        and graph.has_node(corner_h)
    ):
        edge1 = tuple(sorted([start, corner_h]))
        edge2 = tuple(sorted([corner_h, end]))

        if (
            graph.has_edge(*edge1)
            and graph.has_edge(*edge2)
            and edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
        ):
            return [start, corner_h, end]

    # Try vertical-then-horizontal
    corner_v = (end[0], start[1])
    if (
        corner_v != start
        and corner_v != end
        and corner_v not in other_occupied_nodes
        and graph.has_node(corner_v)
    ):
        edge1 = tuple(sorted([start, corner_v]))
        edge2 = tuple(sorted([corner_v, end]))

        if (
            graph.has_edge(*edge1)
            and graph.has_edge(*edge2)
            and edge1 not in other_occupied_edges
            and edge2 not in other_occupied_edges
        ):
            return [start, corner_v, end]

    return None


def _try_square_corner_pair(
    path: list,
    seg1_start: int,
    seg1_end: int,
    seg2_start: int,
    seg2_end: int,
    seg1_dir: str,
    seg2_dir: str,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Try to square out a corner between two perpendicular segments.

    Args:
        path: Current path
        seg1_start, seg1_end: Indices of first segment
        seg2_start, seg2_end: Indices of second segment
        seg1_dir, seg2_dir: Directions of the segments
        graph: NetworkX graph
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Squared segment if successful, None otherwise
    """
    start_node = path[seg1_start]
    end_node = path[seg2_end]

    # Try both possible L-shaped connections
    if seg1_dir == "horizontal" and seg2_dir == "vertical":
        # Try horizontal-then-vertical
        corner1 = (start_node[0], end_node[1])
        # Try vertical-then-horizontal
        corner2 = (end_node[0], start_node[1])
    elif seg1_dir == "vertical" and seg2_dir == "horizontal":
        # Try vertical-then-horizontal
        corner1 = (end_node[0], start_node[1])
        # Try horizontal-then-vertical
        corner2 = (start_node[0], end_node[1])
    else:
        return None

    # Test both corner options
    for corner in [corner1, corner2]:
        if (
            corner != start_node
            and corner != end_node
            and graph.has_node(corner)
            and corner not in other_occupied_nodes
        ):
            # Check if both edges exist and are available
            edge1 = tuple(sorted([start_node, corner]))
            edge2 = tuple(sorted([corner, end_node]))

            if (
                graph.has_edge(*edge1)
                and graph.has_edge(*edge2)
                and edge1 not in other_occupied_edges
                and edge2 not in other_occupied_edges
            ):
                return [start_node, corner, end_node]

    return None


def _smooth_identified_corners(
    path: list,
    corners: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Apply aggressive smoothing to identified corners.

    Args:
        path: The current path
        corners: List of corner indices
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Smoothed path
    """
    if not corners:
        return path

    smoothed = path[:]

    # Process corners in reverse order to maintain indices
    for corner_idx in reversed(corners):
        # Try to smooth a region around this corner
        start_idx = max(0, corner_idx - 25)  # Look 25 nodes back
        end_idx = min(len(smoothed), corner_idx + 25)  # Look 25 nodes forward

        if end_idx - start_idx < 3:
            continue

        start_node = smoothed[start_idx]
        end_node = smoothed[end_idx - 1]

        # Try to create a direct path between these points
        direct_path = _find_direct_path_with_obstacles(
            start_node, end_node, graph, other_occupied_nodes, other_occupied_edges
        )

        if direct_path and len(direct_path) < (end_idx - start_idx):
            # Replace the segment with the direct path
            smoothed = smoothed[:start_idx] + direct_path + smoothed[end_idx:]

    return smoothed


def _optimize_straight_segments(
    path: list,
    segments: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Optimize long straight segments by checking if they can be shortened.

    Args:
        path: The current path
        segments: List of (start_idx, end_idx, direction) tuples
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Optimized path
    """
    if not segments:
        return path

    optimized = path[:]

    # Process segments in reverse order to maintain indices
    for start_idx, end_idx, direction in reversed(segments):
        if end_idx >= len(optimized) or start_idx < 0:
            continue

        start_node = optimized[start_idx]
        end_node = optimized[end_idx]

        # For straight segments, try direct connection
        if graph.has_edge(start_node, end_node):
            edge = tuple(sorted([start_node, end_node]))
            if edge not in other_occupied_edges:
                # Replace entire segment with direct connection
                optimized = (
                    optimized[:start_idx]
                    + [start_node, end_node]
                    + optimized[end_idx + 1 :]
                )

    return optimized


def _bridge_long_segments(
    path: list,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Try to bridge long distances with more direct paths.

    Args:
        path: The current path
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Path with long segments bridged
    """
    if len(path) < 20:
        return path

    bridged = [path[0]]
    i = 0

    while i < len(path) - 1:
        current = path[i]

        # Look for the furthest point we can reach directly
        best_jump = None
        best_distance = 0

        # Check increasingly distant points
        for jump_size in [50, 30, 20, 10, 5]:
            target_idx = min(i + jump_size, len(path) - 1)
            if target_idx <= i + 1:
                continue

            target = path[target_idx]

            # Try to find a direct path
            direct_path = _find_direct_path_with_obstacles(
                current, target, graph, other_occupied_nodes, other_occupied_edges
            )

            if (
                direct_path and len(direct_path) <= jump_size // 2
            ):  # Significant improvement
                best_jump = (target_idx, direct_path)
                best_distance = target_idx - i
                break  # Take the first (largest) jump that works

        if best_jump:
            target_idx, direct_path = best_jump
            bridged.extend(direct_path[1:])  # Skip the first node (already added)
            i = target_idx
        else:
            # No jump possible, add the next node
            if i + 1 < len(path):
                bridged.append(path[i + 1])
            i += 1

    return bridged


def _find_direct_path_with_obstacles(
    start: tuple,
    end: tuple,
    graph: nx.Graph,
    other_occupied_nodes: set,
    other_occupied_edges: set,
) -> list:
    """
    Find a direct path between two distant points, considering obstacles.

    Args:
        start: Starting node
        end: Ending node
        graph: NetworkX graph for connectivity checking
        other_occupied_nodes: Nodes occupied by other paths
        other_occupied_edges: Edges occupied by other paths

    Returns:
        Direct path if found, None otherwise
    """
    # Try simple L-shaped path first
    simple_path = _find_simple_path(
        start, end, graph, other_occupied_nodes, other_occupied_edges
    )
    if simple_path:
        return simple_path

    # Try a few intermediate points for longer distances
    row_diff = end[0] - start[0]
    col_diff = end[1] - start[1]

    # Try quarter points and three-quarter points
    for fraction in [0.25, 0.5, 0.75]:
        intermediate = (
            int(start[0] + row_diff * fraction),
            int(start[1] + col_diff * fraction),
        )

        if (
            intermediate != start
            and intermediate != end
            and graph.has_node(intermediate)
            and intermediate not in other_occupied_nodes
        ):
            # Try path through this intermediate point
            path1 = _find_simple_path(
                start, intermediate, graph, other_occupied_nodes, other_occupied_edges
            )
            path2 = _find_simple_path(
                intermediate, end, graph, other_occupied_nodes, other_occupied_edges
            )

            if path1 and path2:
                # Combine paths, avoiding duplicate intermediate node
                combined = path1 + path2[1:]
                return combined

    return None


def _update_global_tracking(
    old_path: list,
    new_path: list,
    all_occupied_nodes: set,
    all_occupied_edges: set,
):
    """
    Update global node and edge tracking when a path changes.

    Args:
        old_path: The original path
        new_path: The new path
        all_occupied_nodes: Global set of occupied nodes to update
        all_occupied_edges: Global set of occupied edges to update
    """
    # Remove old nodes and edges
    for node in set(old_path) - set(new_path):
        all_occupied_nodes.discard(node)

    old_edges = {
        tuple(sorted((old_path[i], old_path[i + 1]))) for i in range(len(old_path) - 1)
    }
    new_edges = {
        tuple(sorted((new_path[i], new_path[i + 1]))) for i in range(len(new_path) - 1)
    }

    for edge in old_edges - new_edges:
        all_occupied_edges.discard(edge)

    # Add new nodes and edges
    for node in set(new_path) - set(old_path):
        all_occupied_nodes.add(node)

    for edge in new_edges - old_edges:
        all_occupied_edges.add(edge)


# --- Path Straightening ---


def _straighten_paths(
    all_paths: List[List[Tuple[int, int]]], graph: nx.Graph, iterations: int = 10
) -> List[List[Tuple[int, int]]]:
    """
    Post-processes a set of paths to reduce corners and smooth long sections.

    This function applies corner detection and targeted smoothing for paths
    with hundreds of nodes, focusing on actual direction changes rather than
    processing every node. It continues iterating until no more improvements
    can be made.

    Args:
        all_paths: A list of paths to be straightened.
        graph: The graph on which the paths exist, used for weight lookups.
        iterations: The maximum number of times to repeat the straightening process.

    Returns:
        A new list of paths with corners potentially straightened and smoothed.
    """
    paths_to_modify = [list(p) for p in all_paths]
    max_iterations = iterations

    for iter_num in range(max_iterations):
        print(
            f"Step 5.1.4.{iter_num + 1}: Smoothing iteration {iter_num + 1}/{max_iterations}..."
        )
        paths_changed_in_iteration = False
        total_nodes_saved = 0

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

            # Create occupied sets for *other* paths for collision detection.
            current_path_nodes = set(path)
            current_path_edges = {
                tuple(sorted((path[i], path[i + 1]))) for i in range(len(path) - 1)
            }
            other_occupied_nodes = all_occupied_nodes - current_path_nodes
            other_occupied_edges = all_occupied_edges - current_path_edges

            original_length = len(path)
            path_improved_this_round = False

            # Keep trying to improve this path until no more improvements can be made
            path_optimization_iterations = 0
            max_path_iterations = 20

            while path_optimization_iterations < max_path_iterations:
                path_optimization_iterations += 1
                path_changed_this_iteration = False

                # Analyze path structure by reading graph edges directly
                path_structure = _analyze_path_structure(path)
                segments = path_structure["segments"]
                corners = path_structure["corners"]
                straight_runs = path_structure["straight_runs"]

                if (
                    path_optimization_iterations == 1
                ):  # Only print on first iteration for each path
                    print(
                        f"    Path {original_idx}: {len(path)} nodes, {len(segments)} segments, {len(corners)} corners, {len(straight_runs)} straight runs"
                    )

                    # Debug: Show first few segments and corners for longer paths
                    if len(path) > 50:
                        print(f"      First 5 segments: {segments[:5]}")
                        print(f"      First 10 corners: {corners[:10]}")
                        if len(path) > 10:
                            sample_nodes = [
                                f"({path[i][0]},{path[i][1]})"
                                for i in range(0, min(len(path), 20), 2)
                            ]
                            print(
                                f"      Sample path nodes: {' -> '.join(sample_nodes)}"
                            )

                # Pass 1: Direct corner squaring - join straight sections and eliminate unnecessary corners
                squared_path = _square_out_corners(
                    path,
                    path_structure,
                    graph,
                    other_occupied_nodes,
                    other_occupied_edges,
                )
                if len(squared_path) < len(path):
                    if path_optimization_iterations == 1:
                        print(
                            f"      Corner squaring: {len(path)} -> {len(squared_path)} nodes"
                        )
                    _update_global_tracking(
                        path, squared_path, all_occupied_nodes, all_occupied_edges
                    )
                    path[:] = squared_path
                    path_changed_this_iteration = True
                    path_improved_this_round = True

                # Pass 2: Single corner straightening for remaining corners
                corner_changes = 0
                modified_in_pass = True
                corner_pass_count = 0
                while modified_in_pass and len(path) >= 3 and corner_pass_count < 5:
                    modified_in_pass = False
                    corner_pass_count += 1
                    for i in range(1, len(path) - 1):
                        new_corner = _try_straighten_one_corner(
                            path, i, graph, other_occupied_nodes, other_occupied_edges
                        )
                        if new_corner:
                            old_path = path[:]
                            path[i] = new_corner
                            _update_global_tracking(
                                old_path, path, all_occupied_nodes, all_occupied_edges
                            )
                            modified_in_pass = True
                            path_changed_this_iteration = True
                            path_improved_this_round = True
                            corner_changes += 1
                            break

                if corner_changes > 0 and path_optimization_iterations == 1:
                    print(
                        f"      Single corner straightening: {corner_changes} corners modified"
                    )

                # Pass 3: Multi-segment bridging for long distances
                bridged_path = _bridge_long_segments(
                    path, graph, other_occupied_nodes, other_occupied_edges
                )
                if len(bridged_path) < len(path):
                    if path_optimization_iterations == 1:
                        print(
                            f"      Multi-segment bridging: {len(path)} -> {len(bridged_path)} nodes"
                        )
                    _update_global_tracking(
                        path, bridged_path, all_occupied_nodes, all_occupied_edges
                    )
                    path[:] = bridged_path
                    path_changed_this_iteration = True
                    path_improved_this_round = True

                # If no changes were made in this iteration, stop optimizing this path
                if not path_changed_this_iteration:
                    break

            # Update the path in the main list
            paths_to_modify[original_idx] = path

            if path_improved_this_round:
                paths_changed_in_iteration = True
                nodes_saved = original_length - len(path)
                total_nodes_saved += nodes_saved
                if nodes_saved > 0:
                    print(
                        f"    Path {original_idx}: {original_length} -> {len(path)} nodes (saved {nodes_saved}, {path_optimization_iterations} optimization iterations)"
                    )

        print(
            f"  Iteration {iter_num + 1} complete: {total_nodes_saved} total nodes saved"
        )

        # If no paths were improved in this iteration, stop the overall process
        if not paths_changed_in_iteration:
            print(
                f"  No further improvements possible. Stopping after {iter_num + 1} iterations."
            )
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
    substation_pairs: List[tuple] = None,
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
        for more chaotic path exploration and ramps up aggressively
        towards the end to force convergence on a low-congestion solution.
    4.  **Adjacent Routing**: Connections between the same substation pairs
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
    if substation_pairs:
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
            group_requests.sort(key=lambda x: manhattan_distance(x[1]["start"], x[1]["end"]), reverse=True)
            
            # Use shortest path in group for overall group sorting
            min_distance = min(manhattan_distance(path_requests[i]["start"], path_requests[i]["end"]) for i in indices)
            sorted_group_indices.append((min_distance, [x[0] for x in group_requests]))
        
        # Sort groups by their minimum distance
        sorted_group_indices.sort(key=lambda x: x[0], reverse=True)
        
        # Flatten to get final order
        indexed_requests = []
        for _, group_indices in sorted_group_indices:
            for idx in group_indices:
                indexed_requests.append((idx, path_requests[idx]))
    else:
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
            current_substation_pair = substation_pairs[req_idx] if substation_pairs else None
            applied_penalties = _apply_penalties_to_graph(
                graph, edge_usage, node_usage, current_penalty, start_node, end_node,
                substation_pair=current_substation_pair,
                all_paths=all_paths,
                current_path_idx=req_idx,
                substation_pairs=substation_pairs
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
    all_paths = _straighten_paths(all_paths, graph, iterations=10)

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
