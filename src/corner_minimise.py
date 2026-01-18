def steps_to_segments(steps: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Calculates the segment list from the path.

    Converts a list like [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    into [(2, 0), (0, 2)] where each tuple represents a condensed
    segment in a single direction.

    Args:
        steps: List of absolute coordinates representing the path

    Returns:
        list[tuple[int, int]]: Condensed segment list with relative movements
    """
    if len(steps) <= 1:
        return []

    # Convert absolute coordinates to deltas (relative movements)
    delta_list = []
    for i in range(len(steps) - 1):
        dx = steps[i + 1][0] - steps[i][0]
        dy = steps[i + 1][1] - steps[i][1]
        delta_list.append((dx, dy))

    # Use optimise_segment to condense consecutive same-direction movements
    return optimise_segment(delta_list)


def optimise_segment(segment_in: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Given a list of tuples, check if any two subsequent entries
    are the same direction e.g. up/down or left/right - if so, combine

    Args:
        segment_in (list[tuple[int, int]]): Original unoptimised segment

    Returns:
        list[tuple[int, int]]: Optimised segment
    """
    if len(segment_in) <= 1:
        return segment_in

    optimised_segments: list[tuple[int, int]] = [segment_in[0]]

    for step in segment_in[1:]:
        last_segment = optimised_segments[-1]

        # Check if both segments are in the same direction
        # Both horizontal (x changes, y is 0 for both)
        both_horizontal = last_segment[1] == 0 and step[1] == 0
        # Both vertical (y changes, x is 0 for both)
        both_vertical = last_segment[0] == 0 and step[0] == 0

        if both_horizontal or both_vertical:
            # Combine by adding the deltas together
            combined_segment = (
                last_segment[0] + step[0],
                last_segment[1] + step[1],
            )
            # Replace the last segment with the combined one
            optimised_segments[-1] = combined_segment
        else:
            # Different directions, add as new segment
            optimised_segments.append(step)

    return optimised_segments


def segment_to_path(
    segment_in: list[tuple[int, int]], starting_point: tuple[int, int]
) -> list[tuple[int, int]]:
    """Converts a condensed segment list back into a full path of absolute coordinates.

    This is the inverse of steps_to_segments. Converts a list like [(2, 0), (0, 2)]
    back into [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)].

    Args:
        segment_in: Condensed segment list with relative movements

    Returns:
        list[tuple[int, int]]: Full path as absolute coordinates
    """
    if len(segment_in) == 0:
        return [starting_point]

    path = [starting_point]  # Start at origin
    current_x, current_y = starting_point[0], starting_point[1]

    for dx, dy in segment_in:
        # Determine the direction and number of steps
        if dx != 0:
            # Horizontal movement
            step_x = 1 if dx > 0 else -1
            num_steps = abs(dx)
            for _ in range(num_steps):
                current_x += step_x
                path.append((current_x, current_y))
        elif dy != 0:
            # Vertical movement
            step_y = 1 if dy > 0 else -1
            num_steps = abs(dy)
            for _ in range(num_steps):
                current_y += step_y
                path.append((current_x, current_y))

    return path


def find_alternative_paths(
    segment: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Find alternative paths by swapping adjacent segment pairs.

    Searches for L/R then U/D then L/R patterns (and U/D then L/R then U/D)
    and creates alternative paths by swapping the order of the middle segment
    with either the previous or next segment.

    Args:
        segment: Condensed segment list with relative movements

    Returns:
        list[list[tuple[int, int]]]: List of alternative segment paths
    """
    alternative_paths: list[list[tuple[int, int]]] = []

    if len(segment) < 3:
        return []

    def _find_alternative(
        segment_in: list[tuple[int, int]],
    ) -> list[list[tuple[int, int]]]:
        inner_paths: list[list[tuple[int, int]]] = []
        for i in range(1, len(segment_in) - 1):
            prev_segment = segment_in[i - 1]
            current_segment = segment_in[i]
            next_segment = segment_in[i + 1]

            # Check if there is a corner to consider
            # ... if both the x and y has changed
            # sum all three directions together
            sum_vector = (
                prev_segment[0] + current_segment[0] + next_segment[0],
                prev_segment[1] + current_segment[1] + next_segment[1],
            )
            if sum_vector[0] != 0 and sum_vector[1] != 0:
                # This is a potential double corner/dog leg
                # Create alternative path by swapping prev and current segments
                inner_paths.append(
                    segment_in[: i - 1]
                    + [current_segment, prev_segment]
                    + segment_in[i + 1 :]
                )

                # Create alternative path by swapping current and next segments
                inner_paths.append(
                    segment_in[:i]
                    + [next_segment, current_segment]
                    + segment_in[i + 2 :]
                )
        # optimise each segment
        inner_paths = [optimise_segment(segment) for segment in inner_paths]
        return inner_paths

    new_paths = _find_alternative(segment_in=segment)
    for path in new_paths:
        if path not in alternative_paths:
            alternative_paths.append(path)

    return alternative_paths
