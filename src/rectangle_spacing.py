"""
Fast iterative algorithm to space rectangles to guarantee no overlap/intersection between them.
"""

import math
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


def _classify_rectangles(
    rectangles: List[Tuple[float, float, float, float]]
) -> Tuple[List[int], List[int]]:
    """Classifies rectangles into 'large' and 'small' based on area.

    Large rectangles are those in the 75th percentile of area or larger.

    Args:
        rectangles: List of rectangles (x1, y1, x2, y2).

    Returns:
        A tuple containing two lists: indices of large rectangles, and indices of small rectangles.
    """
    if len(rectangles) < 4:
        return list(range(len(rectangles))), []

    areas = [abs(r[2] - r[0]) * abs(r[3] - r[1]) for r in rectangles]
    if not areas:
        return [], []

    threshold = np.percentile(areas, 75)

    # If threshold is 0, all are large to avoid division issues or strange behavior
    if threshold == 0:
        return list(range(len(rectangles))), []

    large_indices = [i for i, area in enumerate(areas) if area >= threshold]
    small_indices = [i for i, area in enumerate(areas) if area < threshold]

    # Ensure there's at least one large rectangle if possible
    if not large_indices and small_indices:
        max_area = -1
        max_idx = -1
        for i in small_indices:
            if areas[i] > max_area:
                max_area = areas[i]
                max_idx = i
        large_indices.append(max_idx)
        small_indices.remove(max_idx)

    return large_indices, small_indices


def _group_small_rectangles(
    rectangles: List[Tuple[float, float, float, float]],
    large_indices: List[int],
    small_indices: List[int],
) -> dict:
    """Groups small rectangles with their nearest large rectangle.

    Args:
        rectangles: List of all rectangles.
        large_indices: List of indices for large rectangles.
        small_indices: List of indices for small rectangles.

    Returns:
        A dictionary mapping each large rectangle index to a list of its associated small rectangle indices.
    """
    if not large_indices or not small_indices:
        return {}

    centers = _get_rectangle_centers(rectangles)
    large_centers = {i: centers[i] for i in large_indices}

    groups = {i: [] for i in large_indices}

    for small_idx in small_indices:
        small_center = centers[small_idx]

        # Find closest large rectangle
        closest_large_idx = -1
        min_dist_sq = float("inf")

        for large_idx, large_center in large_centers.items():
            dist_sq = (small_center[0] - large_center[0]) ** 2 + (
                small_center[1] - large_center[1]
            ) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_large_idx = large_idx

        if closest_large_idx != -1:
            groups[closest_large_idx].append(small_idx)

    return groups


def _get_rectangle_centers(
    rectangles: List[Tuple[float, float, float, float]],
) -> List[Tuple[float, float]]:
    """Get center points of all rectangles.

    Args:
        rectangles: List of rectangles (x1, y1, x2, y2).

    Returns:
        List of center points (x, y).
    """
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in rectangles]


def _calculate_initial_relationships(
    rectangles: List[Tuple[float, float, float, float]],
) -> dict:
    """Calculate initial relative angles and distances between all rectangle pairs.

    Args:
        rectangles: The initial list of rectangles (x1, y1, x2, y2).

    Returns:
        Dictionary mapping (i, j) pairs to (angle, distance) tuples.
    """
    relationships = {}
    centers = _get_rectangle_centers(rectangles)

    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            center_i = centers[i]
            center_j = centers[j]

            dx = center_j[0] - center_i[0]
            dy = center_j[1] - center_i[1]

            distance = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) if distance > 1e-10 else 0

            relationships[(i, j)] = (angle, distance)

    return relationships


def _calculate_minimum_separation(
    rect1: Tuple[float, float, float, float],
    rect2: Tuple[float, float, float, float],
    padding1: float,
    padding2: float,
) -> float:
    """Calculate minimum required separation between rectangle centers.

    Args:
        rect1: First rectangle (x1, y1, x2, y2).
        rect2: Second rectangle (x1, y1, x2, y2).
        padding1: Padding for first rectangle.
        padding2: Padding for second rectangle.

    Returns:
        Minimum distance required between centers.
    """
    width1 = abs(rect1[2] - rect1[0])
    height1 = abs(rect1[3] - rect1[1])
    width2 = abs(rect2[2] - rect2[0])
    height2 = abs(rect2[3] - rect2[1])

    # Conservative estimate: half-widths + half-heights + paddings
    min_separation = (
        (width1 + width2) / 2 + (height1 + height2) / 2 + padding1 + padding2
    )

    return min_separation


def _check_overlap_simple(
    rect1: Tuple[float, float, float, float],
    rect2: Tuple[float, float, float, float],
    padding1: float,
    padding2: float,
) -> bool:
    """Fast overlap check using bounding boxes with padding.

    Args:
        rect1: First rectangle (x1, y1, x2, y2).
        rect2: Second rectangle (x1, y1, x2, y2).
        padding1: Padding for first rectangle.
        padding2: Padding for second rectangle.

    Returns:
        True if rectangles overlap with padding, False otherwise.
    """
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2

    # Expand rectangles by padding
    x1_1 -= padding1
    y1_1 -= padding1
    x2_1 += padding1
    y2_1 += padding1

    x1_2 -= padding2
    y1_2 -= padding2
    x2_2 += padding2
    y2_2 += padding2

    # Check for overlap
    return not (x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1)


def _separate_overlapping_rectangles(
    rectangles: List[Tuple[float, float, float, float]],
    padding_steps: List[int],
    grid_size: int,
) -> List[Tuple[float, float, float, float]]:
    """Separate overlapping rectangles while maintaining relative angles.

    Args:
        rectangles: List of rectangles (x1, y1, x2, y2).
        padding_steps: Padding steps for each rectangle.
        grid_size: Grid size for padding calculations.

    Returns:
        List of separated rectangles.
    """
    current_rects = [list(rect) for rect in rectangles]  # Make mutable
    paddings = [steps * grid_size for steps in padding_steps]

    # Calculate initial relationships to preserve
    initial_relationships = _calculate_initial_relationships(rectangles)

    max_iterations = 50  # Much lower iteration limit

    for iteration in range(max_iterations):
        moved_any = False

        # Check all pairs for overlaps and angle preservation
        for i in range(len(current_rects)):
            for j in range(i + 1, len(current_rects)):
                rect_i = tuple(current_rects[i])
                rect_j = tuple(current_rects[j])

                # Calculate current centers
                center_i_x = (rect_i[0] + rect_i[2]) / 2
                center_i_y = (rect_i[1] + rect_i[3]) / 2
                center_j_x = (rect_j[0] + rect_j[2]) / 2
                center_j_y = (rect_j[1] + rect_j[3]) / 2

                # Calculate current distance and angle
                current_dx = center_j_x - center_i_x
                current_dy = center_j_y - center_i_y
                current_distance = math.sqrt(
                    current_dx * current_dx + current_dy * current_dy
                )

                # Get initial relationship
                initial_angle, initial_distance = initial_relationships.get(
                    (i, j), (0, 100)
                )

                # Calculate minimum required separation
                min_separation = _calculate_minimum_separation(
                    rect_i, rect_j, paddings[i], paddings[j]
                )

                # Determine target distance (larger of initial or minimum required)
                target_distance = max(initial_distance, min_separation)

                # Check if adjustment is needed
                needs_adjustment = False
                if _check_overlap_simple(rect_i, rect_j, paddings[i], paddings[j]):
                    needs_adjustment = True
                elif current_distance < min_separation:
                    needs_adjustment = True

                if needs_adjustment:
                    moved_any = True

                    if current_distance < 1e-6:
                        # Same position - use initial angle to separate
                        target_j_x = center_i_x + target_distance * math.cos(
                            initial_angle
                        )
                        target_j_y = center_i_y + target_distance * math.sin(
                            initial_angle
                        )

                        # Move j to target position
                        width_j = rect_j[2] - rect_j[0]
                        height_j = rect_j[3] - rect_j[1]

                        current_rects[j][0] = target_j_x - width_j / 2
                        current_rects[j][1] = target_j_y - height_j / 2
                        current_rects[j][2] = target_j_x + width_j / 2
                        current_rects[j][3] = target_j_y + height_j / 2
                    else:
                        # Calculate ideal position for j to maintain initial angle
                        ideal_j_x = center_i_x + target_distance * math.cos(
                            initial_angle
                        )
                        ideal_j_y = center_i_y + target_distance * math.sin(
                            initial_angle
                        )

                        # Calculate movement needed
                        move_j_x = (ideal_j_x - center_j_x) * 0.3  # Damping factor
                        move_j_y = (ideal_j_y - center_j_y) * 0.3

                        # Also move i slightly in opposite direction for stability
                        move_i_x = -move_j_x * 0.2
                        move_i_y = -move_j_y * 0.2

                        # Apply movements
                        width_i = rect_i[2] - rect_i[0]
                        height_i = rect_i[3] - rect_i[1]
                        width_j = rect_j[2] - rect_j[0]
                        height_j = rect_j[3] - rect_j[1]

                        # Move rectangle i
                        current_rects[i][0] += move_i_x
                        current_rects[i][1] += move_i_y
                        current_rects[i][2] = current_rects[i][0] + width_i
                        current_rects[i][3] = current_rects[i][1] + height_i

                        # Move rectangle j
                        current_rects[j][0] += move_j_x
                        current_rects[j][1] += move_j_y
                        current_rects[j][2] = current_rects[j][0] + width_j
                        current_rects[j][3] = current_rects[j][1] + height_j

        if not moved_any:
            print(f"No overlaps found, converged at iteration {iteration + 1}")
            break

    return [tuple(rect) for rect in current_rects]


def _snap_to_grid(
    current_positions: List[Tuple[float, float, float, float]],
    original_positions: List[Tuple[float, float, float, float]],
    shifts: List[Tuple[float, float]],
    grid_size: int,
):
    """Snap all positions to grid and update final shifts.

    Args:
        current_positions: The list of current rectangle positions to be modified.
        original_positions: The original, unmodified positions.
        shifts: The list of total shifts to be updated.
        grid_size: The grid size to snap to.
    """
    for i in range(len(current_positions)):
        x1, y1, x2, y2 = current_positions[i]

        # Snap to grid
        snapped_x1 = round(x1 / grid_size) * grid_size
        snapped_y1 = round(y1 / grid_size) * grid_size
        snapped_x2 = round(x2 / grid_size) * grid_size
        snapped_y2 = round(y2 / grid_size) * grid_size

        current_positions[i] = (snapped_x1, snapped_y1, snapped_x2, snapped_y2)

        # Update final shifts
        orig_x1, orig_y1, _, _ = original_positions[i]
        shifts[i] = (snapped_x1 - orig_x1, snapped_y1 - orig_y1)


def space_rectangles(
    rectangles: List[Tuple[float, float, float, float]],
    grid_size: int = 25,
    debug_images: bool = False,
    padding_steps: Optional[List[int]] = None,
    map_bounds: Tuple[float, float] = (22000, 22000),
) -> List[Tuple[float, float]]:
    """
    Space rectangles using a hierarchical iterative algorithm to guarantee no overlap.
    This method first spaces out larger "anchor" rectangles, moves associated smaller
    rectangles with them, and then performs a final pass to resolve all remaining overlaps.

    Args:
        rectangles: List of tuples (x1, y1, x2, y2) representing rectangle corners
        grid_size: Grid size for snapping movements (default: 25)
        debug_images: Whether to generate PNG images for each iteration (default: False)
        padding_steps: Number of grid_size steps to add as padding on each side.
        map_bounds: Tuple of (max_x, max_y) bounds for the map

    Returns:
        List of tuples (xshift, yshift) representing how far each rectangle moved
    """
    original_positions = rectangles.copy()
    if padding_steps is None:
        padding_steps = [1] * len(rectangles)

    print("Step 2.3: Using hierarchical iterative algorithm for rectangle spacing...")

    # 1. Classify rectangles into large and small based on area
    large_indices, small_indices = _classify_rectangles(rectangles)

    # If no meaningful groups can be formed, use the simple single-pass algorithm
    if not large_indices or not small_indices:
        print("  No large/small distinction possible, running single-pass separation.")
        separated_rects = _separate_overlapping_rectangles(
            rectangles, padding_steps, grid_size
        )
    else:
        print(
            f"  Classified {len(large_indices)} large and {len(small_indices)} small rectangles."
        )

        # 2. Group small rectangles with their nearest large one
        groups = _group_small_rectangles(rectangles, large_indices, small_indices)

        # 3. First pass: space out large rectangles only
        print("  Running separation pass for large rectangles...")
        large_rects = [rectangles[i] for i in large_indices]
        large_padding_steps = [padding_steps[i] for i in large_indices]

        separated_large_rects = _separate_overlapping_rectangles(
            large_rects, large_padding_steps, grid_size
        )

        # Calculate shifts for large rectangles
        large_rect_shifts = {}
        for i, original_large_index in enumerate(large_indices):
            orig_rect = rectangles[original_large_index]
            new_rect = separated_large_rects[i]
            shift_x = new_rect[0] - orig_rect[0]
            shift_y = new_rect[1] - orig_rect[1]
            large_rect_shifts[original_large_index] = (shift_x, shift_y)

        # Apply shifts to create intermediate positions for all rectangles
        intermediate_rects = [list(r) for r in rectangles]
        for large_index, shift in large_rect_shifts.items():
            # Apply shift to the large rectangle itself
            r_large = intermediate_rects[large_index]
            r_large[0] += shift[0]
            r_large[1] += shift[1]
            r_large[2] += shift[0]
            r_large[3] += shift[1]

            # Apply the same shift to all associated small rectangles
            for small_index in groups.get(large_index, []):
                r_small = intermediate_rects[small_index]
                r_small[0] += shift[0]
                r_small[1] += shift[1]
                r_small[2] += shift[0]
                r_small[3] += shift[1]

        intermediate_rects_tuples = [tuple(r) for r in intermediate_rects]

        # 4. Second pass: run separation on all rectangles from their new positions
        print("  Running final separation pass for all rectangles...")
        separated_rects = _separate_overlapping_rectangles(
            intermediate_rects_tuples, padding_steps, grid_size
        )

    print("Rectangle separation completed")

    # Apply bounds checking to ensure rectangles stay within map bounds
    max_x, max_y = map_bounds
    margin = 100  # Small margin from edges

    for i, (x1, y1, x2, y2) in enumerate(separated_rects):
        # Check if rectangle is outside bounds
        if x2 > max_x - margin or y2 > max_y - margin or x1 < margin or y1 < margin:
            # Calculate rectangle dimensions
            width = x2 - x1
            height = y2 - y1

            # Clamp position to stay within bounds
            new_x1 = max(margin, min(x1, max_x - margin - width))
            new_y1 = max(margin, min(y1, max_y - margin - height))
            new_x2 = new_x1 + width
            new_y2 = new_y1 + height

            separated_rects[i] = (new_x1, new_y1, new_x2, new_y2)
            print(
                f"  Clamped rectangle {i} to bounds: ({new_x1:.1f}, {new_y1:.1f}, {new_x2:.1f}, {new_y2:.1f})"
            )

    # Calculate shifts
    shifts = []
    for i, (orig_rect, new_rect) in enumerate(zip(original_positions, separated_rects)):
        orig_x1, orig_y1, _, _ = orig_rect
        new_x1, new_y1, _, _ = new_rect
        shifts.append((new_x1 - orig_x1, new_y1 - orig_y1))

    # Snap to grid at the very end
    _snap_to_grid(separated_rects, original_positions, shifts, grid_size)

    if debug_images:
        _generate_debug_image(separated_rects, 1, 0)

    return shifts


def _generate_debug_image(
    rectangles: List[Tuple[float, float, float, float]],
    iteration: int,
    growth_step: int,
) -> None:
    """Generates a debug PNG image showing rectangle positions.

    Args:
        rectangles: The list of rectangles to draw.
        iteration: The current iteration number, for the filename.
        growth_step: The current growth step, for the filename.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, (x1, y1, x2, y2) in enumerate(rectangles):
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)

        rect = patches.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            linewidth=2,
            edgecolor=colors[i % len(colors)],
            facecolor=colors[i % len(colors)],
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Add rectangle number
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        ax.text(
            center_x,
            center_y,
            str(i),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Rectangle Positions - Growth {growth_step}, Iteration {iteration}")
    ax.autoscale_view()
    plt.savefig(
        f"rectangles_growth_{growth_step}_iteration_{iteration:03d}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
