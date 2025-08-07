"""
Bespoke algorithm to space rectangles in 2D space to guarantee no overlap/intersection between them.
"""

import math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional


def space_rectangles(
    rectangles: List[Tuple[float, float, float, float]],
    grid_size: int = 25,
    debug_images: bool = False,
    padding_steps: Optional[List[int]] = None,
) -> List[Tuple[float, float]]:
    """
    Space rectangles to guarantee no overlap/intersection between them.

    Args:
        rectangles: List of tuples (x1, y1, x2, y2) representing rectangle corners
        grid_size: Grid size for snapping movements (default: 25)
        debug_images: Whether to generate PNG images for each iteration (default: False)
        padding_steps: Number of grid_size steps to add as padding on each side.

    Returns:
        List of tuples (xshift, yshift) representing how far each rectangle moved
    """
    # Track original positions and current positions
    original_positions = rectangles.copy()

    if padding_steps is None:
        padding_steps = [1] * len(rectangles)

    # Snap rectangles to the nearest grid_size (25x25) grid
    current_positions = []
    shifts: List[Tuple[float, float]] = []
    for i, (x1, y1, x2, y2) in enumerate(rectangles):
        snapped_x1 = round(x1 / grid_size) * grid_size
        snapped_y1 = round(y1 / grid_size) * grid_size
        snapped_x2 = round(x2 / grid_size) * grid_size
        snapped_y2 = round(y2 / grid_size) * grid_size
        current_positions.append((snapped_x1, snapped_y1, snapped_x2, snapped_y2))

        # Calculate shifts due to snapping (in pixels)
        x_shift = snapped_x1 - x1
        y_shift = snapped_y1 - y1
        shifts.append((x_shift, y_shift))

    # Calculate final target dimensions for each rectangle
    target_half_widths = []
    target_half_heights = []
    for i, (x1, y1, x2, y2) in enumerate(original_positions):
        padding = padding_steps[i] * grid_size
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        target_half_widths.append(width / 2 + padding)
        target_half_heights.append(height / 2 + padding)

    # Determine max growth steps needed
    max_target_half_dim = 0
    if target_half_widths:
        max_target_half_dim = max(max(target_half_widths), max(target_half_heights))

    # Growth starts from grid_size/2 and increases by grid_size each step
    num_growth_steps = (
        math.ceil((max_target_half_dim - grid_size / 2) / grid_size)
        if grid_size > 0
        else 0
    )
    num_growth_steps = max(0, num_growth_steps)  # ensure non-negative

    max_iterations = 1000  # Safety limit

    for growth_step in range(num_growth_steps + 1):
        print(f"--- Processing growth step {growth_step}/{num_growth_steps} ---")

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"Growth step {growth_step}, Iteration {iteration}")

            # Create shapely polygons for current positions with growing sizes
            polygons = []
            for i, (x1, y1, x2, y2) in enumerate(current_positions):
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Calculate current size for this growth step
                # Starts at grid_size/2 and grows by grid_size each step
                current_growth = (growth_step + 0.5) * grid_size
                half_width = min(current_growth, target_half_widths[i])
                half_height = min(current_growth, target_half_heights[i])

                poly = Polygon(
                    [
                        (center_x - half_width, center_y - half_height),
                        (center_x + half_width, center_y - half_height),
                        (center_x + half_width, center_y + half_height),
                        (center_x - half_width, center_y + half_height),
                    ]
                )
                polygons.append(poly)

            # Initialize force tracking for each rectangle
            force_to_be_applied = [False] * len(rectangles)
            directions = [None] * len(rectangles)

            # Check all rectangles against each other
            conflicts_found = False

            for i in range(len(polygons)):
                for j in range(i + 1, len(polygons)):
                    poly_i, poly_j = polygons[i], polygons[j]

                    # Check if one is wholly inside the other
                    if poly_i.within(poly_j):
                        conflicts_found = True
                        force_to_be_applied[i] = True
                        force_to_be_applied[j] = True

                        # Calculate midpoints
                        mid_i = (poly_i.centroid.x, poly_i.centroid.y)
                        mid_j = (poly_j.centroid.x, poly_j.centroid.y)

                        if (
                            abs(mid_i[0] - mid_j[0]) < 1e-6
                            and abs(mid_i[1] - mid_j[1]) < 1e-6
                        ):
                            # Same midpoints - move inner rectangle down
                            force_to_be_applied[j] = False  # Don't move outer rectangle
                            _apply_direction(directions, i, 270)  # Move inner down
                        else:
                            # Move away from each other
                            angle_i_to_j = math.degrees(
                                math.atan2(mid_j[1] - mid_i[1], mid_j[0] - mid_i[0])
                            )
                            angle_j_to_i = (angle_i_to_j + 180) % 360

                            _apply_direction(
                                directions, i, angle_j_to_i
                            )  # Move i away from j
                            _apply_direction(
                                directions, j, angle_i_to_j
                            )  # Move j away from i

                    elif poly_j.within(poly_i):
                        conflicts_found = True
                        force_to_be_applied[i] = True
                        force_to_be_applied[j] = True

                        # Calculate midpoints
                        mid_i = (poly_i.centroid.x, poly_i.centroid.y)
                        mid_j = (poly_j.centroid.x, poly_j.centroid.y)

                        if (
                            abs(mid_i[0] - mid_j[0]) < 1e-6
                            and abs(mid_i[1] - mid_j[1]) < 1e-6
                        ):
                            # Same midpoints - move inner rectangle down
                            force_to_be_applied[i] = False  # Don't move outer rectangle
                            _apply_direction(directions, j, 270)  # Move inner down
                        else:
                            # Move away from each other
                            angle_i_to_j = math.degrees(
                                math.atan2(mid_j[1] - mid_i[1], mid_j[0] - mid_i[0])
                            )
                            angle_j_to_i = (angle_i_to_j + 180) % 360

                            _apply_direction(
                                directions, i, angle_j_to_i
                            )  # Move i away from j
                            _apply_direction(
                                directions, j, angle_i_to_j
                            )  # Move j away from i

                    # Check for intersection
                    elif poly_i.intersects(poly_j) and not poly_i.touches(poly_j):
                        conflicts_found = True
                        force_to_be_applied[i] = True
                        force_to_be_applied[j] = True

                        # Get intersection
                        intersection = poly_i.intersection(poly_j)

                        # Calculate midpoints for fallback
                        mid_i = (poly_i.centroid.x, poly_i.centroid.y)
                        mid_j = (poly_j.centroid.x, poly_j.centroid.y)

                        # Move away from each other based on midpoints
                        angle_i_to_j = math.degrees(
                            math.atan2(mid_j[1] - mid_i[1], mid_j[0] - mid_i[0])
                        )
                        angle_j_to_i = (angle_i_to_j + 180) % 360

                        _apply_direction(
                            directions, i, angle_j_to_i
                        )  # Move i away from j
                        _apply_direction(
                            directions, j, angle_i_to_j
                        )  # Move j away from i

            # If no conflicts found, we're done with this growth step's iterations
            if not conflicts_found:
                break

            # Apply forces
            for i in range(len(current_positions)):
                if force_to_be_applied[i] and directions[i] is not None:
                    # Calculate movement
                    angle_rad = math.radians(directions[i])
                    dx = grid_size * math.cos(angle_rad)
                    dy = grid_size * math.sin(angle_rad)

                    # Apply movement
                    x1, y1, x2, y2 = current_positions[i]
                    new_x1 = x1 + dx
                    new_y1 = y1 + dy
                    new_x2 = x2 + dx
                    new_y2 = y2 + dy

                    # Snap to grid
                    new_x1 = round(new_x1 / grid_size) * grid_size
                    new_y1 = round(new_y1 / grid_size) * grid_size
                    new_x2 = round(new_x2 / grid_size) * grid_size
                    new_y2 = round(new_y2 / grid_size) * grid_size

                    current_positions[i] = (new_x1, new_y1, new_x2, new_y2)

                    # Update shifts
                    orig_x1, orig_y1, _, _ = original_positions[i]
                    shifts[i] = (new_x1 - orig_x1, new_y1 - orig_y1)

            # Generate debug image if requested
            if debug_images:
                rects_for_debug = [p.bounds for p in polygons]
                _generate_debug_image(rects_for_debug, iteration, growth_step)

    return shifts


def _apply_direction(
    directions: List[Optional[float]], index: int, new_angle: float
) -> None:
    """Apply a new direction to a rectangle, combining with existing direction if present."""
    if directions[index] is None:
        directions[index] = new_angle
    else:
        # Convert angles to vectors, add them, then convert back to angle
        existing_rad = math.radians(directions[index])
        new_rad = math.radians(new_angle)

        # Convert to unit vectors
        existing_x = math.cos(existing_rad)
        existing_y = math.sin(existing_rad)
        new_x = math.cos(new_rad)
        new_y = math.sin(new_rad)

        # Add vectors
        combined_x = existing_x + new_x
        combined_y = existing_y + new_y

        # Convert back to angle
        if abs(combined_x) < 1e-10 and abs(combined_y) < 1e-10:
            # Vectors cancel out, keep existing direction
            pass
        else:
            combined_angle = math.degrees(math.atan2(combined_y, combined_x))
            directions[index] = combined_angle % 360


def _generate_debug_image(
    rectangles: List[Tuple[float, float, float, float]],
    iteration: int,
    growth_step: int,
) -> None:
    """Generate a debug PNG image showing rectangle positions."""
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
