import drawsvg as draw


def draw_3w_tx_left_right(
    obj_x: float,
    obj_y: float,
    grid_step: int,
    colour_map: dict[int, str],
    winding_voltages: tuple[float, float, float],
) -> tuple[draw.Group, dict[int, dict], list[tuple[float, float]]]:
    """Draws a horizontal (left-right) three winding transformer at
    the specified location

    Args:
        obj_x (float): X coordinate of transformer
        obj_y (float): Y coordnate of transformer
        grid_step (int): Grid step used to draw the object
        colour_map (dict[int, str]): Map of voltage levels to colours
        winding_voltages (tuple[float, float, float]): W1/W2/W3 voltages

    Returns:
        tuple[draw.Group, dict[int, dict], list[tuple[float, float]]]:
            - [0] is the draw.Group with the geometry
            - [1] is a connection point dictionary
            - [2] is a list of grid points to mark as element
    """
    obj_group = draw.Group()

    # winding 1 at left
    # winding 2 at right
    # winding 3 is in the middle, then offset down (south)
    # total width is 4*grid_step, with height of 2*grid_step

    radius = (2 * grid_step) / 3
    symbol_left_x = obj_x + grid_step
    symbol_center_x = symbol_left_x + grid_step

    circle1_y = circle2_y = obj_y
    circle1_x = symbol_center_x - radius / 2
    circle2_x = symbol_center_x + radius / 2
    circle3_x = symbol_center_x
    circle3_y = obj_y + radius

    # unpack colours
    w1colour = colour_map.get(winding_voltages[0], "black")
    w2colour = colour_map.get(winding_voltages[1], "black")
    w3colour = colour_map.get(winding_voltages[2], "black")

    # --- Left Terminal Line ---
    left_line_start_x = obj_x
    left_line_end_x = symbol_left_x
    left_line = draw.Line(
        left_line_start_x,
        obj_y,
        left_line_end_x,
        obj_y,
        stroke=w1colour,
        stroke_width=2,
    )
    obj_group.append(left_line)

    # --- Left Circle ---
    left_circle = draw.Circle(
        circle1_x,
        circle1_y,
        radius,
        fill="transparent",
        stroke=w1colour,
        stroke_width=3,
    )
    obj_group.append(left_circle)

    # --- Right Circle ---
    right_circle = draw.Circle(
        circle2_x,
        circle2_y,
        radius,
        fill="transparent",
        stroke=w2colour,
        stroke_width=3,
    )
    obj_group.append(right_circle)

    # --- Right Terminal Line ---
    right_line_start_x = symbol_left_x + 2 * grid_step
    right_line_end_x = right_line_start_x + grid_step
    right_line = draw.Line(
        right_line_start_x,
        obj_y,
        right_line_end_x,
        obj_y,
        stroke=w2colour,
        stroke_width=2,
    )
    obj_group.append(right_line)

    # tertiary winding circle
    tertiary_circle = draw.Circle(
        circle3_x,
        circle3_y,
        radius,
        fill="transparent",
        stroke=w3colour,
        stroke_width=3,
    )
    obj_group.append(tertiary_circle)

    # draw tertiary line
    tert_line_start_y = circle3_y + radius
    tert_line_end_y = circle3_y + radius + grid_step
    tertiary_line = draw.Line(
        circle3_x,
        tert_line_start_y,
        circle3_x,
        tert_line_end_y,
        stroke=w3colour,
        stroke_width=2,
    )
    obj_group.append(tertiary_line)

    # Mark grid points for the transformer elements
    grid_points_to_mark = [
        (left_line_start_x, obj_y),
        (left_line_end_x, obj_y),
        (symbol_center_x, obj_y),
        (right_line_start_x, obj_y),
        (right_line_end_x, obj_y),
    ]

    # terminal dict
    terminals = {
        1: {
            "coords": (left_line_start_x, obj_y),  # left terminal
            "voltage": winding_voltages[0],
        },
        2: {
            "coords": (right_line_end_x, obj_y),  # right terminal
            "voltage": winding_voltages[1],
        },
        3: {
            "coords": (circle3_x, tert_line_end_y),  # tertiary terminal
            "voltage": winding_voltages[2],
        },
    }

    # return the drawing group and the grid points to mark
    return obj_group, terminals, grid_points_to_mark


def draw_3w_tx_up_down(
    obj_x: float,
    obj_y: float,
    grid_step: int,
    colour_map: dict[int, str],
    winding_voltages: tuple[float, float, float],
) -> tuple[draw.Group, dict[int, dict], list[tuple[float, float]]]:
    """Draws a vertical (up-down) three winding transformer at
    the specified location

    Args:
        obj_x (float): X coordinate of transformer
        obj_y (float): Y coordnate of transformer
        grid_step (int): Grid step used to draw the object
        colour_map (dict[int, str]): Map of voltage levels to colours
        winding_voltages (tuple[float, float, float]): W1/W2/W3 voltages

    Returns:
        tuple[draw.Group, dict[int, dict], list[tuple[float, float]]]:
            - [0] is the draw.Group with the geometry
            - [1] is a connection point dictionary
            - [2] is a list of grid points to mark as element
    """
    obj_group = draw.Group()

    # winding 1 at top
    # winding 2 at bottom
    # winding 3 is in the middle, then offset right
    # total height is 4*grid_step, with width of 2*grid_step

    radius = (2 * grid_step) / 3
    symbol_top_y = obj_y + grid_step
    symbol_center_y = symbol_top_y + grid_step

    circle1_x = circle2_x = obj_x
    circle1_y = symbol_center_y - radius / 2
    circle2_y = symbol_center_y + radius / 2
    circle3_x = obj_x + radius
    circle3_y = symbol_center_y

    # unpack colours
    w1colour = colour_map.get(winding_voltages[0], "black")
    w2colour = colour_map.get(winding_voltages[1], "black")
    w3colour = colour_map.get(winding_voltages[2], "black")

    # --- Top Terminal Line ---
    top_line_start_y = obj_y
    top_line_end_y = symbol_top_y
    top_line = draw.Line(
        obj_x,
        top_line_start_y,
        obj_x,
        top_line_end_y,
        stroke=w1colour,
        stroke_width=2,
    )
    obj_group.append(top_line)

    # --- Top Circle ---
    top_circle = draw.Circle(
        circle1_x,
        circle1_y,
        radius,
        fill="transparent",
        stroke=w1colour,
        stroke_width=3,
    )
    obj_group.append(top_circle)

    # --- Bottom Circle ---
    bottom_circle = draw.Circle(
        circle2_x,
        circle2_y,
        radius,
        fill="transparent",
        stroke=w2colour,
        stroke_width=3,
    )
    obj_group.append(bottom_circle)

    # --- Bottom Terminal Line ---
    bottom_line_start_y = symbol_top_y + 2 * grid_step
    bottom_line_end_y = bottom_line_start_y + grid_step
    bottom_line = draw.Line(
        obj_x,
        bottom_line_start_y,
        obj_x,
        bottom_line_end_y,
        stroke=w2colour,
        stroke_width=2,
    )
    obj_group.append(bottom_line)

    # tertiary winding circle
    tertiary_circle = draw.Circle(
        circle3_x,
        circle3_y,
        radius,
        fill="transparent",
        stroke=w3colour,
        stroke_width=3,
    )
    obj_group.append(tertiary_circle)

    # draw tertiary line
    tert_line_start_x = circle3_x + radius
    tert_line_end_x = circle3_x + radius + grid_step
    tertiary_line = draw.Line(
        tert_line_start_x,
        circle3_y,
        tert_line_end_x,
        circle3_y,
        stroke=w3colour,
        stroke_width=2,
    )
    obj_group.append(tertiary_line)

    # Mark grid points for the transformer elements
    grid_points_to_mark = [
        (obj_x, top_line_start_y),
        (obj_x, top_line_end_y),
        (obj_x, symbol_center_y),
        (obj_x, bottom_line_start_y),
        (obj_x, bottom_line_end_y),
    ]

    # terminal dict
    terminals = {
        1: {
            "coords": (obj_x, top_line_start_y),  # top terminal
            "voltage": winding_voltages[0],
        },
        2: {
            "coords": (obj_x, bottom_line_end_y),  # bottom terminal
            "voltage": winding_voltages[1],
        },
        3: {
            "coords": (tert_line_end_x, circle3_y),  # tertiary terminal
            "voltage": winding_voltages[2],
        },
    }

    # return the drawing group and the grid points to mark
    return obj_group, terminals, grid_points_to_mark
