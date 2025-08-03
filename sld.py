# CB notes
"""
general notes
- for flexibility, each bay should allow each cb to be replaced with an isol or a solid connection
- there might also be benefit in requiring all bay types to match the sub, e.g. a double switched sub should only have double switched bays BUT you could specify that one of the bays has no connection on one side, making it a single switched bay?
- only 66 kV and higher, no 22kv or 33kv
- dont show isols if CB presnet, dont show double isol if there is meant to be a CB, but no CB (e.g. GNTS)


single switched arrangements:
    - need to support bus-ties on multi planar bus arrangements (Ballarat NTH). same sub, we also need to consider how the diagram will be oriented (there are CBs on both side of the single switched bus here - for the diagram)
    - need to support subs which dont have buses, e.g. COBDEN (---[]-<>-[]--- with tx at <> for load)

double switched subs:
    - need to support single switched lines off of double switched buses (e.g. GTS)
    -

breaker and a half subs:
    - need to support double switched and single switched bays e.g. MLTS - make sure bus spacing is correct

special cases:
    - need to support ring bus subs, rings arnt always 4, e.g. CAMPERDOWN is 4 cb 1 isol ring, COLAC is 3c ring
    - need to support 3-bus double switched arrangements (e.g. TTS, though i cant see any direct bus-ties, so planar bus arrangement might be possible)
    - need to support unused bays, e.g. SMTS bay next to F2 tx on 500 kV side


"""


# one substation has many buses
# one substation has many bays, which reference one or two buses

# Standard library imports
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations

# Third-party imports
import drawsvg as draw
import networkx as nx
import numpy as np
import utm
import yaml

# Local application/library specific imports
import findpath

# --- Constants ---
MAP_DIMS = 7000
BUS_LABEL_FONT_SIZE = 15
TITLE_MAX_SEARCH_RADIUS_PX = 300
TITLE_FONT_SIZE = 40
BUSBAR_WEIGHT = 8
LINE_CROSS_WEIGHT = 15
NEAR_SUB_WEIGHT = 4
ELEMENT_WEIGHT = 50
GRID_STEP = 25
SUBSTATIONS_DATA_FILE = r"C:\Users\DamienVermeer\Downloads\substation_definitions.yaml"
TEMPLATE_FILE = r"C:\Users\DamienVermeer\Downloads\index.template.html"
OUTPUT_SVG = "example.svg"
OUTPUT_HTML = "index.html"

# below colours from AEMO NEM SLD pdf for consistency
COLOUR_MAP = {
    22: "#4682B4",
    33: "#006400",
    66: "#A0522D",
    110: "#FF0000",
    132: "#FF0000",
    220: "#0000FF",
    275: "#FF00FF",
    330: "#FF8C00",
    500: "#FFDC00",
}


# --- Enums and Dataclasses ---
@dataclass
class DrawingParams:
    grid_step: int = GRID_STEP
    bay_width: int = 50
    cb_size: int = 25
    isolator_size: int = 25


class SwitchType(Enum):
    EMPTY = auto()
    DIRECT = auto()
    CB = auto()
    ISOL = auto()
    UNKNOWN = auto()


@dataclass
class Substation:
    name: str
    lat: float
    long: float
    voltage_kv: int
    tags: list[str] = field(default_factory=list)
    rotation: int = 0
    definition: str = ""
    buses: dict = field(default_factory=dict)
    connections: dict = field(default_factory=dict)
    scaled_x: float = 0
    scaled_y: float = 0
    x: float = 0.0
    y: float = 0.0
    title: str = ""
    use_x: float = 0.0  # Final drawing coordinate
    use_y: float = 0.0  # Final drawing coordinate

    def __post_init__(self):
        self.grid_points = {}  # Store (x,y) -> weight dictionary for grid points
        self.connection_points: dict[str, tuple[float, float]] = {}
        self.objects = []  # Store objects associated with this substation

    def draw_objects(
        self, parent_group: draw.Group, params: DrawingParams = DrawingParams()
    ) -> draw.Group:
        """Draw all objects associated with the substation."""
        import math  # Import at the top for rotation calculations

        conn_points = {}  # Initialize conn_points here
        for obj in self.objects:
            obj_x = obj["rel_x"] * params.grid_step
            obj_y = obj["rel_y"] * params.grid_step
            rotation = obj.get("rotation", 0)

            # Create a group for this object
            obj_group = draw.Group()

            if obj["type"] == "tx":
                # Amended code to draw the IEC 60617 transformer symbol (interlocking circles)
                # Assumes (obj_x, obj_y) is the desired center of the entire symbol object.
                # Assumes a Y-down coordinate system.

                # Get colours from metadata
                w1_voltage = obj.get("metadata", {}).get("w1")
                w2_voltage = obj.get("metadata", {}).get("w2")
                colour1 = COLOUR_MAP.get(w1_voltage, "black")
                colour2 = COLOUR_MAP.get(w2_voltage, "black")

                # --- Define Geometry ---
                # To make the symbol fit nicely within a grid cell, the radius should be
                # smaller than half the grid step. A third is a good proportion.
                radius = 2 * params.grid_step / 3

                # The vertical distance between the center of the whole object (obj_y)
                # and the center of each circle is half the radius.
                offset = radius / 2

                # --- Top Circle ---
                circle1_x = obj_x
                circle1_y = obj_y - offset
                top_circle = draw.Circle(
                    circle1_x,
                    circle1_y,
                    radius,
                    fill="transparent",
                    stroke=colour1,
                    stroke_width=3,
                )
                obj_group.append(top_circle)

                # --- Bottom Circle ---
                circle2_x = obj_x
                circle2_y = obj_y + offset
                bottom_circle = draw.Circle(
                    circle2_x,
                    circle2_y,
                    radius,
                    fill="transparent",
                    stroke=colour2,
                    stroke_width=3,
                )
                obj_group.append(bottom_circle)

                # --- Terminal Lines ---
                # Define a length for the terminal lines - 50% longer as requested
                line_length = params.grid_step

                # Top terminal line
                top_line_start_y = circle1_y - radius  # Top point of the top circle
                top_line_end_y = top_line_start_y - line_length
                top_line = draw.Line(
                    obj_x,
                    top_line_start_y,
                    obj_x,
                    top_line_end_y,
                    stroke=colour1,
                    stroke_width=2,
                )
                obj_group.append(top_line)

                # Bottom terminal line
                bottom_line_start_y = (
                    circle2_y + radius
                )  # Bottom point of the bottom circle
                bottom_line_end_y = bottom_line_start_y + line_length
                bottom_line = draw.Line(
                    obj_x,
                    bottom_line_start_y,
                    obj_x,
                    bottom_line_end_y,
                    stroke=colour2,
                    stroke_width=2,
                )
                obj_group.append(bottom_line)

                # Winding text removed as requested

                # Store connection points (we'll apply rotation later if needed)
                conn_points = {}
                if "connections" in obj:
                    for i, conn_id in enumerate(obj["connections"]):
                        if i == 0:  # First connection at far end of top terminal line
                            conn_points[conn_id] = (
                                circle1_x,
                                top_line_end_y,
                            )  # At the far end away from transformer
                        elif (
                            i == 1
                        ):  # Second connection at far end of bottom terminal line
                            conn_points[conn_id] = (
                                circle2_x,
                                bottom_line_end_y,
                            )  # At the far end away from transformer
                        # Additional connections would be handled here

                    # Mark the connection points for pathfinding
                    mark_grid_point(
                        self, circle1_x, top_line_end_y, weight=0
                    )  # Connection point has weight 0 for pathfinding
                    mark_grid_point(
                        self, circle2_x, bottom_line_end_y, weight=0
                    )  # Connection point has weight 0 for pathfinding

                # Mark grid points for the transformer body
                mark_grid_point(self, circle1_x, circle1_y)
                mark_grid_point(self, circle2_x, circle2_y)

            elif obj["type"] == "gen":
                # Generator is a "lollipop" - a circle on a stick
                radius = 2 * params.grid_step / 3
                voltage = obj.get("metadata", {}).get("voltage")
                colour = COLOUR_MAP.get(voltage, "black")
                text = obj.get("metadata", {}).get("text", "G")

                # The reference point (obj_x, obj_y) is the bottom of the stick.
                line_start_y = obj_y
                line_end_y = obj_y - params.grid_step
                circle_center_x = obj_x
                # The circle sits on top of the stick, so its center is radius-distance above the stick's end
                circle_center_y = line_end_y - radius

                # Draw the stick (line)
                obj_group.append(
                    draw.Line(
                        obj_x,
                        line_start_y,
                        obj_x,
                        line_end_y,
                        stroke=colour,
                        stroke_width=2,
                    )
                )

                # Draw circle
                obj_group.append(
                    draw.Circle(
                        circle_center_x,
                        circle_center_y,
                        radius,
                        fill="transparent",
                        stroke=colour,
                        stroke_width=2,
                    )
                )

                # Draw text inside circle
                obj_group.append(
                    draw.Text(
                        text,
                        font_size=params.grid_step * 0.7,
                        x=circle_center_x,
                        y=circle_center_y,
                        text_anchor="middle",
                        dominant_baseline="central",
                        fill=colour,
                        stroke_width=0,
                    )
                )
                # Mark grid points for the line and the circle center
                mark_grid_point(self, obj_x, obj_y)  # bottom of stick
                mark_grid_point(self, obj_x, line_end_y)  # top of stick
                mark_grid_point(
                    self, circle_center_x, circle_center_y
                )  # circle center

            # Apply rotation if specified
            if rotation != 0:
                # Create a container group with rotation transform
                rotated_group = draw.Group(
                    transform=f"rotate({rotation}, {obj_x}, {obj_y})"
                )
                # Add the entire object group to the rotated group
                rotated_group.append(obj_group)
                obj_group = rotated_group

                # Apply the same rotation to the connection points
                if "connections" in obj:
                    rotation_rad = math.radians(rotation)
                    for conn_id, (px, py) in conn_points.items():
                        # Translate to origin
                        tx = px - obj_x
                        ty = py - obj_y
                        # Rotate
                        rx = tx * math.cos(rotation_rad) - ty * math.sin(rotation_rad)
                        ry = tx * math.sin(rotation_rad) + ty * math.cos(rotation_rad)
                        # Translate back
                        rotated_x = rx + obj_x
                        rotated_y = ry + obj_y
                        # Update the connection point with rotated coordinates
                        self.connection_points[conn_id] = (rotated_x, rotated_y)

                        # Also update the grid points for pathfinding with rotated coordinates
                        mark_grid_point(
                            self, rotated_x, rotated_y, weight=0
                        )  # Connection point has weight 0 for pathfinding
            else:
                # For non-rotated objects, store the connection points directly
                if "connections" in obj:
                    for conn_id, (px, py) in conn_points.items():
                        # Store connection points for non-rotated objects
                        self.connection_points[conn_id] = (px, py)

            parent_group.append(obj_group)

        return parent_group

    def get_drawing_bbox(
        self, params: DrawingParams
    ) -> tuple[float, float, float, float]:
        """Calculates the bounding box (min_x, min_y, max_x, max_y) of the substation drawing."""
        if not self.definition and not self.objects:
            return 0, 0, 0, 0

        bay_defs = self.definition.strip().split("\n")
        num_bays = len(bay_defs)
        max_elements = 0
        for bay_def in bay_defs:
            # A simple way to estimate height is by counting elements.
            # This will need to be more robust.
            max_elements = max(max_elements, len(bay_def))

        min_x = -params.grid_step
        max_x = (num_bays - 1) * 2 * params.grid_step + params.grid_step
        min_y = 0  # Assuming top bus is at y=0
        # Estimate height based on number of elements * space per element
        max_y = max_elements * (params.cb_size + params.grid_step)

        # Include objects in the bounding box calculation
        for obj in self.objects:
            # Objects have rel_x and rel_y which are in grid steps from origin
            obj_x = obj["rel_x"] * params.grid_step
            obj_y = obj["rel_y"] * params.grid_step

            # For TX objects, consider the two circles
            if obj["type"] == "tx":
                # Each circle is 25px (1 grid step) in diameter
                # Circles are placed side by side
                obj_min_x = obj_x - params.grid_step
                obj_max_x = obj_x + params.grid_step * 2
                obj_min_y = obj_y - params.grid_step / 2
                obj_max_y = obj_y + params.grid_step * 1.5
            elif obj["type"] == "gen":
                radius = 2 * params.grid_step / 3
                # The reference point (obj_x, obj_y) is the bottom of the stick.
                # The lollipop is drawn upwards from there.
                obj_min_x = obj_x - radius
                obj_max_x = obj_x + radius
                obj_max_y = obj_y
                # Top of lollipop is top of circle: (line_end_y - radius) - radius
                obj_min_y = (obj_y - params.grid_step) - (2 * radius)
            else:  # Generic object handling for other types
                obj_min_x = obj_x - params.grid_step / 2
                obj_max_x = obj_x + params.grid_step / 2
                obj_min_y = obj_y - params.grid_step / 2
                obj_max_y = obj_y + params.grid_step / 2

            # Update the bounding box
            min_x = min(min_x, obj_min_x)
            max_x = max(max_x, obj_max_x)
            min_y = min(min_y, obj_min_y)
            max_y = max(max_y, obj_max_y)

        return min_x, min_y, max_x, max_y


# --- Data Loading ---
def load_substations_from_yaml(filename: str) -> dict[str, Substation]:
    """Load substations from YAML file into a dictionary."""
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    substations_map = {}

    for sub_data in data["substations"]:
        # Create substation
        substation = Substation(
            name=sub_data["name"],
            title=sub_data.get("title", sub_data["name"]),
            lat=sub_data["lat"],
            long=sub_data["long"],
            voltage_kv=sub_data["voltage_kv"],
            tags=sub_data.get("tags", [sub_data["name"]]),
            rotation=sub_data.get("rotation", 0),
            definition=sub_data.get("def", ""),
            buses=sub_data.get("buses", {}),
            connections=sub_data.get("connections", {}),
        )

        # Add objects to the substation if present in the data
        if "objects" in sub_data:
            substation.objects = sub_data["objects"]

        substations_map[substation.name] = substation
    return substations_map


# --- Drawing Helpers ---
def mark_grid_point(sub: "Substation", x: float, y: float, weight: int = 25) -> None:
    """Mark a grid point in the substation's grid_points dictionary with a weight."""
    sub.grid_points[(x, y)] = weight


def draw_switch(
    x: float,
    y: float,
    parent_group: draw.Group,
    switch_type: SwitchType,
    orientation: str = "vertical",
    rotation_angle: int = 45,
    params: DrawingParams = DrawingParams(),
    colour: str = "black",
) -> draw.Group:
    """Generic function to draw a switch (CB or isolator) at given coordinates."""
    if switch_type == SwitchType.CB:
        # Circuit breaker is drawn as a rectangle
        parent_group.append(
            draw.Rectangle(
                x - params.cb_size / 2,
                y - params.cb_size / 2,
                params.cb_size,
                params.cb_size,
                fill="white",
                stroke=colour,
            )
        )
    elif switch_type == SwitchType.UNKNOWN:
        # Unknown switch type is drawn as a question mark
        parent_group.append(
            draw.Text(
                "?",
                font_size=params.cb_size * 1,
                x=x,
                y=y,
                text_anchor="middle",
                dominant_baseline="central",
                fill="black",
                stroke_width=0,
            )
        )
    elif switch_type == SwitchType.ISOL:
        # Isolator is drawn as a rotated line
        if orientation == "vertical":
            parent_group.append(
                draw.Line(
                    x,
                    y - params.isolator_size / 2,
                    x,
                    y + params.isolator_size / 2,
                    stroke=colour,
                    stroke_width=2,
                    transform=f"rotate({rotation_angle}, {x}, {y})",
                )
            )
        else:  # horizontal
            parent_group.append(
                draw.Line(
                    x - params.isolator_size / 2,
                    y,
                    x + params.isolator_size / 2,
                    y,
                    stroke=colour,
                    stroke_width=2,
                    transform=f"rotate({-rotation_angle}, {x}, {y})",
                )
            )
    return parent_group


# --- Bay Drawing Functions ---
def draw_bay_from_string(
    xoff: float,
    parent_group: draw.Group,
    bay_def: str,
    sub: Substation,
    is_first_bay: bool,
    params: DrawingParams = DrawingParams(),
    previous_bay_elements: list = None,
    y_offset: int = 0,
) -> draw.Group:
    """
    Draw a single bay based on a definition string using the new substation language.
    """
    colour = COLOUR_MAP.get(sub.voltage_kv, "black")

    # Parse the definition string into elements
    elements = []
    char_index = 0

    while char_index < len(bay_def):
        char = bay_def[char_index]

        # Handle busbar objects
        if char == "|":
            # Count consecutive | characters for busbar ID
            bus_start_index = char_index
            while char_index < len(bay_def) and bay_def[char_index] == "|":
                char_index += 1
            bus_id = char_index - bus_start_index
            elements.append({"type": "busbar", "subtype": "standard", "id": bus_id})

        elif char == "s":
            elements.append({"type": "busbar", "subtype": "string"})
            char_index += 1

        elif char == "N":
            elements.append({"type": "busbar", "subtype": "null"})
            char_index += 1

        elif char == "t":
            # Check for 'ts' variant
            if char_index + 1 < len(bay_def) and bay_def[char_index + 1] == "s":
                elements.append({"type": "busbar", "subtype": "tie_cb_thin"})
                char_index += 2
            else:
                elements.append({"type": "busbar", "subtype": "tie_cb"})
                char_index += 1

        elif char == "i":
            # Check for 'is' variant
            if char_index + 1 < len(bay_def) and bay_def[char_index + 1] == "s":
                elements.append({"type": "busbar", "subtype": "tie_isol_thin"})
                char_index += 2
            else:
                elements.append({"type": "busbar", "subtype": "tie_isol"})
                char_index += 1

        # Handle element objects
        elif char == "x":
            elements.append({"type": "element", "subtype": "cb"})
            char_index += 1

        elif char == "/":
            elements.append({"type": "element", "subtype": "isolator"})
            char_index += 1

        elif char == "d":
            elements.append({"type": "element", "subtype": "direct"})
            char_index += 1

        elif char == "E":
            elements.append({"type": "element", "subtype": "empty"})
            char_index += 1

        # Handle connection objects
        elif char.isdigit():
            num_start_index = char_index
            while char_index < len(bay_def) and bay_def[char_index].isdigit():
                char_index += 1
            conn_id = int(bay_def[num_start_index:char_index])
            elements.append({"type": "connection", "id": conn_id})

        else:
            # Warn about unrecognised characters
            print(
                f"WARNING: Unrecognised character '{char}' at position {char_index} in bay definition '{bay_def}' for substation '{sub.name}'"
            )
            char_index += 1

    y_pos = -y_offset

    # Draw elements with proper connecting lines
    last_y = y_pos

    for i, element in enumerate(elements):
        if element["type"] == "busbar":
            # Draw connecting line from previous element if needed
            if i > 0 and last_y != y_pos:
                parent_group.append(
                    draw.Line(xoff, last_y, xoff, y_pos, stroke=colour, stroke_width=2)
                )
                # Mark intermediate grid points
                steps = int((y_pos - last_y) / params.grid_step)
                for step in range(1, steps):
                    mark_grid_point(
                        sub,
                        xoff,
                        last_y + step * params.grid_step,
                        weight=ELEMENT_WEIGHT,
                    )

            y_pos = draw_busbar_object(
                element,
                xoff,
                y_pos,
                parent_group,
                sub,
                is_first_bay,
                params,
                colour,
                previous_bay_elements,
            )
            last_y = y_pos

        elif element["type"] == "element":
            # Draw connecting line from previous element if needed
            if i > 0 and last_y != y_pos:
                parent_group.append(
                    draw.Line(xoff, last_y, xoff, y_pos, stroke=colour, stroke_width=2)
                )
                # Mark intermediate grid points
                steps = int((y_pos - last_y) / params.grid_step)
                for step in range(1, steps):
                    mark_grid_point(
                        sub,
                        xoff,
                        last_y + step * params.grid_step,
                        weight=ELEMENT_WEIGHT,
                    )

            y_pos = draw_element_object(
                element, xoff, y_pos, parent_group, sub, params, colour
            )
            last_y = y_pos

            # Add grid step spacing after each element only if there's another non-connection element following
            next_element_idx = i + 1
            while (
                next_element_idx < len(elements)
                and elements[next_element_idx]["type"] == "connection"
            ):
                next_element_idx += 1

        elif element["type"] == "connection":
            draw_connection_object(element, xoff, y_pos, parent_group, sub, colour)

    return parent_group


def draw_busbar_object(
    element,
    xoff,
    y_pos,
    parent_group,
    sub,
    is_first_bay,
    params,
    colour,
    previous_bay_elements=None,
):
    """Draw a busbar object at the specified position."""
    subtype = element["subtype"]

    # Check if previous bay has a busbar at the same y position for continuity
    extend_left = False
    if previous_bay_elements and not is_first_bay:
        # Find if there's a busbar at the same relative position in the previous bay
        current_busbar_index = 0
        for prev_element in previous_bay_elements:
            if (
                prev_element["type"] == "busbar"
                and prev_element["subtype"] == "standard"
            ):
                if current_busbar_index == 0:  # This is the matching busbar position
                    extend_left = True
                    break
                current_busbar_index += 1

    if subtype == "standard":
        # Determine line start position
        line_start_x = xoff - (
            2 * params.grid_step if extend_left else params.grid_step
        )

        # Draw thick horizontal line spanning 3*GRID_STEP (or 4*GRID_STEP if extending)
        parent_group.append(
            draw.Line(
                line_start_x,
                y_pos,
                xoff + params.grid_step,
                y_pos,
                stroke=colour,
                stroke_width=5,
            )
        )
        # Mark grid points with BUSBAR_WEIGHT
        if extend_left:
            mark_grid_point(
                sub, xoff - 2 * params.grid_step, y_pos, weight=BUSBAR_WEIGHT
            )
        mark_grid_point(sub, xoff - params.grid_step, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff + params.grid_step, y_pos, weight=BUSBAR_WEIGHT)

        # Add text label if first bay
        if is_first_bay:
            bus_id = element["id"]
            bus_name = sub.buses.get(bus_id, f"Bus {bus_id}")
            parent_group.append(
                draw.Text(
                    bus_name,
                    x=xoff - params.grid_step - 5,
                    y=y_pos - 8,
                    font_size=BUS_LABEL_FONT_SIZE,
                    text_anchor="end",
                    stroke_width=0,
                )
            )

    elif subtype == "string":
        # Determine line start position
        line_start_x = xoff - (
            2 * params.grid_step if extend_left else params.grid_step
        )

        # Draw normal thickness horizontal line spanning 3*GRID_STEP (or 4*GRID_STEP if extending)
        parent_group.append(
            draw.Line(
                line_start_x,
                y_pos,
                xoff + params.grid_step,
                y_pos,
                stroke=colour,
                stroke_width=2,
            )
        )
        # Mark grid points with BUSBAR_WEIGHT
        if extend_left:
            mark_grid_point(
                sub, xoff - 2 * params.grid_step, y_pos, weight=BUSBAR_WEIGHT
            )
        mark_grid_point(sub, xoff - params.grid_step, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff + params.grid_step, y_pos, weight=BUSBAR_WEIGHT)

    elif subtype == "null":
        # No line drawn, but mark grid points spanning 3*GRID_STEP (or 4*GRID_STEP if extending)
        if extend_left:
            mark_grid_point(
                sub, xoff - 2 * params.grid_step, y_pos, weight=BUSBAR_WEIGHT
            )
        mark_grid_point(sub, xoff - params.grid_step, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff, y_pos, weight=BUSBAR_WEIGHT)
        mark_grid_point(sub, xoff + params.grid_step, y_pos, weight=BUSBAR_WEIGHT)

    elif subtype in ["tie_cb", "tie_cb_thin"]:
        # Draw busbar with circuit breaker tie
        line_width = 5 if subtype == "tie_cb" else 2

        # Determine left line start position
        left_line_start_x = xoff - (
            2 * params.grid_step if extend_left else params.grid_step
        )

        # Left line segment (extended if needed)
        parent_group.append(
            draw.Line(
                left_line_start_x,
                y_pos,
                xoff - params.grid_step / 2,
                y_pos,
                stroke=colour,
                stroke_width=line_width,
            )
        )

        # Circuit breaker square (25x25)
        parent_group.append(
            draw.Rectangle(
                xoff - params.grid_step / 2,
                y_pos - params.grid_step / 2,
                params.grid_step,
                params.grid_step,
                fill="white",
                stroke=colour,
            )
        )

        # Right line segment
        parent_group.append(
            draw.Line(
                xoff + params.grid_step / 2,
                y_pos,
                xoff + params.grid_step,
                y_pos,
                stroke=colour,
                stroke_width=line_width,
            )
        )

        # Mark grid points with ELEMENT_WEIGHT
        if extend_left:
            mark_grid_point(
                sub, xoff - 2 * params.grid_step, y_pos, weight=ELEMENT_WEIGHT
            )
        mark_grid_point(sub, xoff - params.grid_step, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff + params.grid_step, y_pos, weight=ELEMENT_WEIGHT)

    elif subtype in ["tie_isol", "tie_isol_thin"]:
        # Draw busbar with isolator tie
        line_width = 5 if subtype == "tie_isol" else 2

        # Determine left line start position
        left_line_start_x = xoff - (
            2 * params.grid_step if extend_left else params.grid_step
        )

        # Left line segment (extended if needed)
        parent_group.append(
            draw.Line(
                left_line_start_x,
                y_pos,
                xoff - params.grid_step / 2,
                y_pos,
                stroke=colour,
                stroke_width=line_width,
            )
        )

        # 45-degree isolator line (25px wide)
        isolator_half_size = params.grid_step / 2
        parent_group.append(
            draw.Line(
                xoff - isolator_half_size,
                y_pos - isolator_half_size,
                xoff + isolator_half_size,
                y_pos + isolator_half_size,
                stroke=colour,
                stroke_width=2,
            )
        )

        # Right line segment
        parent_group.append(
            draw.Line(
                xoff + params.grid_step / 2,
                y_pos,
                xoff + params.grid_step,
                y_pos,
                stroke=colour,
                stroke_width=line_width,
            )
        )

        # Mark grid points with ELEMENT_WEIGHT
        if extend_left:
            mark_grid_point(
                sub, xoff - 2 * params.grid_step, y_pos, weight=ELEMENT_WEIGHT
            )
        mark_grid_point(sub, xoff - params.grid_step, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff + params.grid_step, y_pos, weight=ELEMENT_WEIGHT)

    return y_pos


def draw_element_object(element, xoff, y_pos, parent_group, sub, params, colour):
    """Draw an element object at the specified position."""
    subtype = element["subtype"]

    if subtype == "cb":
        # Circuit breaker: 25px line + square + 25px line
        # First vertical line (exactly 25px)
        parent_group.append(
            draw.Line(
                xoff,
                y_pos,
                xoff,
                y_pos + params.grid_step,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos + params.grid_step, weight=ELEMENT_WEIGHT)

        # Square (full grid step size, centered in middle grid step)
        square_center_y = y_pos + params.grid_step + (params.grid_step // 2)
        parent_group.append(
            draw.Rectangle(
                xoff - params.grid_step // 2,
                square_center_y - params.grid_step // 2,
                params.grid_step,
                params.grid_step,
                fill="white",
                stroke=colour,
            )
        )
        mark_grid_point(sub, xoff, y_pos + 2 * params.grid_step, weight=ELEMENT_WEIGHT)

        # Second vertical line (exactly 25px)
        parent_group.append(
            draw.Line(
                xoff,
                y_pos + 2 * params.grid_step,
                xoff,
                y_pos + 3 * params.grid_step,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos + 3 * params.grid_step, weight=ELEMENT_WEIGHT)

    elif subtype == "isolator":
        # Isolator: 25px line + 45° line + 25px line
        # First vertical line (exactly 25px)
        parent_group.append(
            draw.Line(
                xoff,
                y_pos,
                xoff,
                y_pos + params.grid_step,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos + params.grid_step, weight=ELEMENT_WEIGHT)

        # 45-degree line (centered in middle grid step)
        isolator_center_y = y_pos + params.grid_step + (params.grid_step // 2)
        isolator_half_size = params.grid_step // 2
        parent_group.append(
            draw.Line(
                xoff - isolator_half_size,
                isolator_center_y - isolator_half_size,
                xoff + isolator_half_size,
                isolator_center_y + isolator_half_size,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos + 2 * params.grid_step, weight=ELEMENT_WEIGHT)

        # Second vertical line (exactly 25px)
        parent_group.append(
            draw.Line(
                xoff,
                y_pos + 2 * params.grid_step,
                xoff,
                y_pos + 3 * params.grid_step,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos + 3 * params.grid_step, weight=ELEMENT_WEIGHT)

    elif subtype == "direct":
        # Direct connection: single vertical line spanning 3*GRID_STEP
        parent_group.append(
            draw.Line(
                xoff,
                y_pos,
                xoff,
                y_pos + 3 * params.grid_step,
                stroke=colour,
                stroke_width=2,
            )
        )
        mark_grid_point(sub, xoff, y_pos, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos + params.grid_step, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos + 2 * params.grid_step, weight=ELEMENT_WEIGHT)
        mark_grid_point(sub, xoff, y_pos + 3 * params.grid_step, weight=ELEMENT_WEIGHT)

    elif subtype == "empty":
        # Empty element: no drawing, no grid marking, but advance position
        pass

    return y_pos + 3 * params.grid_step


def parse_bay_elements(bay_def: str) -> list:
    """Parse a bay definition string into elements for continuity checking."""
    elements = []
    char_index = 0

    while char_index < len(bay_def):
        char = bay_def[char_index]

        # Handle busbar objects
        if char == "|":
            # Count consecutive | characters for busbar ID
            bus_start_index = char_index
            while char_index < len(bay_def) and bay_def[char_index] == "|":
                char_index += 1
            bus_id = char_index - bus_start_index
            elements.append({"type": "busbar", "subtype": "standard", "id": bus_id})

        elif char == "s":
            elements.append({"type": "busbar", "subtype": "string"})
            char_index += 1

        elif char == "N":
            elements.append({"type": "busbar", "subtype": "null"})
            char_index += 1

        elif char == "t":
            # Check for 'ts' variant
            if char_index + 1 < len(bay_def) and bay_def[char_index + 1] == "s":
                elements.append({"type": "busbar", "subtype": "tie_cb_thin"})
                char_index += 2
            else:
                elements.append({"type": "busbar", "subtype": "tie_cb"})
                char_index += 1

        elif char == "i":
            # Check for 'is' variant
            if char_index + 1 < len(bay_def) and bay_def[char_index + 1] == "s":
                elements.append({"type": "busbar", "subtype": "tie_isol_thin"})
                char_index += 2
            else:
                elements.append({"type": "busbar", "subtype": "tie_isol"})
                char_index += 1

        # Handle element objects
        elif char == "x":
            elements.append({"type": "element", "subtype": "cb"})
            char_index += 1

        elif char == "/":
            elements.append({"type": "element", "subtype": "isolator"})
            char_index += 1

        elif char == "d":
            elements.append({"type": "element", "subtype": "direct"})
            char_index += 1

        elif char == "E":
            elements.append({"type": "element", "subtype": "empty"})
            char_index += 1

        # Handle connection objects
        elif char.isdigit():
            num_start_index = char_index
            while char_index < len(bay_def) and bay_def[char_index].isdigit():
                char_index += 1
            conn_id = int(bay_def[num_start_index:char_index])
            elements.append({"type": "connection", "id": conn_id})

        else:
            # Warn about unrecognised characters (but don't include substation name as it's not available here)
            print(
                f"WARNING: Unrecognised character '{char}' at position {char_index} in bay definition '{bay_def}'"
            )
            char_index += 1

    return elements


def draw_connection_object(element, xoff, y_pos, parent_group, sub, colour):
    """Draw a connection object at the specified position."""
    conn_id = element["id"]
    connection_name = sub.connections.get(conn_id)

    if connection_name:
        # Store the connection point for pathfinding
        sub.connection_points[connection_name] = (xoff, y_pos)
        mark_grid_point(sub, xoff, y_pos, weight=0)  # Connection points have weight 0
    else:
        # Draw an unused connection circle
        parent_group.append(draw.Circle(xoff, y_pos, 4, fill=colour, stroke="none"))


def get_substation_group(sub: Substation, params: DrawingParams, rotation=0):
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    dg = draw.Group(
        stroke_width=2,
        transform=f"rotate({rotation}, {center_x}, {center_y})",
    )

    bay_defs = sub.definition.strip().split("\n")

    # Pre-calculate the maximum y-offset needed for any bay to align all busbars
    max_y_offset = 0
    parsed_bays = [parse_bay_elements(bay_def) for bay_def in bay_defs]
    for elements in parsed_bays:
        y_offset = 0
        first_busbar_idx = next(
            (i for i, el in enumerate(elements) if el["type"] == "busbar"), -1
        )
        if first_busbar_idx > 0:
            # Count elements before the first busbar
            for i in range(first_busbar_idx):
                if elements[i]["type"] == "element":
                    y_offset += 3 * params.grid_step
            # Add one grid step for the connecting line to the first busbar
            y_offset += params.grid_step
        max_y_offset = max(max_y_offset, y_offset)

    previous_bay_elements = None
    for i, bay_def in enumerate(bay_defs):
        xoff = 2 * params.grid_step * i  # Use 2*GRID_STEP spacing between bays (50px)
        is_first_bay = i == 0

        # check if the first element is a busbar type, if so, pass an offset
        # ... of one grid step
        if parsed_bays[i][0]["type"] == "busbar":
            y_offset = params.grid_step  # not sure why though?
        else:
            y_offset = max_y_offset

        # Parse previous bay elements for continuity checking
        if i > 0:
            previous_bay_elements = parsed_bays[i - 1]

        dg = draw_bay_from_string(
            xoff,
            dg,
            bay_def,
            sub,
            is_first_bay,
            params,
            previous_bay_elements,
            y_offset=y_offset,
        )

    # Draw objects after bays
    if sub.objects:
        dg = sub.draw_objects(parent_group=dg, params=params)

    return dg


# --- Layout and Positioning Functions ---
def calculate_initial_scaled_positions(substations: list[Substation]):
    """Converts lat/lon to UTM and scales them to fit the map."""
    if not substations:
        return

    # Convert all substation lat/longs to UTM at once to ensure they are in the same zone
    lats = np.array([sub.lat for sub in substations])
    longs = np.array([sub.long for sub in substations])
    eastings, northings, _, _ = utm.from_latlon(lats, longs)

    min_east = np.min(eastings)
    min_north = np.min(northings)

    for i, sub in enumerate(substations):
        sub.x = eastings[i] - min_east
        sub.y = northings[i] - min_north

    # Find min/max values for scaling
    min_x = min(sub.x for sub in substations)
    max_x = max(sub.x for sub in substations)
    min_y = min(sub.y for sub in substations)
    max_y = max(sub.y for sub in substations)

    # Calculate ranges
    x_range = max_x - min_x
    y_range = max_y - min_y

    # Determine scaling factor for both dimensions
    scale_factor_x = MAP_DIMS * 0.9 / x_range if x_range > 0 else 1
    scale_factor_y = MAP_DIMS * 0.9 / y_range if y_range > 0 else 1
    scale_factor = min(scale_factor_x, scale_factor_y)

    # Apply scaling and translation
    print("\nScaled coordinates:")
    for sub in substations:
        sub.scaled_x = (sub.x - min_x) * scale_factor * 4 + (MAP_DIMS * 0.2)
        sub.scaled_y = (sub.y - min_y) * scale_factor * 4 + (MAP_DIMS * 0.2)
        print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")


def apply_spring_layout(substations: list[Substation]) -> dict:
    """Applies a NetworkX spring layout to adjust substation positions."""
    nodes = [sub.name for sub in substations]
    initial_pos = {sub.name: (sub.scaled_x, sub.scaled_y) for sub in substations}

    G = nx.Graph()
    G.add_nodes_from(nodes)

    # Find connections between substations to create edges
    connection_map = {}
    for sub in substations:
        for conn_name in sub.connections.values():
            if conn_name:
                connection_map.setdefault(conn_name, []).append(sub.name)

    # Add edges for substations sharing a connection
    for sub_names in connection_map.values():
        unique_names = list(set(sub_names))
        if len(unique_names) > 1:
            for i in range(len(unique_names)):
                for j in range(i + 1, len(unique_names)):
                    G.add_edge(unique_names[i], unique_names[j])

    if G.number_of_edges() == 0:
        return initial_pos

    # Calculate average distance for setting spring layout parameter `k`
    distances = [
        np.linalg.norm(np.array(initial_pos[u]) - np.array(initial_pos[v]))
        for u, v in G.edges()
    ]
    avg_distance = np.mean(distances) if distances else 0

    if avg_distance > 0:
        print("\nApplying spring layout to adjust substation positions...")
        return nx.spring_layout(
            G, pos=initial_pos, iterations=4, k=avg_distance, seed=0
        )

    return initial_pos


def calculate_final_positions(
    substations: list[Substation], final_pos: dict, params: DrawingParams
):
    """Calculates final drawing coordinates after layout adjustments and snapping."""
    substation_map = {sub.name: sub for sub in substations}

    min_x_pos = min(pos[0] for pos in final_pos.values())
    max_x_pos = max(pos[0] for pos in final_pos.values())
    min_y_pos = min(pos[1] for pos in final_pos.values())
    max_y_pos = max(pos[1] for pos in final_pos.values())

    x_pos_range = max_x_pos - min_x_pos
    y_pos_range = max_y_pos - min_y_pos

    print("\nNormalizing coordinates to MAP_DIMSxMAP_DIMS area")
    for name, coords in final_pos.items():
        sub = substation_map[name]

        norm_x = (coords[0] - min_x_pos) / x_pos_range if x_pos_range > 0 else 0.5
        norm_y = (coords[1] - min_y_pos) / y_pos_range if y_pos_range > 0 else 0.5

        _15 = MAP_DIMS * 0.15
        _90 = MAP_DIMS * 0.9
        raw_x = norm_x * (_90 - _15) + _15
        raw_y = norm_y * (_90 - _15) + _15

        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)

        # Find the first busbar position (y=0 in local coordinates)
        first_busbar_y = 0

        # Calculate where the first busbar should be in global coordinates
        target_busbar_global_y = round(raw_y / params.grid_step) * params.grid_step

        # Calculate the substation's use coordinates to place first busbar at target
        sub.scaled_x = round(raw_x / params.grid_step) * params.grid_step
        sub.scaled_y = target_busbar_global_y - first_busbar_y

    print("\nFinal SVG coordinates after network balancing:")
    for sub in substations:
        print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")

        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)
        width = max_x - min_x
        height = max_y - min_y

        use_x = sub.scaled_x - (min_x + width / 2)
        use_y = (MAP_DIMS - sub.scaled_y) - (min_y + height / 2)
        sub.use_x = use_x
        sub.use_y = use_y

    # Re-snap positions to ensure first busbar aligns to grid
    for sub in substations:
        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)

        # Find the first busbar position (y=0 in local coordinates)
        first_busbar_local_y = 0

        # Calculate current global position of first busbar
        current_busbar_global_y = sub.use_y + first_busbar_local_y

        # Snap to grid
        snapped_busbar_global_y = (
            round(current_busbar_global_y / params.grid_step) * params.grid_step
        )
        snapped_x = round((sub.use_x + min_x) / params.grid_step) * params.grid_step

        # Update use coordinates
        sub.use_x = snapped_x - min_x
        sub.use_y = snapped_busbar_global_y - first_busbar_local_y


# --- Main Drawing Orchestration ---
def populate_pathfinding_grid(
    substations: list[Substation], points: list[list], params: DrawingParams
):
    """Populates the global pathfinding grid with points from all substations."""
    num_steps = len(points)
    step = params.grid_step
    for sub in substations:
        # First, calculate all global grid points for the substation to find its bounding box
        sub_global_grid_points = {}
        min_gx, min_gy = num_steps, num_steps
        max_gx, max_gy = -1, -1

        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        rotation_rad = math.radians(sub.rotation)

        for local_x, local_y in sub.grid_points:
            rel_x = local_x - center_x
            rel_y = local_y - center_y
            rotated_x = rel_x * math.cos(rotation_rad) - rel_y * math.sin(rotation_rad)
            rotated_y = rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)
            rotated_local_x = rotated_x + center_x
            rotated_local_y = rotated_y + center_y

            global_x = sub.use_x + rotated_local_x
            global_y = sub.use_y + rotated_local_y

            grid_x = int(round(global_x / step))
            grid_y = int(round(global_y / step))

            if 0 <= grid_x < num_steps and 0 <= grid_y < num_steps:
                weight = sub.grid_points.get((local_x, local_y))
                sub_global_grid_points[(grid_x, grid_y)] = weight
                min_gx = min(min_gx, grid_x)
                max_gx = max(max_gx, grid_x)
                min_gy = min(min_gy, grid_y)
                max_gy = max(max_gy, grid_y)

        # Apply a penalty in the expanded bounding box of the substation
        if min_gx <= max_gx:  # Check if any points were found
            expanded_min_gx = max(0, min_gx - 2)
            expanded_max_gx = min(num_steps - 1, max_gx + 2)
            expanded_min_gy = max(0, min_gy - 2)
            expanded_max_gy = min(num_steps - 1, max_gy + 2)

            for gx in range(expanded_min_gx, expanded_max_gx + 1):
                for gy in range(expanded_min_gy, expanded_max_gy + 1):
                    if points[gx][gy] == 0:  # Only apply to empty space
                        points[gx][gy] = NEAR_SUB_WEIGHT

        # Place the substation's own points on the grid, overwriting the penalty
        for (grid_x, grid_y), weight in sub_global_grid_points.items():
            points[grid_x][grid_y] = weight


def calculate_connection_points(
    substations: list[Substation], params: DrawingParams
) -> dict:
    """Calculates global coordinates for all connection points."""
    all_connections = {}
    for sub in substations:
        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        rotation_rad = math.radians(sub.rotation)

        for linedef, local_coords in sub.connection_points.items():
            if not linedef:
                continue

            local_x, local_y = local_coords
            rel_x = local_x - center_x
            rel_y = local_y - center_y
            rotated_x = rel_x * math.cos(rotation_rad) - rel_y * math.sin(rotation_rad)
            rotated_y = rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)
            rotated_local_x = rotated_x + center_x
            rotated_local_y = rotated_y + center_y

            global_coords = (sub.use_x + rotated_local_x, sub.use_y + rotated_local_y)
            connection_data = {"coords": global_coords, "voltage": sub.voltage_kv}
            all_connections.setdefault(linedef, []).append(connection_data)
    return all_connections


def draw_connections(
    drawing: draw.Drawing, all_connections: dict, points: list[list], step: int
):
    """Finds paths and draws connections between substations."""
    num_steps = len(points)

    def _distance(a, b) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    valid_connections = {k: v for k, v in all_connections.items() if len(v) == 2}
    sorted_connections = sorted(
        valid_connections.items(),
        key=lambda item: _distance(item[1][0]["coords"], item[1][1]["coords"]),
    )

    for _, connection_points in sorted_connections:
        start_coord_px = connection_points[0]["coords"]
        end_coord_px = connection_points[1]["coords"]

        voltage1 = connection_points[0]["voltage"]
        voltage2 = connection_points[1]["voltage"]
        colour = COLOUR_MAP.get(voltage1, "black") if voltage1 == voltage2 else "black"

        start_coord = (
            int(start_coord_px[0] // step),
            int(start_coord_px[1] // step),
        )
        end_coord = (
            int(end_coord_px[0] // step),
            int(end_coord_px[1] // step),
        )

        start_coord = (
            max(0, min(start_coord[0], num_steps - 1)),
            max(0, min(start_coord[1], num_steps - 1)),
        )
        end_coord = (
            max(0, min(end_coord[0], num_steps - 1)),
            max(0, min(end_coord[1], num_steps - 1)),
        )

        points[start_coord[0]][start_coord[1]] = 0
        points[end_coord[0]][end_coord[1]] = 0

        print(f"Finding path from {start_coord} to {end_coord}")
        try:
            # The grid is updated with path weights after each run.
            # A new graph is created from the grid on each call.
            path, points, _ = findpath.run_gridsearch(
                start_coord,
                end_coord,
                points,
                path_weight=LINE_CROSS_WEIGHT,
            )
            if len(path) > 1:
                print(f"Drawing path with {len(path)} points")
                for i in range(len(path) - 1):
                    start, end = path[i], path[i + 1]
                    drawing.append(
                        draw.Line(
                            start[0] * step,
                            start[1] * step,
                            end[0] * step,
                            end[1] * step,
                            stroke=colour,
                            stroke_width=2,
                        )
                    )
        except Exception as e:
            print(f"Error finding path: {e}")


def draw_titles(
    drawing: draw.Drawing,
    substations: list[Substation],
    points: list[list],
    params: DrawingParams,
):
    """Draws titles for each substation in a clear area."""
    num_steps = len(points)
    step = params.grid_step
    max_search_radius_grid = TITLE_MAX_SEARCH_RADIUS_PX // step

    for sub in substations:
        if not sub.title:
            continue

        text_width_px = len(sub.title) * TITLE_FONT_SIZE * 0.6
        text_height_px = TITLE_FONT_SIZE
        text_width_grid = int(math.ceil(text_width_px / step)) + 1
        text_height_grid = int(math.ceil(text_height_px / step)) + 1

        min_x, min_y, max_x, max_y = sub.get_drawing_bbox(params)
        local_center_x = (min_x + max_x) / 2
        local_center_y = (min_y + max_y) / 2
        center_x = sub.use_x + local_center_x
        center_y = sub.use_y + local_center_y
        center_gx = int(round(center_x / step))
        center_gy = int(round(center_y / step))

        found_spot = False
        for r_grid in range(1, max_search_radius_grid + 1):
            for angle_deg in range(0, 360, 15):
                angle_rad = math.radians(angle_deg)
                gx = center_gx + int(r_grid * math.cos(angle_rad))
                gy = center_gy + int(r_grid * math.sin(angle_rad))

                is_clear = True
                for i in range(text_width_grid):
                    for j in range(text_height_grid):
                        check_gx, check_gy = gx + i, gy - j
                        if not (
                            0 <= check_gx < num_steps
                            and 0 <= check_gy < num_steps
                            and points[check_gx][check_gy] == 0
                        ):
                            is_clear = False
                            break
                    if not is_clear:
                        break

                if is_clear:
                    title_x, title_y = gx * step, gy * step
                    drawing.append(
                        draw.Text(
                            sub.title,
                            font_size=TITLE_FONT_SIZE,
                            x=title_x,
                            y=title_y,
                            fill="black",
                        )
                    )
                    for i in range(text_width_grid):
                        for j in range(text_height_grid):
                            mark_gx, mark_gy = gx + i, gy - j
                            if 0 <= mark_gx < num_steps and 0 <= mark_gy < num_steps:
                                points[mark_gx][mark_gy] = 1
                    found_spot = True
                    break
            if found_spot:
                break

        if not found_spot:
            print(
                f"Warning: Could not find a free spot for title of substation {sub.name}"
            )


def render_substation_svg(
    substation: Substation, params: DrawingParams = None, filename: str = None
) -> str:
    """
    Render a single substation as an SVG image for documentation purposes.

    Args:
        substation: The substation to render
        params: Drawing parameters (uses defaults if None)
        filename: Optional filename to save the SVG (if None, returns SVG string)

    Returns:
        SVG content as a string
    """
    if params is None:
        params = DrawingParams()

    # Get the bounding box of the substation
    min_x, min_y, max_x, max_y = substation.get_drawing_bbox(params)

    # Add some padding around the substation
    padding = 2 * params.grid_step
    svg_width = max_x - min_x + 2 * padding
    svg_height = max_y - min_y + 2 * padding

    # Create the drawing with appropriate size
    drawing = draw.Drawing(svg_width, svg_height, origin=(0, 0))

    # Create a temporary copy of the substation with adjusted use coordinates
    # to center it in the SVG with padding
    temp_sub = Substation(
        name=substation.name,
        lat=substation.lat,
        long=substation.long,
        voltage_kv=substation.voltage_kv,
        tags=substation.tags,
        rotation=substation.rotation,
        definition=substation.definition,
        buses=substation.buses,
        connections=substation.connections,
    )

    # Copy objects and other attributes
    temp_sub.objects = substation.objects.copy() if substation.objects else []
    temp_sub.grid_points = substation.grid_points.copy()
    temp_sub.connection_points = substation.connection_points.copy()

    # Set use coordinates to position the substation with padding
    temp_sub.use_x = padding - min_x
    temp_sub.use_y = padding - min_y

    # Generate the substation group
    substation_group = get_substation_group(
        temp_sub, params, rotation=substation.rotation
    )

    # Add the substation to the drawing
    drawing.append(draw.Use(substation_group, temp_sub.use_x, temp_sub.use_y))

    # Add a title
    title_x = svg_width / 2
    title_y = padding / 2
    drawing.append(
        draw.Text(
            substation.name,
            font_size=BUS_LABEL_FONT_SIZE,
            x=title_x,
            y=title_y,
            text_anchor="middle",
            fill="black",
            stroke_width=0,
        )
    )

    # Add voltage level indicator
    voltage_text = f"{substation.voltage_kv} kV"
    voltage_colour = COLOUR_MAP.get(substation.voltage_kv, "black")
    drawing.append(
        draw.Text(
            voltage_text,
            font_size=BUS_LABEL_FONT_SIZE * 0.8,
            x=title_x,
            y=title_y + BUS_LABEL_FONT_SIZE + 5,
            text_anchor="middle",
            fill=voltage_colour,
            stroke_width=0,
        )
    )

    # Add grid overlay for documentation (optional - can be removed)
    grid_colour = "#E0E0E0"
    grid_stroke_width = 0.5

    # Vertical grid lines
    for x in range(0, int(svg_width) + 1, params.grid_step):
        drawing.append(
            draw.Line(
                x,
                0,
                x,
                svg_height,
                stroke=grid_colour,
                stroke_width=grid_stroke_width,
                opacity=0.3,
            )
        )

    # Horizontal grid lines
    for y in range(0, int(svg_height) + 1, params.grid_step):
        drawing.append(
            draw.Line(
                0,
                y,
                svg_width,
                y,
                stroke=grid_colour,
                stroke_width=grid_stroke_width,
                opacity=0.3,
            )
        )

    # Get SVG content as string
    svg_content = drawing.as_svg()

    # Save to file if filename provided
    if filename:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(svg_content)
        print(f"Saved substation SVG to {filename}")

    return svg_content


# --- Output Generation ---
def generate_substation_documentation_svgs(
    substations: list[Substation], output_dir: str = "substation_docs"
):
    """
    Generate individual SVG files for each substation for documentation purposes.

    Args:
        substations: List of substations to render
        output_dir: Directory to save the SVG files
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    params = DrawingParams()

    print(f"\nGenerating documentation SVGs for {len(substations)} substations...")

    for substation in substations:
        # Create a safe filename from the substation name
        safe_name = "".join(
            c for c in substation.name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        safe_name = safe_name.replace(" ", "_")
        filename = os.path.join(output_dir, f"{safe_name}.svg")

        try:
            render_substation_svg(substation, params, filename)
        except Exception as e:
            print(f"Error rendering {substation.name}: {e}")

    print(f"Documentation SVGs saved to {output_dir}/")


def generate_output_files(drawing: draw.Drawing, substations: list[Substation]):
    """Saves the SVG and generates the final HTML file."""
    drawing.save_svg(OUTPUT_SVG)

    locations_data = []
    for sub in substations:
        title = sub.title if sub.title else sub.name
        leaflet_y = MAP_DIMS - sub.use_y
        leaflet_x = sub.use_x
        locations_data.append(
            f'{{ title: "{title}", coords: [{leaflet_y}, {leaflet_x}] }}'
        )

    locations_json_string = (
        "[\n        " + ",\n        ".join(locations_data) + "\n    ]"
    )

    with open(OUTPUT_SVG, "r", encoding="utf-8") as f:
        svg_content = f.read()

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template_content = f.read()

    svg_content_escaped = svg_content.replace("`", "\\`")
    html_content = template_content.replace("%%SVG_CONTENT%%", svg_content_escaped)
    html_content = html_content.replace("%%LOCATIONS_DATA%%", locations_json_string)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nGenerated {OUTPUT_HTML} with embedded SVG.")


# --- Main Execution ---
def main():
    """Main function to run the SLD generation process."""
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--docs":
        # Generate documentation SVGs
        substation_map = load_substations_from_yaml(SUBSTATIONS_DATA_FILE)
        substations = list(substation_map.values())
        generate_substation_documentation_svgs(substations)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Generate single substation SVG
        if len(sys.argv) < 3:
            print("Usage: python sld.py --single <substation_name>")
            return

        substation_name = sys.argv[2]
        substation_map = load_substations_from_yaml(SUBSTATIONS_DATA_FILE)

        if substation_name not in substation_map:
            print(f"Substation '{substation_name}' not found.")
            print(f"Available substations: {', '.join(substation_map.keys())}")
            return

        substation = substation_map[substation_name]
        filename = f"{substation_name.replace(' ', '_')}_single.svg"
        render_substation_svg(substation, filename=filename)
        return

    params = DrawingParams()

    # 1. Load data
    substation_map = load_substations_from_yaml(SUBSTATIONS_DATA_FILE)
    substations = list(substation_map.values())

    # 2. Calculate initial positions
    calculate_initial_scaled_positions(substations)

    # 3. Apply network layout adjustments
    final_pos = apply_spring_layout(substations)

    # 4. Calculate final drawing positions (use_x, use_y)
    calculate_final_positions(substations, final_pos, params)

    # 5. Create substation drawing groups
    substation_groups = {
        sub.name: get_substation_group(sub, params, rotation=sub.rotation)
        for sub in substations
    }

    # 6. Draw substations onto the main canvas
    drawing = draw.Drawing(MAP_DIMS, MAP_DIMS, origin=(0, 0))
    for sub in substations:
        drawing.append(draw.Use(substation_groups[sub.name], sub.use_x, sub.use_y))

    # 7. Prepare for and draw connections
    num_steps = MAP_DIMS // GRID_STEP + 1
    points = [[0 for _ in range(num_steps)] for _ in range(num_steps)]

    populate_pathfinding_grid(substations, points, params)
    all_connections = calculate_connection_points(substations, params)
    draw_connections(drawing, all_connections, points, GRID_STEP)

    # draw red circules on grid keep out areas for debugging
    for i in range(num_steps):
        for j in range(num_steps):
            if points[i][j] < 1:
                col = "grey"
            elif points[i][j] > 10:
                col = "red"
            else:
                col = "orange"
            drawing.append(draw.Circle(i * GRID_STEP, j * GRID_STEP, 1, fill=col))

    # Draw circles at connection points for debugging
    for connection in all_connections.values():
        for point in connection:
            coords = point["coords"]
            voltage = point["voltage"]
            colour = COLOUR_MAP.get(voltage, "black")
            drawing.append(draw.Circle(coords[0], coords[1], 5, fill=colour))

    # 8. Draw titles
    draw_titles(drawing, substations, points, params)

    # 9. Generate output files
    generate_output_files(drawing, substations)


if __name__ == "__main__":
    main()
