# Standard library imports
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations

# Third-party imports
import drawsvg as draw
import numpy as np
import utm
import yaml
import svgelements
import os
import networkx as nx

# Local application/library specific imports
import findpath
from rectangle_spacing import space_rectangles

# --- Constants ---
MAP_DIMS = 16000
BUS_LABEL_FONT_SIZE = 15
TITLE_MAX_SEARCH_RADIUS_PX = 300
TITLE_FONT_SIZE = 20
BUSBAR_WEIGHT = 8
NEAR_SUB_WEIGHT = 4
ELEMENT_WEIGHT = 50000
GRID_STEP = 25
PATHFINDING_ITERATIONS = 5
CONGESTION_PENALTY = 19
# Default font family to use for all text
DEFAULT_FONT_FAMILY = "Roboto"
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent
SUBSTATIONS_DATA_FILE = SCRIPT_DIR / "substation_definitions.yaml"
TEMPLATE_FILE = SCRIPT_DIR / "index.template.html"
OUTPUT_SVG = "sld.svg"
OUTPUT_HTML = "index.html"
VERSION = "1"

# below colours from AEMO NEM SLD pdf for consistency
COLOUR_MAP = {
    11: "#4682B4",
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
    child_definitions: list = field(default_factory=list)
    object_popups: list = field(default_factory=list)
    scaled_x: float = 0
    scaled_y: float = 0
    x: float = 0.0
    y: float = 0.0
    use_x: float = 0.0  # Final drawing coordinate
    use_y: float = 0.0  # Final drawing coordinate

    def __post_init__(self):
        self.grid_points = {}  # Store (x,y) -> weight dictionary for grid points
        self.connection_points: dict[str, list[dict]] = {}
        self.objects = []  # Store objects associated with this substation

    def draw_objects(
        self,
        parent_group: draw.Group,
        objects_to_draw: list,
        params: DrawingParams = DrawingParams(),
        x_offset: float = 0,
        y_offset: float = 0,
        owner_id: str = "main",
    ) -> draw.Group:
        """Draw all objects associated with the substation.

        This method iterates through a list of object definitions, draws them
        onto the provided parent group, and marks their positions on the
        substation's pathfinding grid. It handles various object types like
        transformers and generators.

        Args:
            parent_group: The `draw.Group` to which the objects will be added.
            objects_to_draw: A list of dictionaries, each defining an object.
            params: Drawing parameters for sizes and steps.
            x_offset: An additional X offset for all objects in the group.
            y_offset: An additional Y offset for all objects in the group.
            owner_id: The identifier for the owner of these objects, used for
                pathfinding penalties.

        Returns:
            The parent group with the new objects drawn onto it.
        """
        import math  # Import at the top for rotation calculations

        conn_points = {}  # Initialize conn_points here
        for obj in objects_to_draw:
            # The origin for objects is the leftmost point of the first busbar.
            # The first busbar starts at x = -grid_step and y = 0.
            origin_x = -params.grid_step
            origin_y = 0
            obj_x = origin_x + obj["rel_x"] * params.grid_step + x_offset
            obj_y = origin_y + obj["rel_y"] * params.grid_step + y_offset
            rotation = obj.get("rotation", 0)

            # Create a group for this object
            obj_group = draw.Group()

            if obj["type"] == "tx-ud":
                # Draw up-down transformer. Winding 1 is at the top.
                # (obj_x, obj_y) is the connection point for winding 1.
                # Total height is 4 * grid_step.

                # Get colours from metadata
                w1_voltage = obj.get("metadata", {}).get("w1")
                w2_voltage = obj.get("metadata", {}).get("w2")
                colour1 = COLOUR_MAP.get(w1_voltage, "black")
                colour2 = COLOUR_MAP.get(w2_voltage, "black")

                # --- Define Geometry ---
                # Symbol is two interlocking circles in a 2*grid_step space.
                radius = (2 * params.grid_step) / 3

                # Circles are centered in the middle 2 grid steps of the 4 grid step total height.
                symbol_top_y = obj_y + params.grid_step
                symbol_center_y = symbol_top_y + params.grid_step

                circle1_y = symbol_center_y - radius / 2
                circle2_y = symbol_center_y + radius / 2

                # --- Top Terminal Line ---
                top_line_start_y = obj_y
                top_line_end_y = symbol_top_y
                top_line = draw.Line(
                    obj_x,
                    top_line_start_y,
                    obj_x,
                    top_line_end_y,
                    stroke=colour1,
                    stroke_width=2,
                )
                obj_group.append(top_line)

                # --- Top Circle ---
                top_circle = draw.Circle(
                    obj_x,
                    circle1_y,
                    radius,
                    fill="transparent",
                    stroke=colour1,
                    stroke_width=3,
                )
                obj_group.append(top_circle)

                # --- Bottom Circle ---
                bottom_circle = draw.Circle(
                    obj_x,
                    circle2_y,
                    radius,
                    fill="transparent",
                    stroke=colour2,
                    stroke_width=3,
                )
                obj_group.append(bottom_circle)

                # --- Bottom Terminal Line ---
                bottom_line_start_y = symbol_top_y + 2 * params.grid_step
                bottom_line_end_y = bottom_line_start_y + params.grid_step
                bottom_line = draw.Line(
                    obj_x,
                    bottom_line_start_y,
                    obj_x,
                    bottom_line_end_y,
                    stroke=colour2,
                    stroke_width=2,
                )
                obj_group.append(bottom_line)

                # Mark grid points for the transformer elements
                mark_grid_point(
                    self,
                    obj_x,
                    top_line_start_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    obj_x,
                    top_line_end_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    obj_x,
                    symbol_center_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    obj_x,
                    bottom_line_start_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    obj_x,
                    bottom_line_end_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )

                # Store connection points
                conn_points = {}
                if "connections" in obj:
                    for terminal_num, conn_name in obj["connections"].items():
                        coords = None
                        voltage = None
                        if terminal_num == 1:  # Top terminal
                            coords = (obj_x, top_line_start_y)
                            voltage = w1_voltage
                        elif terminal_num == 2:  # Bottom terminal
                            coords = (obj_x, bottom_line_end_y)
                            voltage = w2_voltage

                        if coords:
                            conn_points[conn_name] = {
                                "coords": coords,
                                "voltage": voltage,
                                "owner": owner_id,
                            }

            elif obj["type"] == "tx-lr":
                # Draw left-right transformer. Winding 1 is on the left.
                # (obj_x, obj_y) is the connection point for winding 1.
                # Total width is 4 * grid_step.

                # Get colours from metadata
                w1_voltage = obj.get("metadata", {}).get("w1")
                w2_voltage = obj.get("metadata", {}).get("w2")
                colour1 = COLOUR_MAP.get(w1_voltage, "black")
                colour2 = COLOUR_MAP.get(w2_voltage, "black")

                # --- Define Geometry ---
                # Symbol is two interlocking circles in a 2*grid_step space.
                radius = (2 * params.grid_step) / 3

                # Circles are centered in the middle 2 grid steps of the 4 grid step total width.
                symbol_left_x = obj_x + params.grid_step
                symbol_center_x = symbol_left_x + params.grid_step

                circle1_x = symbol_center_x - radius / 2
                circle2_x = symbol_center_x + radius / 2

                # --- Left Terminal Line ---
                left_line_start_x = obj_x
                left_line_end_x = symbol_left_x
                left_line = draw.Line(
                    left_line_start_x,
                    obj_y,
                    left_line_end_x,
                    obj_y,
                    stroke=colour1,
                    stroke_width=2,
                )
                obj_group.append(left_line)

                # --- Left Circle ---
                left_circle = draw.Circle(
                    circle1_x,
                    obj_y,
                    radius,
                    fill="transparent",
                    stroke=colour1,
                    stroke_width=3,
                )
                obj_group.append(left_circle)

                # --- Right Circle ---
                right_circle = draw.Circle(
                    circle2_x,
                    obj_y,
                    radius,
                    fill="transparent",
                    stroke=colour2,
                    stroke_width=3,
                )
                obj_group.append(right_circle)

                # --- Right Terminal Line ---
                right_line_start_x = symbol_left_x + 2 * params.grid_step
                right_line_end_x = right_line_start_x + params.grid_step
                right_line = draw.Line(
                    right_line_start_x,
                    obj_y,
                    right_line_end_x,
                    obj_y,
                    stroke=colour2,
                    stroke_width=2,
                )
                obj_group.append(right_line)

                # Mark grid points for the transformer elements
                mark_grid_point(
                    self,
                    left_line_start_x,
                    obj_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    left_line_end_x,
                    obj_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    symbol_center_x,
                    obj_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    right_line_start_x,
                    obj_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    right_line_end_x,
                    obj_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )

                # Store connection points
                conn_points = {}
                if "connections" in obj:
                    for terminal_num, conn_name in obj["connections"].items():
                        coords = None
                        voltage = None
                        if terminal_num == 1:  # Left terminal
                            coords = (left_line_start_x, obj_y)
                            voltage = w1_voltage
                        elif terminal_num == 2:  # Right terminal
                            coords = (right_line_end_x, obj_y)
                            voltage = w2_voltage

                        if coords:
                            conn_points[conn_name] = {
                                "coords": coords,
                                "voltage": voltage,
                                "owner": owner_id,
                            }

            elif obj["type"] == "gen":
                # Generator is a circle. The reference point (obj_x, obj_y) is the center.
                voltage = obj.get("metadata", {}).get("voltage")
                colour = COLOUR_MAP.get(voltage, "black")

                circle_center_x = obj_x
                circle_center_y = obj_y

                # Draw circle
                obj_group.append(
                    draw.Circle(
                        circle_center_x,
                        circle_center_y,
                        params.grid_step,
                        fill="transparent",
                        stroke=colour,
                        stroke_width=2,
                    )
                )

                # Mark grid point for the circle center and perimeter
                mark_grid_point(
                    self,
                    circle_center_x,
                    circle_center_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    circle_center_x,
                    circle_center_y - params.grid_step,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    circle_center_x,
                    circle_center_y + params.grid_step,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    circle_center_x - params.grid_step,
                    circle_center_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )
                mark_grid_point(
                    self,
                    circle_center_x + params.grid_step,
                    circle_center_y,
                    weight=ELEMENT_WEIGHT,
                    owner_id=owner_id,
                )

            # Apply rotation if specified
            if rotation != 0 and obj["type"] not in ["gen", "tx-ud", "tx-lr"]:
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
                        # Rotate (for Y-down coordinate system, this is a clockwise rotation)
                        rx = tx * math.cos(rotation_rad) + ty * math.sin(rotation_rad)
                        ry = -tx * math.sin(rotation_rad) + ty * math.cos(rotation_rad)
                        # Translate back
                        rotated_x = rx + obj_x
                        rotated_y = ry + obj_y
                        # Update the connection point with rotated coordinates
                        self.connection_points.setdefault(conn_id, []).append(
                            (rotated_x, rotated_y)
                        )

                        # The grid point was already marked with local coordinates.
                        # The rotation is handled globally in populate_pathfinding_grid.
            else:
                # For non-rotated objects, store the connection points directly
                if "connections" in obj:
                    for conn_id, data in conn_points.items():
                        # Store connection points for non-rotated objects
                        self.connection_points.setdefault(conn_id, []).append(data)

            parent_group.append(obj_group)

            # Draw text for 'gen' object separately to avoid manual rotation calculation
            if obj["type"] == "gen":
                text = obj.get("metadata", {}).get("text", "G")
                voltage = obj.get("metadata", {}).get("voltage")
                colour = COLOUR_MAP.get(voltage, "black")

                # The reference point (obj_x, obj_y) is the center of the circle.
                circle_center_x = obj_x
                circle_center_y = obj_y

                # Store popup info for generator objects with metadata.info
                if obj.get("metadata", {}).get("info"):
                    # Calculate global coordinates for the popup
                    global_circle_x = (
                        self.use_x + circle_center_x * params.grid_step / GRID_STEP
                    )
                    global_circle_y = (
                        self.use_y + circle_center_y * params.grid_step / GRID_STEP
                    )

                    info_text = obj["metadata"]["info"]
                    info_text = info_text.replace('"', '\\"')
                    info_text = info_text.replace("\n", "<br>")

                    # Apply y-axis inversion for Leaflet coordinates
                    leaflet_x = global_circle_x
                    leaflet_y = MAP_DIMS - global_circle_y

                    self.object_popups.append(
                        {
                            "info": info_text,
                            "coords": (
                                leaflet_y,
                                leaflet_x,
                            ),  # Format as [y, x] for Leaflet
                        }
                    )

                # Draw text (always horizontal)
                parent_group.append(
                    draw.Text(
                        text,
                        font_size=params.grid_step * 0.9,
                        x=circle_center_x,
                        y=circle_center_y,
                        text_anchor="middle",
                        dominant_baseline="central",
                        fill=colour,
                        stroke_width=0,
                        font_family=DEFAULT_FONT_FAMILY,
                    )
                )

        return parent_group


def get_substation_bbox_from_svg(
    substation: "Substation", params: "DrawingParams"
) -> tuple[float, float, float, float]:
    """Calculates the bounding box of a substation by rendering it to SVG.

    This function renders a substation's definition to a temporary, unrotated
    SVG file. It then uses the `svgelements` library to parse the SVG and
    calculate its precise bounding box. This is used to determine the
    substation's dimensions for layout and spacing calculations.

    Args:
        substation: The `Substation` object to measure.
        params: The drawing parameters to use for rendering.

    Returns:
        A tuple (min_x, min_y, max_x, max_y) representing the
        unrotated bounding box relative to the substation's local origin.
    """
    # Create a temporary drawing of a fixed large size
    temp_drawing = draw.Drawing(2000, 2000, origin=(0, 0))

    # Create a temporary copy of the substation to avoid modifying the original
    temp_sub = Substation(
        name=substation.name,
        lat=substation.lat,
        long=substation.long,
        voltage_kv=substation.voltage_kv,
        tags=substation.tags,
        rotation=0,  # Render unrotated to get the base bounding box
        definition=substation.definition,
        buses=substation.buses,
        connections=substation.connections,
        child_definitions=substation.child_definitions,
    )
    temp_sub.objects = substation.objects.copy() if substation.objects else []

    # Center the substation in the temporary drawing to ensure all parts are rendered
    temp_sub.use_x = 1000
    temp_sub.use_y = 1000

    # Generate the substation group (unrotated)
    # We pass a dummy bbox to get_substation_group as it's not used for drawing, only for rotation center.
    # Since we render unrotated, the center is not important here.
    substation_group = get_substation_group(
        temp_sub, params, (0, 0, 0, 0), rotation=0, draw_title=False
    )

    temp_drawing.append(draw.Use(substation_group, temp_sub.use_x, temp_sub.use_y))

    # Save to a temporary file
    safe_name = "".join(c for c in substation.name if c.isalnum() or c in ("_", "-"))
    temp_svg_path = f"temp_{safe_name}.svg"
    temp_drawing.save_svg(temp_svg_path)

    try:
        # Use svgelements to get the bounding box
        svg = svgelements.SVG.parse(temp_svg_path)
        x, y, x_max, y_max = svg.bbox()
        width = x_max - x
        height = y_max - y

        # The bbox is global to the temp SVG. We need to make it relative to the substation's
        # local origin. The substation was placed at (1000, 1000).
        min_x = x - temp_sub.use_x
        min_y = y - temp_sub.use_y
        max_x = min_x + width
        max_y = min_y + height

        return min_x, min_y, max_x, max_y

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_svg_path):
            os.remove(temp_svg_path)


def get_rotated_bbox(
    bbox: tuple[float, float, float, float], rotation_deg: int
) -> tuple[float, float, float, float]:
    """Calculates the axis-aligned bounding box of a rotated rectangle.

    Args:
        bbox: A tuple (min_x, min_y, max_x, max_y) of the unrotated rectangle.
        rotation_deg: The rotation angle in degrees.

    Returns:
        A tuple (min_x, min_y, max_x, max_y) representing the new
        axis-aligned bounding box that encloses the rotated rectangle.
    """
    if rotation_deg % 360 == 0:
        return bbox

    min_x, min_y, max_x, max_y = bbox
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    rotation_rad = math.radians(rotation_deg)
    cos_r = math.cos(rotation_rad)
    sin_r = math.sin(rotation_rad)

    corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]

    rotated_corners = []
    for x, y in corners:
        # Translate to origin for rotation
        rel_x = x - center_x
        rel_y = y - center_y
        # Rotate (for Y-down coordinate system, this is a clockwise rotation)
        rotated_x = rel_x * cos_r + rel_y * sin_r
        rotated_y = -rel_x * sin_r + rel_y * cos_r
        # Translate back
        rotated_corners.append((rotated_x + center_x, rotated_y + center_y))

    rotated_xs = [p[0] for p in rotated_corners]
    rotated_ys = [p[1] for p in rotated_corners]

    return min(rotated_xs), min(rotated_ys), max(rotated_xs), max(rotated_ys)


# --- Data Loading ---
def load_substations_from_yaml(filename: str) -> dict[str, Substation]:
    """Load substations from a YAML file into a dictionary.

    Args:
        filename: The path to the YAML file containing substation definitions.

    Returns:
        A dictionary mapping substation names to `Substation` objects.
    """
    with open(filename, "r") as f:
        data = yaml.safe_load(f)

    substations_map = {}

    for sub_data in data["substations"]:
        # Create substation
        substation = Substation(
            name=sub_data["name"],
            lat=sub_data["lat"],
            long=sub_data["long"],
            voltage_kv=sub_data["voltage_kv"],
            tags=sub_data.get("tags", [sub_data["name"]]),
            rotation=sub_data.get("rotation", 0),
            definition=sub_data.get("def", ""),
            buses=sub_data.get("buses", {}),
            connections=sub_data.get("connections", {}),
            child_definitions=sub_data.get("child_definitions", []),
        )

        # Add objects to the substation if present in the data
        if "objects" in sub_data:
            substation.objects = sub_data["objects"]

        substations_map[substation.name] = substation
    return substations_map


# --- Drawing Helpers ---
def mark_grid_point(
    sub: "Substation", x: float, y: float, weight: int = 25, owner_id: str = "main"
) -> None:
    """Marks a grid point in the substation's grid_points dictionary.

    This function stores a pathfinding weight and an owner ID for a specific
    coordinate in the substation's local grid. This information is used later
    to build the global pathfinding grid.

    Args:
        sub: The `Substation` object to which the grid point belongs.
        x: The x-coordinate of the grid point.
        y: The y-coordinate of the grid point.
        weight: The pathfinding cost associated with this point.
        owner_id: The identifier for the owner of this grid point.
    """
    sub.grid_points[(x, y)] = (weight, owner_id)


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
    """Generic function to draw a switch (CB or isolator) at given coordinates.

    Args:
        x: The x-coordinate for the center of the switch.
        y: The y-coordinate for the center of the switch.
        parent_group: The `draw.Group` to which the switch will be added.
        switch_type: The type of switch to draw (e.g., CB, ISOL).
        orientation: The orientation of the switch ('vertical' or 'horizontal').
        rotation_angle: The angle for isolator switches.
        params: Drawing parameters for sizes.
        colour: The stroke colour for the switch.

    Returns:
        The parent group with the new switch drawn onto it.
    """
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
                font_family=DEFAULT_FONT_FAMILY,
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
    owner_id: str = "main",
) -> draw.Group:
    """Draws a single bay based on a definition string.

    This function parses a bay definition string, then iterates through the
    elements (busbars, switches, connections) to draw them vertically. It
    handles connecting lines between elements and alignment.

    Args:
        xoff: The x-offset for the entire bay.
        parent_group: The `draw.Group` to which the bay will be added.
        bay_def: The string defining the bay's layout.
        sub: The parent `Substation` object.
        is_first_bay: Flag indicating if this is the first bay, used for
            drawing bus labels.
        params: Drawing parameters for sizes and steps.
        previous_bay_elements: A list of elements from the previous bay, used
            to determine busbar continuity.
        y_offset: A y-offset to align the start of the bay.
        owner_id: The identifier for the owner of this bay.

    Returns:
        The parent group with the new bay drawn onto it.
    """
    colour = COLOUR_MAP.get(sub.voltage_kv, "black")
    elements = parse_bay_elements(bay_def)
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
                        owner_id=owner_id,
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
                owner_id=owner_id,
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
                        owner_id=owner_id,
                    )

            y_pos = draw_element_object(
                element,
                xoff,
                y_pos,
                parent_group,
                sub,
                params,
                colour,
                owner_id=owner_id,
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
            should_draw_dot = False
            if i > 0 and i < len(elements) - 1:
                prev_element = elements[i - 1]
                next_element = elements[i + 1]
                if (
                    prev_element["type"] == "element"
                    and prev_element["subtype"] != "empty"
                    and next_element["type"] == "element"
                    and next_element["subtype"] != "empty"
                ):
                    should_draw_dot = True
            draw_connection_object(
                element,
                xoff,
                y_pos,
                parent_group,
                sub,
                colour,
                draw_dot=should_draw_dot,
                owner_id=owner_id,
            )

    return parent_group


def _mark_busbar_grid_points(
    sub: "Substation",
    xoff: float,
    y_pos: float,
    extend_left: bool,
    weight: int,
    owner_id: str,
    params: DrawingParams,
):
    """Marks grid points for a busbar element.

    Args:
        sub: The `Substation` object.
        xoff: The central x-offset of the busbar segment.
        y_pos: The y-position of the busbar.
        extend_left: Whether the busbar should extend further to the left.
        weight: The pathfinding weight to assign to the grid points.
        owner_id: The owner identifier for the grid points.
        params: Drawing parameters.
    """
    x_positions = [
        xoff - params.grid_step,
        xoff,
        xoff + params.grid_step,
    ]
    if extend_left:
        x_positions.insert(0, xoff - 2 * params.grid_step)

    for x in x_positions:
        mark_grid_point(sub, x, y_pos, weight=weight, owner_id=owner_id)


def _draw_standard_element_frame(
    parent_group: draw.Group,
    xoff: float,
    y_pos: float,
    colour: str,
    params: DrawingParams,
):
    """Draws the top and bottom connecting lines for a standard 3-step element.

    Args:
        parent_group: The `draw.Group` to draw on.
        xoff: The x-coordinate for the element.
        y_pos: The starting y-coordinate for the element.
        colour: The stroke colour for the lines.
        params: Drawing parameters.
    """
    # Top line
    parent_group.append(
        draw.Line(
            xoff, y_pos, xoff, y_pos + params.grid_step, stroke=colour, stroke_width=2
        )
    )
    # Bottom line
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


def _draw_standard_element_symbol(
    parent_group: draw.Group,
    subtype: str,
    xoff: float,
    y_pos: float,
    colour: str,
    params: DrawingParams,
):
    """Draws the central symbol for a standard 3-step element.

    Args:
        parent_group: The `draw.Group` to draw on.
        subtype: The subtype of the element ('cb', 'isolator', 'unknown').
        xoff: The x-coordinate for the element.
        y_pos: The starting y-coordinate for the element.
        colour: The stroke colour for the symbol.
        params: Drawing parameters.
    """
    symbol_center_y = y_pos + params.grid_step + (params.grid_step / 2)
    if subtype == "cb":
        parent_group.append(
            draw.Rectangle(
                xoff - params.grid_step / 2,
                symbol_center_y - params.grid_step / 2,
                params.grid_step,
                params.grid_step,
                fill="white",
                stroke=colour,
            )
        )
    elif subtype == "unknown":
        parent_group.append(
            draw.Text(
                "?",
                font_size=params.grid_step,
                x=xoff,
                y=symbol_center_y,
                text_anchor="middle",
                dominant_baseline="central",
                fill=colour,
                stroke_width=0,
                font_family=DEFAULT_FONT_FAMILY,
            )
        )
    elif subtype == "isolator":
        isolator_half_size = params.grid_step / 2
        parent_group.append(
            draw.Line(
                xoff - isolator_half_size,
                symbol_center_y - isolator_half_size,
                xoff + isolator_half_size,
                symbol_center_y + isolator_half_size,
                stroke=colour,
                stroke_width=2,
            )
        )


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
    owner_id: str = "main",
):
    """Draw a busbar object at the specified position.

    Handles drawing different types of busbars (standard, string, tie-ins)
    and marking their grid points for pathfinding.

    Args:
        element: The dictionary defining the busbar element.
        xoff: The x-offset for the bay.
        y_pos: The y-position for the busbar.
        parent_group: The `draw.Group` to draw on.
        sub: The parent `Substation` object.
        is_first_bay: Flag for drawing bus labels.
        params: Drawing parameters.
        colour: The stroke colour.
        previous_bay_elements: Elements of the previous bay for continuity checks.
        owner_id: The owner identifier for pathfinding.

    Returns:
        The new y-position after drawing the object.
    """
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
        _mark_busbar_grid_points(
            sub, xoff, y_pos, extend_left, BUSBAR_WEIGHT, owner_id, params
        )

        # Add text label if first bay
        if is_first_bay:
            bus_id = element["id"]
            bus_name = sub.buses.get(bus_id, "")
            parent_group.append(
                draw.Text(
                    bus_name,
                    x=xoff - params.grid_step - 5,
                    y=y_pos,
                    font_size=BUS_LABEL_FONT_SIZE,
                    text_anchor="end",
                    dominant_baseline="central",
                    stroke_width=0,
                    font_family=DEFAULT_FONT_FAMILY,
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
        _mark_busbar_grid_points(
            sub, xoff, y_pos, extend_left, BUSBAR_WEIGHT, owner_id, params
        )

    elif subtype == "null":
        # No line drawn, but mark grid points spanning 3*GRID_STEP (or 4*GRID_STEP if extending)
        _mark_busbar_grid_points(
            sub, xoff, y_pos, extend_left, BUSBAR_WEIGHT, owner_id, params
        )

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
        _mark_busbar_grid_points(
            sub, xoff, y_pos, extend_left, ELEMENT_WEIGHT, owner_id, params
        )

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
        _mark_busbar_grid_points(
            sub, xoff, y_pos, extend_left, ELEMENT_WEIGHT, owner_id, params
        )

    return y_pos


def draw_element_object(
    element, xoff, y_pos, parent_group, sub, params, colour, owner_id: str = "main"
):
    """Draw an element object at the specified position.

    Handles drawing different types of bay elements (CB, isolator, etc.)
    and marking their grid points for pathfinding.

    Args:
        element: The dictionary defining the element.
        xoff: The x-offset for the bay.
        y_pos: The starting y-position for the element.
        parent_group: The `draw.Group` to draw on.
        sub: The parent `Substation` object.
        params: Drawing parameters.
        colour: The stroke colour.
        owner_id: The owner identifier for pathfinding.

    Returns:
        The new y-position after drawing the object.
    """
    subtype = element["subtype"]

    if subtype in ["cb", "unknown", "isolator"]:
        _draw_standard_element_frame(parent_group, xoff, y_pos, colour, params)
        _draw_standard_element_symbol(
            parent_group, subtype, xoff, y_pos, colour, params
        )
        for i in range(4):
            mark_grid_point(
                sub,
                xoff,
                y_pos + i * params.grid_step,
                weight=ELEMENT_WEIGHT,
                owner_id=owner_id,
            )

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
        for i in range(4):
            mark_grid_point(
                sub,
                xoff,
                y_pos + i * params.grid_step,
                weight=ELEMENT_WEIGHT,
                owner_id=owner_id,
            )

    elif subtype == "empty":
        # Empty element: no drawing, no grid marking, but advance position
        pass

    return y_pos + 3 * params.grid_step


def parse_bay_elements(bay_def: str) -> list:
    """Parse a bay definition string into a list of element dictionaries.

    Args:
        bay_def: The string defining the bay's layout.

    Returns:
        A list of dictionaries, where each dictionary represents a parsed
        element from the definition string.
    """
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

        elif char == "?":
            elements.append({"type": "element", "subtype": "unknown"})
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


def draw_connection_object(
    element,
    xoff,
    y_pos,
    parent_group,
    sub,
    colour,
    draw_dot: bool = False,
    owner_id: str = "main",
):
    """Draw a connection object at the specified position.

    This function doesn't draw a visible object itself (unless a dot is
    requested) but records the location of a named connection point for
    later pathfinding between substations.

    Args:
        element: The dictionary defining the connection.
        xoff: The x-coordinate of the connection point.
        y_pos: The y-coordinate of the connection point.
        parent_group: The `draw.Group` to which a dot might be added.
        sub: The parent `Substation` object.
        colour: The colour for the optional dot.
        draw_dot: Whether to draw a visible dot at the connection point.
        owner_id: The owner identifier for pathfinding.
    """
    conn_id = element["id"]
    connection_name = sub.connections.get(conn_id)

    if connection_name:
        # Store the connection point for pathfinding, including voltage
        connection_data = {
            "coords": (xoff, y_pos),
            "voltage": sub.voltage_kv,
            "owner": owner_id,
        }
        sub.connection_points.setdefault(connection_name, []).append(connection_data)
        mark_grid_point(
            sub, xoff, y_pos, weight=ELEMENT_WEIGHT, owner_id=owner_id
        )  # Connection points are now handled by pathfinder logic
    if draw_dot:
        parent_group.append(draw.Circle(xoff, y_pos, 5, fill=colour, stroke="none"))


def get_substation_group(
    sub: Substation,
    params: DrawingParams,
    bbox: tuple[float, float, float, float],
    rotation=0,
    draw_title: bool = True,
):
    """Creates a `draw.Group` containing all SVG elements for a substation.

    This function orchestrates the drawing of all bays, objects, and titles
    for a single substation into a group. The group is rotated around its
    calculated center.

    Args:
        sub: The `Substation` object to draw.
        params: Drawing parameters.
        bbox: The unrotated bounding box of the substation, used to find the
            rotation center.
        rotation: The rotation angle in degrees.
        draw_title: Whether to draw the substation's name as a title.

    Returns:
        A `draw.Group` containing the complete visual representation of the
        substation.
    """
    min_x, min_y, max_x, max_y = bbox
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Snap rotation center to the nearest grid point to ensure grid alignment after rotation
    grid_step = params.grid_step
    center_x = round(center_x / grid_step) * grid_step
    center_y = round(center_y / grid_step) * grid_step

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
        max_y_offset = max(max_y_offset, y_offset)

    previous_bay_elements = None
    for i, bay_def in enumerate(bay_defs):
        xoff = 2 * params.grid_step * i  # Use 2*GRID_STEP spacing between bays (50px)
        is_first_bay = i == 0

        # handle correct offset
        if parsed_bays[i][0]["type"] == "busbar":
            y_offset = 0
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
            owner_id="main",
        )

    # Draw objects after bays
    if sub.objects:
        dg = sub.draw_objects(
            parent_group=dg,
            objects_to_draw=sub.objects,
            params=params,
            owner_id="main",
        )

    # Draw child definitions
    if sub.child_definitions:
        original_voltage = sub.voltage_kv
        original_connections = sub.connections
        original_buses = sub.buses
        for child_def in sub.child_definitions:
            sub.voltage_kv = child_def["voltage_kv"]
            sub.connections = child_def.get("connections", {})
            sub.buses = child_def.get("buses", {})

            child_bay_defs = child_def["def"].strip().split("\n")
            child_parsed_bays = [
                parse_bay_elements(bay_def) for bay_def in child_bay_defs
            ]

            # Pre-calculate the maximum y-offset needed for any bay to align all busbars
            child_max_y_offset = 0
            for elements in child_parsed_bays:
                y_offset_for_alignment = 0
                first_busbar_idx = next(
                    (i for i, el in enumerate(elements) if el["type"] == "busbar"), -1
                )
                if first_busbar_idx > 0:
                    # Count elements before the first busbar
                    for i in range(first_busbar_idx):
                        if elements[i]["type"] == "element":
                            y_offset_for_alignment += 3 * params.grid_step
                child_max_y_offset = max(child_max_y_offset, y_offset_for_alignment)

            child_previous_bay_elements = None
            for i, bay_def in enumerate(child_bay_defs):
                base_xoff = child_def["rel_x"] * params.grid_step
                # Allow any x offset value, not just multiples of bay_width
                bay_width = 2 * params.grid_step
                xoff = base_xoff + (bay_width * i)

                # handle correct y offset for this bay within the child
                y_offset_for_alignment = 0
                if child_parsed_bays[i] and child_parsed_bays[i][0]["type"] != "busbar":
                    y_offset_for_alignment = child_max_y_offset

                # y_offset is negative of desired y_pos start
                # The final y_offset should combine the relative position and the alignment.
                y_offset = (
                    -(child_def["rel_y"] * params.grid_step) + y_offset_for_alignment
                )

                if i > 0:
                    child_previous_bay_elements = child_parsed_bays[i - 1]

                draw_bay_from_string(
                    xoff,
                    dg,
                    bay_def,
                    sub,
                    is_first_bay=(i == 0),  # Child bays can now have bus labels
                    params=params,
                    previous_bay_elements=child_previous_bay_elements,
                    y_offset=y_offset,
                    owner_id=f"child_{i}",
                )

            # Draw child objects
            if "objects" in child_def and child_def["objects"]:
                bay_width = 2 * params.grid_step
                child_x_offset = child_def["rel_x"] * params.grid_step
                child_x_offset = round(child_x_offset / bay_width) * bay_width
                child_y_offset = child_def["rel_y"] * params.grid_step
                dg = sub.draw_objects(
                    parent_group=dg,
                    objects_to_draw=child_def["objects"],
                    params=params,
                    x_offset=child_x_offset,
                    y_offset=child_y_offset,
                    owner_id=f"child_{i}",
                )
        sub.voltage_kv = original_voltage
        sub.connections = original_connections
        sub.buses = original_buses

    if draw_title:
        # Add title centered above the substation
        min_x, min_y, max_x, _ = bbox
        title_x = (min_x + max_x) / 2
        # Place title one and a half grid steps above the top of the bounding box
        title_y = min_y - (1.5 * params.grid_step)
        dg.append(
            draw.Text(
                sub.name,
                font_size=TITLE_FONT_SIZE,
                x=title_x,
                y=title_y,
                text_anchor="middle",
                dominant_baseline="text-after-edge",
                fill="black",
                stroke_width=0,
                font_family=DEFAULT_FONT_FAMILY,
            )
        )

        # Mark grid points under the title to prevent line crossovers
        # Approximate text width. A common heuristic is num_chars * font_size * 0.6
        text_width = len(sub.name) * TITLE_FONT_SIZE * 0.6
        start_x = title_x - text_width / 2
        end_x = title_x + text_width / 2

        # Find the grid points that this text spans
        grid_y = round(title_y / params.grid_step) * params.grid_step

        grid_start_x_idx = math.floor(start_x / params.grid_step)
        grid_end_x_idx = math.ceil(end_x / params.grid_step)

        # Mark the grid row of the title, and the row above it
        for y_offset in [-params.grid_step, 0]:
            current_grid_y = grid_y + y_offset
            for i in range(grid_start_x_idx, grid_end_x_idx + 1):
                grid_x = i * params.grid_step
                mark_grid_point(
                    sub, grid_x, current_grid_y, weight=ELEMENT_WEIGHT, owner_id="main"
                )

    return dg


# --- Layout and Positioning Functions ---
def calculate_initial_scaled_positions(substations: list[Substation]):
    """Converts lat/lon to UTM and scales them to fit the map.

    This function takes a list of substations, converts their geographic
    coordinates to UTM, and then scales and translates these UTM coordinates
    to fit within the main drawing canvas dimensions (`MAP_DIMS`).

    Args:
        substations: A list of `Substation` objects to position.
    """
    if not substations:
        return

    # Convert all substation lat/longs to UTM at once to ensure they are in the same zone
    lats = np.array([sub.lat for sub in substations])
    longs = np.array([sub.long for sub in substations])
    eastings, northings, _, _ = utm.from_latlon(lats, longs)
    northings = [-x for x in northings]

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

    # Apply scaling and translation to fit within MAP_DIMS
    # Use margin of 5% on each side (10% total)
    margin = MAP_DIMS * 0.05

    for sub in substations:
        sub.scaled_x = (sub.x - min_x) * scale_factor + margin
        sub.scaled_y = (sub.y - min_y) * scale_factor + margin


def calculate_connection_points(
    substations: list[Substation],
    params: DrawingParams,
    bboxes: dict[str, tuple[float, float, float, float]],
) -> dict[str, list[dict]]:
    """Calculates global coordinates for all connection points.

    This function iterates through all substations and their locally defined
    connection points. It applies the substation's final rotation and
    translation to convert these local points into global coordinates on the
    main canvas.

    Args:
        substations: A list of all `Substation` objects.
        params: Drawing parameters.
        bboxes: A dictionary mapping substation names to their unrotated
            bounding boxes.

    Returns:
        A dictionary where keys are line definition names and values are lists
        of connection point data (including global coordinates).
    """
    all_connections: dict[str, list[dict]] = {}
    for sub in substations:
        min_x, min_y, max_x, max_y = bboxes[sub.name]
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Snap rotation center to the nearest grid point to match get_substation_group
        grid_step = params.grid_step
        center_x = round(center_x / grid_step) * grid_step
        center_y = round(center_y / grid_step) * grid_step

        rotation_rad = math.radians(sub.rotation)

        for linedef, connection_data_list in sub.connection_points.items():
            if not linedef:
                continue

            for connection_data in connection_data_list:
                local_coords = connection_data["coords"]
                voltage = connection_data["voltage"]
                owner = connection_data.get("owner", "main")

                local_x, local_y = local_coords
                rel_x = local_x - center_x
                rel_y = local_y - center_y
                # Rotate (for Y-down coordinate system, this is a clockwise rotation)
                rotated_x = rel_x * math.cos(rotation_rad) + rel_y * math.sin(
                    rotation_rad
                )
                rotated_y = -rel_x * math.sin(rotation_rad) + rel_y * math.cos(
                    rotation_rad
                )
                rotated_local_x = rotated_x + center_x
                rotated_local_y = rotated_y + center_y

                global_coords = (
                    sub.use_x + rotated_local_x,
                    sub.use_y + rotated_local_y,
                )
                new_connection_data = {
                    "coords": global_coords,
                    "voltage": voltage,
                    "substation": sub.name,
                    "owner": owner,
                }
                all_connections.setdefault(linedef, []).append(new_connection_data)
    return all_connections


def draw_connections(
    drawing: draw.Drawing,
    all_connections: dict,
    points: list[list],
    grid_owners: list[list],
    step: int,
    sub_global_bounds: dict,
):
    """Finds paths and draws connections between substations.

    This function prepares pathfinding requests from the global connection
    points, calls the pathfinder to get the routes, and then draws the
    resulting paths onto the main drawing canvas.

    Args:
        drawing: The main `draw.Drawing` object.
        all_connections: A dictionary of all connection points.
        points: The 2D grid of pathfinding weights.
        grid_owners: The 2D grid of pathfinding point owners.
        step: The grid step size.
        sub_global_bounds: A dictionary mapping substation names to their
            global bounding boxes on the grid.
    """
    num_steps = len(points)

    def _distance(a, b) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    valid_connections = {k: v for k, v in all_connections.items() if len(v) == 2}
    sorted_connections = sorted(
        valid_connections.items(),
        key=lambda item: _distance(item[1][0]["coords"], item[1][1]["coords"]),
    )

    path_requests = []
    path_metadata = []
    all_connection_nodes = set()

    for _, connection_points in sorted_connections:
        start_coord_px = connection_points[0]["coords"]
        end_coord_px = connection_points[1]["coords"]

        sub1_name = connection_points[0]["substation"]
        sub2_name = connection_points[1]["substation"]
        owner1 = connection_points[0]["owner"]
        owner2 = connection_points[1]["owner"]

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

        # The pathfinder uses (row, col) which is (y, x). Our coords are (x, y).
        start_node = (start_coord[1], start_coord[0])
        end_node = (end_coord[1], end_coord[0])

        all_connection_nodes.add(start_node)
        all_connection_nodes.add(end_node)

        # Ensure start/end points are clear for pathfinding
        points[start_node[0]][start_node[1]] = 0
        points[end_node[0]][end_node[1]] = 0

        request = {
            "start": start_node,
            "end": end_node,
            "substations": {sub1_name, sub2_name},
            "start_owner": (sub1_name, owner1),
            "end_owner": (sub2_name, owner2),
        }

        if sub1_name == sub2_name and sub1_name in sub_global_bounds:
            request["bounds"] = sub_global_bounds[sub1_name]

        path_requests.append(request)
        path_metadata.append({"colour": colour})

    print(f"Step 5.1: Finding {len(path_requests)} paths...")
    try:
        all_paths = findpath.run_all_gridsearches(
            path_requests=path_requests,
            points=points,
            grid_owners=grid_owners,
            congestion_penalty_increment=CONGESTION_PENALTY,
            all_connection_nodes=all_connection_nodes,
            busbar_weight=BUSBAR_WEIGHT,
            busbar_crossing_penalty=100000,
        )

        # Build a map of nodes to the orientation of paths passing through them.
        node_orientations = {}
        for path_idx, path in enumerate(all_paths):
            if not path:
                continue
            for j in range(len(path) - 1):
                p1, p2 = path[j], path[j + 1]
                is_vertical = p1[1] == p2[1]
                for node in (p1, p2):
                    if node not in node_orientations:
                        node_orientations[node] = {"v": set(), "h": set()}
                    if is_vertical:
                        node_orientations[node]["v"].add(path_idx)
                    else:
                        node_orientations[node]["h"].add(path_idx)

        for i, path in enumerate(all_paths):
            if len(path) > 1:
                colour = path_metadata[i]["colour"]
                # The path is returned as (row, col) tuples.
                # Start the path data string with a "Move to" command.
                start_node = path[0]
                path_data = f"M {start_node[1] * step} {start_node[0] * step}"

                # Add "Line to" commands for the rest of the path.
                for j in range(1, len(path)):
                    p_curr = path[j]
                    p_prev = path[j - 1]
                    is_vertical = p_curr[1] == p_prev[1]

                    # A barrier is a busbar or another path that is horizontal.
                    # Vertical paths should break when crossing a horizontal barrier.
                    orientations_curr = node_orientations.get(p_curr, {"h": set()})
                    is_h_path_curr = bool(orientations_curr["h"] - {i})
                    p_curr_is_barrier = (
                        points[p_curr[0]][p_curr[1]] == BUSBAR_WEIGHT
                    ) or is_h_path_curr

                    orientations_prev = node_orientations.get(p_prev, {"h": set()})
                    is_h_path_prev = bool(orientations_prev["h"] - {i})
                    p_prev_is_barrier = (
                        points[p_prev[0]][p_prev[1]] == BUSBAR_WEIGHT
                    ) or is_h_path_prev

                    # Check for vertical crossings of horizontal barriers to add a visual gap.
                    if is_vertical and p_curr_is_barrier and not p_prev_is_barrier:
                        # Path is entering a barrier node from a non-barrier node.
                        # Shorten the line to create a gap before the barrier.
                        direction = p_curr[0] - p_prev[0]
                        path_data += f" L {p_curr[1] * step} {p_curr[0] * step - (7 * direction)}"
                    elif is_vertical and not p_curr_is_barrier and p_prev_is_barrier:
                        # Path is leaving a barrier node to a non-barrier node.
                        # Start the new line with a gap after the barrier.
                        direction = p_curr[0] - p_prev[0]
                        path_data += f" M {p_prev[1] * step} {p_prev[0] * step + (7 * direction)}"
                        path_data += f" L {p_curr[1] * step} {p_curr[0] * step}"
                    else:
                        # Default behavior: draw a continuous line segment.
                        path_data += f" L {p_curr[1] * step} {p_curr[0] * step}"

                # Create a single, consolidated path element.
                drawing.append(
                    draw.Path(
                        d=path_data,
                        stroke=colour,
                        stroke_width=2,
                        fill="none",
                    )
                )
    except Exception as e:
        print(f"Error finding paths: {e}")


def render_substation_svg(
    substation: Substation, params: DrawingParams = None, filename: str = None
) -> str:
    """Renders a single substation as a standalone SVG image.

    This is primarily used for generating documentation images. It calculates
    the required SVG dimensions, centers the substation, adds a title, and
    optionally saves it to a file.

    Args:
        substation: The `Substation` object to render.
        params: Drawing parameters. If None, defaults are used.
        filename: If provided, the SVG is saved to this path.

    Returns:
        The SVG content as a string.
    """
    if params is None:
        params = DrawingParams()

    # Get the bounding box of the substation
    min_x, min_y, max_x, max_y = get_substation_bbox_from_svg(substation, params)

    # Add some padding around the substation
    padding = 2 * params.grid_step
    svg_width = max_x - min_x + 2 * padding
    svg_height = max_y - min_y + 2 * padding

    # Create the drawing with appropriate size
    drawing = draw.Drawing(svg_width, svg_height, origin=(0, 0))
    drawing.append(draw.Rectangle(0, 0, svg_width, svg_height, fill="transparent"))

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
        child_definitions=substation.child_definitions,
    )

    # Copy objects and other attributes
    temp_sub.objects = substation.objects.copy() if substation.objects else []
    temp_sub.grid_points = substation.grid_points.copy()
    temp_sub.connection_points = substation.connection_points.copy()

    # Set use coordinates to position the substation with padding
    temp_sub.use_x = padding - min_x
    temp_sub.use_y = padding - min_y

    # Generate the substation group
    bbox = (min_x, min_y, max_x, max_y)
    substation_group = get_substation_group(
        temp_sub, params, bbox, rotation=substation.rotation
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
            font_family=DEFAULT_FONT_FAMILY,
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
    grid_colour = "#F2F2F2"
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
    """Generates individual SVG files for each substation for documentation.

    Args:
        substations: A list of `Substation` objects to render.
        output_dir: The directory where the SVG files will be saved.
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    params = DrawingParams()

    print(f"\nGenerating documentation SVGs for {len(substations)} substations...")

    for substation in substations:
        # Create a safe filename from the substation name
        safe_name = (
            substation.name.replace("/", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace(" ", "_")
        )
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("_", "-"))
        filename = os.path.join(output_dir, f"{safe_name}.svg")

        try:
            render_substation_svg(substation, params, filename)
        except Exception as e:
            print(f"Error rendering {substation.name}: {e}")

    print(f"Documentation SVGs saved to {output_dir}/")


def generate_output_files(
    drawing: draw.Drawing, substations: list[Substation], sub_bboxes: dict
):
    """Saves the main SVG and generates the final HTML file.

    This function saves the complete drawing to an SVG file, then embeds that
    SVG content into an HTML template. It also injects location data for
    interactive markers into the HTML.

    Args:
        drawing: The final `draw.Drawing` object.
        substations: The list of all `Substation` objects.
        sub_bboxes: A dictionary of substation bounding boxes.
    """
    # Add Google font embedding for Roboto
    drawing.embed_google_font(
        DEFAULT_FONT_FAMILY, text=None
    )  # None means all characters

    # Save the SVG with embedded font
    drawing.save_svg(OUTPUT_SVG)

    locations_data = []
    object_popups_data = []

    for sub in substations:
        title = sub.name if sub.name else sub.name

        min_x, min_y, max_x, max_y = sub_bboxes[sub.name]
        local_center_x = (min_x + max_x) / 2
        local_center_y = (min_y + max_y) / 2

        global_center_x = sub.use_x + local_center_x
        global_center_y = sub.use_y + local_center_y

        # Invert y-axis for Leaflet coordinates
        leaflet_y = MAP_DIMS - global_center_y
        leaflet_x = global_center_x
        locations_data.append(
            f'{{ title: "{title}", coords: [{leaflet_y}, {leaflet_x}] }}'
        )

        # Use the popup data that was stored during object drawing
        # This already contains properly calculated coordinates and formatted info text
        if hasattr(sub, "object_popups") and sub.object_popups:
            for popup in sub.object_popups:
                object_popups_data.append(
                    f'{{ info: "{popup["info"]}", coords: [{popup["coords"][0]}, {popup["coords"][1]}] }}'
                )

    # Create JSON strings
    locations_json_string = (
        "[\n        " + ",\n        ".join(locations_data) + "\n    ]"
    )

    object_popups_json_string = (
        ("[\n        " + ",\n        ".join(object_popups_data) + "\n    ]")
        if object_popups_data
        else "[]"
    )

    with open(OUTPUT_SVG, "r", encoding="utf-8") as f:
        svg_content = f.read()

    with open(TEMPLATE_FILE, "r", encoding="utf-8") as f:
        template_content = f.read()

    svg_content_escaped = svg_content.replace("`", "\\`")
    html_content = template_content.replace("%%SVG_CONTENT%%", svg_content_escaped)
    html_content = html_content.replace("%%VERSION%%", VERSION)
    html_content = html_content.replace("%%LOCATIONS_DATA%%", locations_json_string)
    html_content = html_content.replace("%%OBJECT_POPUPS%%", object_popups_json_string)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Step 6.1: Generated {OUTPUT_HTML} with embedded SVG.")


# --- Main Execution ---
def _handle_cli_args(substation_map: dict[str, Substation]) -> bool:
    """Handles command-line arguments for special modes.

    Checks for arguments like `--docs` or `--single` to run specific
    generation tasks instead of the full map generation.

    Args:
        substation_map: The dictionary of all loaded substations.

    Returns:
        True if a CLI argument was handled (and the main process should exit),
        False otherwise.
    """
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--docs":
            substations = list(substation_map.values())
            generate_substation_documentation_svgs(substations)
            return True
        if sys.argv[1] == "--single":
            if len(sys.argv) < 3:
                print("Usage: python sld.py --single <substation_name>")
                return True
            substation_name = sys.argv[2]
            if substation_name not in substation_map:
                print(f"Substation '{substation_name}' not found.")
                print(f"Available substations: {', '.join(substation_map.keys())}")
                return True
            substation = substation_map[substation_name]
            filename = f"{substation_name.replace(' ', '_')}_single.svg"
            render_substation_svg(substation, filename=filename)
            return True
    return False


def _prepare_substation_layout(
    substations: list[Substation], params: DrawingParams
) -> dict:
    """Calculates layout, spacing, and final positions for all substations.

    This function orchestrates the entire layout process:
    1. Calculates initial scaled positions from lat/lon.
    2. Determines bounding boxes for each substation.
    3. Uses the rectangle spacing algorithm to prevent overlaps.
    4. Sets the final `use_x` and `use_y` coordinates for drawing.

    Args:
        substations: The list of all `Substation` objects.
        params: Drawing parameters.

    Returns:
        A dictionary mapping substation names to their calculated unrotated
        bounding boxes.
    """
    calculate_initial_scaled_positions(substations)

    print("Step 2.1: Calculating substation bounding boxes...")
    sub_bboxes = {}
    for sub in substations:
        min_x, min_y, max_x, max_y = get_substation_bbox_from_svg(sub, params)
        min_x = round(min_x / params.grid_step) * params.grid_step
        min_y = round(min_y / params.grid_step) * params.grid_step
        max_x = round(max_x / params.grid_step) * params.grid_step
        max_y = round(max_y / params.grid_step) * params.grid_step
        sub_bboxes[sub.name] = (min_x, min_y, max_x, max_y)

    rotated_sub_bboxes = {}
    for sub in substations:
        sub.rotation = round(sub.rotation / 90) * 90
        rotated_sub_bboxes[sub.name] = get_rotated_bbox(
            sub_bboxes[sub.name], sub.rotation
        )

    MIN_PADDING_STEPS = 6
    PADDING_RATIO = 35
    paddings_in_steps = []
    for sub in substations:
        min_x, min_y, max_x, max_y = rotated_sub_bboxes[sub.name]
        width_px = max_x - min_x
        height_px = max_y - min_y
        width_grid_steps = width_px / params.grid_step
        height_grid_steps = height_px / params.grid_step
        area_grid_steps = width_grid_steps * height_grid_steps
        padding = max(MIN_PADDING_STEPS, round(area_grid_steps / PADDING_RATIO))
        paddings_in_steps.append(padding)
    print(f"Step 2.2: Calculated dynamic padding: {paddings_in_steps}")

    initial_rects = []
    for sub in substations:
        min_x, min_y, max_x, max_y = rotated_sub_bboxes[sub.name]
        width = max_x - min_x
        height = max_y - min_y
        x1 = sub.scaled_x - width / 2
        y1 = sub.scaled_y - height / 2
        x2 = sub.scaled_x + width / 2
        y2 = sub.scaled_y + height / 2
        initial_rects.append((x1, y1, x2, y2))

    print("Step 2.3: Spacing rectangles to avoid overlap...")
    shifts = space_rectangles(
        rectangles=initial_rects,
        grid_size=params.grid_step,
        debug_images=False,
        padding_steps=paddings_in_steps,
    )

    print("Step 2.4: Finalizing substation positions...")
    for i, sub in enumerate(substations):
        shift_x, shift_y = shifts[i]
        sub.use_x = (
            round((sub.scaled_x + shift_x) / params.grid_step) * params.grid_step
        )
        sub.use_y = (
            round((sub.scaled_y + shift_y) / params.grid_step) * params.grid_step
        )

    return sub_bboxes


def _populate_pathfinding_grid(
    substations: list[Substation],
    sub_bboxes: dict,
    params: DrawingParams,
) -> tuple[list[list[int]], list[list[tuple[str, str]]], dict]:
    """Populates the grid with weights and boundaries for pathfinding.

    This function creates the main data structures for the pathfinder. It
    marks the grid with weights based on where substations and their
    components are located, applying rotations and translations to place them
    correctly in the global grid space.

    Args:
        substations: The list of all `Substation` objects.
        sub_bboxes: A dictionary of substation bounding boxes.
        params: Drawing parameters.

    Returns:
        A tuple containing:
        - points: The 2D grid of pathfinding weights.
        - grid_owners: The 2D grid of owner IDs for each point.
        - sub_global_bounds: A dictionary mapping substation names to their
          global bounding boxes on the grid.
    """
    num_steps = MAP_DIMS // GRID_STEP + 1
    points = [[0 for _ in range(num_steps)] for _ in range(num_steps)]
    grid_owners = [[None for _ in range(num_steps)] for _ in range(num_steps)]
    sub_global_bounds = {}

    for sub in substations:
        min_x, min_y, max_x, max_y = sub_bboxes[sub.name]
        box_margin = 50

        rotation_rad = math.radians(sub.rotation)
        center_x = round(((min_x + max_x) / 2) / params.grid_step) * params.grid_step
        center_y = round(((min_y + max_y) / 2) / params.grid_step) * params.grid_step

        corners = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
        rotated_corners = []
        for x, y in corners:
            rel_x = x - center_x
            rel_y = y - center_y
            rot_x = rel_x * math.cos(rotation_rad) + rel_y * math.sin(rotation_rad)
            rot_y = -rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)
            rotated_corners.append((rot_x + center_x, rot_y + center_y))

        rot_min_x = min(p[0] for p in rotated_corners)
        rot_max_x = max(p[0] for p in rotated_corners)
        rot_min_y = min(p[1] for p in rotated_corners)
        rot_max_y = max(p[1] for p in rotated_corners)

        grid_min_x = max(0, int((sub.use_x + rot_min_x - box_margin) / GRID_STEP))
        grid_min_y = max(0, int((sub.use_y + rot_min_y - box_margin) / GRID_STEP))
        grid_max_x = min(
            num_steps - 1, int((sub.use_x + rot_max_x + box_margin) / GRID_STEP)
        )
        grid_max_y = min(
            num_steps - 1, int((sub.use_y + rot_max_y + box_margin) / GRID_STEP)
        )
        sub_global_bounds[sub.name] = (grid_min_x, grid_min_y, grid_max_x, grid_max_y)

        for grid_y in range(grid_min_y, grid_max_y + 1):
            for grid_x in range(grid_min_x, grid_max_x + 1):
                points[grid_y][grid_x] = 1
                grid_owners[grid_y][grid_x] = (sub.name, "main")

        cos_r = math.cos(rotation_rad)
        sin_r = math.sin(rotation_rad)
        for (local_x, local_y), (weight, owner_id) in sub.grid_points.items():
            rel_x = local_x - center_x
            rel_y = local_y - center_y
            rotated_x = rel_x * cos_r + rel_y * sin_r
            rotated_y = -rel_x * sin_r + rel_y * cos_r
            global_x = sub.use_x + (rotated_x + center_x)
            global_y = sub.use_y + (rotated_y + center_y)
            grid_x = int(round(global_x / GRID_STEP))
            grid_y = int(round(global_y / GRID_STEP))
            if 0 <= grid_y < num_steps and 0 <= grid_x < num_steps:
                points[grid_y][grid_x] = weight
                grid_owners[grid_y][grid_x] = (sub.name, owner_id)

    return points, grid_owners, sub_global_bounds


def main():
    """Main function to run the SLD generation process."""
    print("Step 1: Loading substation data...")
    substation_map = load_substations_from_yaml(SUBSTATIONS_DATA_FILE)
    if _handle_cli_args(substation_map):
        return

    params = DrawingParams()
    substations = list(substation_map.values())

    # 1. Prepare layout
    print("Step 2: Preparing substation layout...")
    sub_bboxes = _prepare_substation_layout(substations, params)

    # 2. Create substation drawing groups
    print("Step 3: Creating substation drawing groups...")
    substation_groups = {
        sub.name: get_substation_group(
            sub, params, sub_bboxes[sub.name], rotation=sub.rotation
        )
        for sub in substations
    }

    # 3. Draw substations onto the main canvas
    print("Step 4: Drawing substations on canvas...")
    drawing = draw.Drawing(MAP_DIMS, MAP_DIMS, origin=(0, 0))
    drawing.append(draw.Rectangle(0, 0, MAP_DIMS, MAP_DIMS, fill="transparent"))
    for sub in substations:
        drawing.append(draw.Use(substation_groups[sub.name], sub.use_x, sub.use_y))

    # 4. Prepare for and draw connections
    print("Step 5: Preparing and drawing connections...")
    points, grid_owners, sub_global_bounds = _populate_pathfinding_grid(
        substations, sub_bboxes, params
    )
    all_connections = calculate_connection_points(substations, params, sub_bboxes)
    draw_connections(
        drawing, all_connections, points, grid_owners, GRID_STEP, sub_global_bounds
    )

    # 5. Generate output files
    print("Step 6: Generating output files...")
    generate_output_files(drawing, substations, sub_bboxes)

    print("\nSLD generation complete.")


if __name__ == "__main__":
    main()
