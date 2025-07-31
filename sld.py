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

from dataclasses import dataclass, field
from enum import Enum, auto
import math
from itertools import combinations
import utm
import json
import numpy as np


# modifying so bus is included in bay definition
import findpath

MAP_DIMS = 5000
BUS_LABEL_FONT_SIZE = 15
TITLE_MAX_SEARCH_RADIUS_PX = 300
TITLE_FONT_SIZE = 40
BUSBAR_WEIGHT = 7


@dataclass
class DrawingParams:
    grid_step: int = 25
    bay_width: int = 50
    cb_size: int = 25
    isolator_size: int = 25


def mark_grid_point(sub: "Substation", x: float, y: float, weight: int = 25) -> None:
    """Mark a grid point in the substation's grid_points dictionary with a weight.

    Args:
        sub (Substation): The substation to mark the grid point in
        x (float): The x coordinate of the grid point
        y (float): The y coordinate of the grid point
        weight (int): The weight to assign to this point, default is 25 (high penalty)
    """
    # Add the point to the substation's grid_points dictionary with its weight
    sub.grid_points[(x, y)] = weight


drawing_params = DrawingParams()


class SwitchType(Enum):
    NOBUS = auto()
    BUSTIE = auto()
    EMPTY = auto()
    DIRECT = auto()
    CB = auto()
    ISOL = auto()


class FeederConnectionPoint(Enum):
    """Defines connection points for feeders on a bay."""

    A = auto()  # Feeder below the top element.
    B = auto()  # Feeder below the middle element (for Breaker-and-a-half).


@dataclass
class BaseBay:
    bus_name: str
    elementAbove: SwitchType = SwitchType.EMPTY  # Element above the first busbar
    elementAboveConnection: str = ""
    elementBelow: SwitchType = SwitchType.CB
    elementBelowConnection: str = ""
    flip: bool = True
    neighbour: "BaseBay" = None

    def __post_init__(self):
        self.points = [[]]
        self.connections = {}


@dataclass
class SingleSwitchedBay(BaseBay):
    pass


@dataclass
class DoubleSwitchedBay(BaseBay):
    other_bus_name: str = ""
    elementOtherBus: SwitchType = SwitchType.CB
    elementOtherBusConnection: str = ""


@dataclass
class BreakerAndHalfBay(DoubleSwitchedBay):
    elementTie: SwitchType = SwitchType.CB
    elementTieConnection: str = ""


@dataclass
class BusTieBay(BaseBay):
    tie_bus_name: str = ""


@dataclass
class Substation:
    name: str
    lat: float
    long: float
    voltage_kv: int
    bus_name: str = ""
    other_bus_name: str = ""
    tags: list[str] = field(default_factory=list)
    rotation: int = 0
    bays: list[BaseBay] = field(default_factory=list)
    scaled_x: float = 0
    scaled_y: float = 0
    x: float = 0.0
    y: float = 0.0
    title: str = ""

    def __post_init__(self):
        self.grid_points = {}  # Store (x,y) -> weight dictionary for grid points
        self.connection_points: dict[str, tuple[float, float]] = {}

    def add_bay(self, bay: BaseBay) -> None:
        self.bays.append(bay)
        # set each bay to its left neighbour
        for i, _ in enumerate(self.bays):
            if i == 0:
                continue
            self.bays[i].neighbour = self.bays[i - 1]

    def get_drawing_bbox(
        self, params: DrawingParams
    ) -> tuple[float, float, float, float]:
        """Calculates the bounding box (min_x, min_y, max_x, max_y) of the substation drawing."""
        if not self.bays:
            return 0, 0, 0, 0

        # The drawing functions create continuous busbars.
        # The leftmost point is determined by the first bay.
        first_bay = self.bays[0]
        xoff = 0
        min_x = 0
        if isinstance(first_bay, (SingleSwitchedBay, DoubleSwitchedBay)):
            min_x = xoff - params.bay_width / 2
        elif isinstance(first_bay, BreakerAndHalfBay):
            min_x = xoff - 50

        # The rightmost point is determined by the last bay.
        last_bay = self.bays[-1]
        xoff = 50 * (len(self.bays) - 1)
        max_x = 0
        if isinstance(last_bay, (SingleSwitchedBay, DoubleSwitchedBay)):
            max_x = xoff + params.bay_width / 2
        elif isinstance(last_bay, BreakerAndHalfBay):
            max_x = xoff + 50

        # Height calculation
        # Account for elementAbove which is drawn above the busbar (negative y)
        min_y = 0
        for bay in self.bays:
            # Check if any bay has elementAbove defined
            if hasattr(bay, "elementAbove") and bay.elementAbove not in (
                SwitchType.EMPTY,
                SwitchType.NOBUS,
            ):
                bay_min_y = -1 * (params.grid_step + params.cb_size + params.grid_step)
                if bay_min_y < min_y:
                    min_y = bay_min_y

        # Calculate max_y (drawings go down with positive y)
        max_y = 0
        for bay in self.bays:
            bay_max_y = 0
            if isinstance(bay, BreakerAndHalfBay):
                bay_max_y = 3 * (params.grid_step + params.cb_size + params.grid_step)
            elif isinstance(bay, DoubleSwitchedBay):
                bay_max_y = 6 * params.grid_step
            elif isinstance(bay, SingleSwitchedBay):
                bay_max_y = params.grid_step + params.cb_size + params.grid_step

            if bay_max_y > max_y:
                max_y = bay_max_y

        # For BreakerAndHalf substations, ensure the total height is an even number of grid steps
        # to keep the rotation center on the grid.
        is_bah = any(isinstance(b, BreakerAndHalfBay) for b in self.bays)
        if is_bah:
            height = max_y - min_y
            height_in_steps = round(height / params.grid_step)
            if height_in_steps % 2 != 0:
                max_y += params.grid_step

        return min_x, min_y, max_x, max_y


def load_substations_from_json(
    filename: str,
) -> tuple[list[Substation], dict, list[tuple]]:
    """Load substations, connections, and network edges from JSON file.

    Returns:
        tuple: (substations_list, connections_dict, network_edges_list)
    """
    with open(filename, "r") as f:
        data = json.load(f)

    substations = []

    for sub_data in data["substations"]:
        # Create substation
        substation = Substation(
            name=sub_data["name"],
            title=sub_data.get("title", sub_data["name"]),
            lat=sub_data["lat"],
            long=sub_data["long"],
            voltage_kv=sub_data["voltage_kv"],
            bus_name=sub_data.get("bus_name", ""),
            other_bus_name=sub_data.get("other_bus_name", ""),
            tags=sub_data.get("tags", [sub_data["name"]]),
            rotation=sub_data.get("rotation", 0),
        )

        # Create bays
        for bay_data in sub_data["bays"]:
            bay_type = bay_data["type"]

            # Convert string enum values to SwitchType enums
            elementAbove = SwitchType[bay_data.get("elementAbove", ["EMPTY", ""])[0]]
            elementAboveConnection = bay_data["elementAbove"][1]
            elementBelow = SwitchType[bay_data["elementBelow"][0]]
            elementBelowConnection = bay_data["elementBelow"][1]
            flip = bay_data.get("flip", True)

            # Get bus names, using substation-level as default
            bus_name = bay_data.get("bus_name", substation.bus_name)

            if bay_type == 1:  # SingleSwitchedBay
                bay = SingleSwitchedBay(
                    bus_name=bus_name,
                    elementAbove=elementAbove,
                    elementAboveConnection=elementAboveConnection,
                    elementBelow=elementBelow,
                    elementBelowConnection=elementBelowConnection,
                    flip=flip,
                )
            elif bay_type == 2:  # DoubleSwitchedBay
                elementOtherBus = SwitchType[bay_data["elementOtherBus"][0]]
                elementOtherBusConnection = bay_data["elementOtherBus"][1]
                other_bus_name = bay_data.get(
                    "other_bus_name", substation.other_bus_name
                )
                bay = DoubleSwitchedBay(
                    bus_name=bus_name,
                    other_bus_name=other_bus_name,
                    elementAbove=elementAbove,
                    elementAboveConnection=elementAboveConnection,
                    elementBelow=elementBelow,
                    elementBelowConnection=elementBelowConnection,
                    elementOtherBus=elementOtherBus,
                    elementOtherBusConnection=elementOtherBusConnection,
                    flip=flip,
                )
            elif bay_type == 3:  # BreakerAndHalfBay
                elementTie, elementTieConnection = (
                    SwitchType[bay_data["elementTie"][0]],
                    bay_data["elementTie"][1],
                )

                elementOtherBus = SwitchType[bay_data["elementOtherBus"][0]]
                elementOtherBusConnection = bay_data["elementOtherBus"][1]
                other_bus_name = bay_data.get(
                    "other_bus_name", substation.other_bus_name
                )
                bay = BreakerAndHalfBay(
                    bus_name=bus_name,
                    other_bus_name=other_bus_name,
                    elementAbove=elementAbove,
                    elementAboveConnection=elementAboveConnection,
                    elementBelow=elementBelow,
                    elementBelowConnection=elementBelowConnection,
                    elementTie=elementTie,
                    elementTieConnection=elementTieConnection,
                    elementOtherBus=elementOtherBus,
                    elementOtherBusConnection=elementOtherBusConnection,
                    flip=flip,
                )
            # BusTieBay is not handled as it's not in the numeric types
            else:
                raise ValueError(f"Unknown bay type: {bay_type}")

            substation.add_bay(bay)

        substations.append(substation)
    return substations


# Load substations from JSON instead of hardcoded definitions

substations = load_substations_from_json(
    r"C:\Users\DamienVermeer\Downloads\substations_data.json"
)

# Convert all substation lat/longs to UTM at once to ensure they are in the same zone
if substations:
    lats = np.array([sub.lat for sub in substations])
    longs = np.array([sub.long for sub in substations])
    eastings, northings, _, _ = utm.from_latlon(lats, longs)
    # find largest east
    min_east = np.min(eastings)
    max_east = np.max(eastings)
    min_north = np.min(northings)
    max_north = np.max(northings)

    for i, sub in enumerate(substations):
        sub.x = eastings[i] - min_east
        sub.y = northings[i] - min_north

import drawsvg as draw

d = draw.Drawing(MAP_DIMS, MAP_DIMS, origin=(0, 0))


def draw_switch(
    x: float,
    y: float,
    parent_group: draw.Group,
    switch_type: SwitchType,
    orientation: str = "vertical",
    rotation_angle: int = 45,
    params: DrawingParams = drawing_params,
) -> draw.Group:
    """Generic function to draw a switch (CB or isolator) at given coordinates.

    Args:
        x (float): x coordinate of the switch center
        y (float): y coordinate of the switch center
        parent_group (draw.Group): parent group to add the switch to
        switch_type (SwitchType): type of switch to draw (CB or ISOL)
        orientation (str): "vertical" or "horizontal"
        rotation_angle (int): angle for isolator rotation

    Returns:
        draw.Group: parent group with the switch added
    """
    if switch_type == SwitchType.CB:
        # Circuit breaker is drawn as a rectangle
        if orientation == "vertical":
            parent_group.append(
                draw.Rectangle(
                    x - params.cb_size / 2,
                    y - params.cb_size / 2,
                    params.cb_size,
                    params.cb_size,
                    fill="white",
                )
            )  # cb
        else:  # horizontal
            parent_group.append(
                draw.Rectangle(
                    x - params.cb_size / 2,
                    y - params.cb_size / 2,
                    params.cb_size,
                    params.cb_size,
                    fill="white",
                )
            )  # cb
    elif switch_type == SwitchType.ISOL:
        # Isolator is drawn as a rotated line
        if orientation == "vertical":
            parent_group.append(
                draw.Line(
                    x,
                    y - params.isolator_size / 2,
                    x,
                    y + params.isolator_size / 2,
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
                    stroke_width=2,
                    transform=f"rotate({-rotation_angle}, {x}, {y})",
                )
            )
    return parent_group


def draw_single_switched_bay(
    xoff: float,
    parent_group: draw.Group,
    bay: BaseBay,
    sub: Substation,
    params: DrawingParams = drawing_params,
) -> draw.Group:
    """Draw a CB or isolator at the given x offset and y direction.

    Args:
        xoff (float): x offset from the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (Bay): bay to draw the CB or isolator for

    Returns:
        draw.Group: parent group with the CB or isolator added
    """
    ys = 1 if bay.flip else -1
    # handle bus - only draw if its of type other than NOBUS
    if bay.elementBelow is SwitchType.NOBUS:
        return parent_group
    parent_group.append(
        draw.Line(
            -params.bay_width / 2 + xoff,
            0,
            params.bay_width / 2 + xoff,
            0,
            stroke_width=5,
        )
    )
    mark_grid_point(sub, xoff, 0, weight=BUSBAR_WEIGHT)
    # and left and right of this point on the busbar
    mark_grid_point(sub, xoff + params.grid_step, 0, weight=BUSBAR_WEIGHT)
    mark_grid_point(sub, xoff - params.grid_step, 0, weight=BUSBAR_WEIGHT)

    # handle bus identifier - only draw if its the first bay on the bus
    if bay.neighbour is None or bay.neighbour.elementBelow is SwitchType.NOBUS:
        parent_group.append(
            draw.Text(
                bay.bus_name,
                x=-params.bay_width / 2 + xoff,
                y=-8,
                font_size=BUS_LABEL_FONT_SIZE,
                anchor="end",
                stroke_width=0,
            )
        )

    # Draw elementAbove (ABOVE the busbar)
    if bay.elementAbove is not SwitchType.EMPTY:
        if bay.elementAbove is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(
                draw.Line(
                    xoff,
                    0,
                    xoff,
                    -1 * (params.grid_step + params.cb_size + params.grid_step),
                )
            )
            mark_grid_point(
                sub, xoff, -1 * (params.grid_step + params.cb_size + params.grid_step)
            )
            sub.connection_points[bay.elementAboveConnection] = (
                xoff,
                ys * (params.grid_step + params.cb_size + params.grid_step),
            )
        else:
            # Draw CB or isolator
            # Line from bus to switch
            parent_group.append(draw.Line(xoff, 0, xoff, -1 * params.grid_step))
            mark_grid_point(sub, xoff, -1 * params.grid_step)
            # Draw the switch at the correct position
            draw_switch(
                xoff,
                -1 * (params.grid_step + params.cb_size / 2),
                parent_group,
                bay.elementAbove,
                "vertical",
                params=params,
            )
            # Mark CB center and sides
            mark_grid_point(sub, xoff, -1 * (params.grid_step))
            mark_grid_point(sub, xoff, -1 * (params.grid_step + params.cb_size))
            # Line from switch upward
            parent_group.append(
                draw.Line(
                    xoff,
                    -1 * (params.grid_step + params.cb_size),
                    xoff,
                    -1 * (params.grid_step + params.cb_size + params.grid_step),
                )
            )
            mark_grid_point(
                sub, xoff, -1 * (params.grid_step + params.cb_size + params.grid_step)
            )
            sub.connection_points[bay.elementAboveConnection] = (
                xoff,
                -1 * (params.grid_step + params.cb_size + params.grid_step),
            )

    # Draw main element (BELOW the busbar)
    if bay.elementBelow is SwitchType.EMPTY:
        # nothing else to do
        return parent_group

    elif bay.elementBelow is SwitchType.DIRECT:
        # Line off bus - but is longer than normal as no cb/isol
        parent_group.append(
            draw.Line(
                xoff,
                0,
                xoff,
                ys * (params.grid_step + params.cb_size + params.grid_step),
            )
        )
        mark_grid_point(
            sub, xoff, ys * (params.grid_step + params.cb_size + params.grid_step)
        )
        sub.connection_points[bay.elementBelowConnection] = (
            xoff,
            ys * (params.grid_step + params.cb_size + params.grid_step),
        )
        return parent_group

    # else its either CB or isolator

    # Line off bus towards switch
    parent_group.append(draw.Line(xoff, 0, xoff, ys * params.grid_step))
    mark_grid_point(sub, xoff, ys * params.grid_step)
    # Draw the switch at the correct position
    draw_switch(
        xoff,
        ys * (params.grid_step + params.cb_size / 2),
        parent_group,
        bay.elementBelow,
        "vertical",
        params=params,
    )
    # Mark CB center and sides
    mark_grid_point(sub, xoff, ys * (params.grid_step))
    mark_grid_point(sub, xoff, ys * (params.grid_step + params.cb_size))
    # Line from switch towards feeder
    parent_group.append(
        draw.Line(
            xoff,
            ys * (params.grid_step + params.cb_size),
            xoff,
            ys * (params.grid_step + params.cb_size + params.grid_step),
        )
    )
    mark_grid_point(
        sub, xoff, ys * (params.grid_step + params.cb_size + params.grid_step)
    )
    sub.connection_points[bay.elementBelowConnection] = (
        xoff,
        ys * (params.grid_step + params.cb_size + params.grid_step),
    )

    return parent_group


def draw_double_switched_bay(
    xoff: float,
    parent_group: draw.Group,
    bay: DoubleSwitchedBay,
    sub: Substation,
    first_bay: bool = False,
    params: DrawingParams = drawing_params,
) -> draw.Group:
    """Draw a double switched bay at the given x offset and y direction.

    Args:
        xoff (float): x offset from the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (DoubleSwitchedBay): bay to draw the double switched bay for

    Returns:
        draw.Group: parent group with the double switched bay added
    """
    # top element is the same as single switched bay - this also handles element0
    parent_group = draw_single_switched_bay(
        xoff,
        parent_group,
        bay,
        sub,
        params=params,
    )

    # handle bottom element
    ys = 1 if bay.flip else -1
    # handle bus - only draw if its of type other than NOBUS
    if bay.elementOtherBus is SwitchType.NOBUS:
        return parent_group

    top_feeder_y = ys * (params.grid_step + params.cb_size + params.grid_step)

    # Correctly calculate double-switched bay coordinates based on flip parameter
    bottom_busbar_distance = 6 * params.grid_step * (1 if bay.flip else -1)
    bottom_bus_y = bottom_busbar_distance

    # Calculate bottom element positions relative to busbars
    if bay.flip:  # Elements go down
        bottom_element_top_y = 4 * params.grid_step
        bottom_element_center_y = 4.5 * params.grid_step
        bottom_element_bottom_y = 5 * params.grid_step
    else:  # Elements go up
        bottom_element_bottom_y = -4 * params.grid_step
        bottom_element_center_y = -4.5 * params.grid_step
        bottom_element_top_y = -5 * params.grid_step

    parent_group.append(
        draw.Line(
            -params.bay_width / 2 + xoff,
            bottom_bus_y,
            params.bay_width / 2 + xoff,
            bottom_bus_y,
            stroke_width=5,
        )
    )

    # Mark bottom busbar points in the grid
    for i in range(int(params.bay_width / params.grid_step) + 1):
        mark_grid_point(
            sub, -params.bay_width / 2 + xoff + i * params.grid_step, bottom_bus_y
        )
    mark_grid_point(sub, xoff, bottom_bus_y, weight=BUSBAR_WEIGHT)

    # handle bus identifier - only draw if its the first bay on the bus
    if (
        bay.neighbour is None
        or not hasattr(bay.neighbour, "elementOtherBus")
        or bay.neighbour.elementOtherBus is SwitchType.NOBUS
    ):
        parent_group.append(
            draw.Text(
                bay.other_bus_name,
                x=-params.bay_width / 2 + xoff,
                y=bottom_bus_y + 8,
                font_size=BUS_LABEL_FONT_SIZE,
                anchor="end",
                stroke_width=0,
            )
        )

    if bay.elementOtherBus is SwitchType.EMPTY:
        # nothing else to do
        return parent_group

    elif bay.elementOtherBus is SwitchType.DIRECT:
        # Line off bus - but is longer than normal as no cb/isol
        parent_group.append(draw.Line(xoff, top_feeder_y, xoff, bottom_bus_y))
        mark_grid_point(sub, xoff, bottom_bus_y)
        sub.connection_points[bay.elementOtherBusConnection] = (
            xoff,
            bottom_bus_y,
        )
        return parent_group

    # else its either CB or isolator

    # Line from top feeder towards switch
    parent_group.append(draw.Line(xoff, top_feeder_y, xoff, bottom_element_top_y))
    mark_grid_point(sub, xoff, bottom_element_top_y)
    # Draw the switch at the correct position
    draw_switch(
        xoff,
        bottom_element_center_y,
        parent_group,
        bay.elementOtherBus,
        "vertical",
        params=params,
    )
    mark_grid_point(sub, xoff, bottom_element_center_y)
    # Line from switch towards bottom bus
    parent_group.append(draw.Line(xoff, bottom_element_bottom_y, xoff, bottom_bus_y))
    mark_grid_point(sub, xoff, bottom_element_bottom_y)

    # Mark grid points for the bottom element and connecting lines
    mark_grid_point(sub, xoff, top_feeder_y)  # Top connection point
    sub.connection_points[bay.elementOtherBusConnection] = (
        xoff,
        top_feeder_y,
    )
    mark_grid_point(sub, xoff, bottom_element_center_y)  # Switch center
    mark_grid_point(sub, xoff, bottom_bus_y, weight=BUSBAR_WEIGHT)  # Bottom busbar

    # Mark intermediate points between top feeder and switch
    for i in range(
        1, int(abs(top_feeder_y - bottom_element_top_y) / params.grid_step) + 1
    ):
        y_pos = top_feeder_y + i * params.grid_step * (
            1 if bottom_element_top_y > top_feeder_y else -1
        )
        mark_grid_point(sub, xoff, y_pos)

    # Mark intermediate points between switch and bottom busbar
    for i in range(
        1, int(abs(bottom_element_bottom_y - bottom_bus_y) / params.grid_step) + 1
    ):
        y_pos = bottom_element_bottom_y + i * params.grid_step * (
            1 if bottom_bus_y > bottom_element_bottom_y else -1
        )
        mark_grid_point(sub, xoff, y_pos)
    mark_grid_point(sub, xoff, bottom_element_bottom_y)

    return parent_group


def draw_breaker_and_half_bay(
    xoff: float,
    parent_group: draw.Group,
    bay: BreakerAndHalfBay,
    sub: Substation,
    params: DrawingParams = drawing_params,
) -> draw.Group:
    """Draw a breaker-and-a-half bay with three elements (top, middle, bottom).

    Args:
        xoff (float): x offset from the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (BreakerAndHalfBay): bay to draw the breaker-and-a-half bay for

    Returns:
        draw.Group: parent group with the breaker-and-a-half bay added
    """
    # Draw top bus with proper length to support the bay width
    parent_group.append(draw.Line(-50 + xoff, 0, 50 + xoff, 0, stroke_width=5))

    # Mark top busbar in grid
    for i in range(int(100 / params.grid_step) + 1):
        mark_grid_point(sub, -50 + xoff + i * params.grid_step, 0, weight=BUSBAR_WEIGHT)
    mark_grid_point(sub, xoff, 0, weight=BUSBAR_WEIGHT)

    # Add bus identifier for top bus if needed
    if bay.neighbour is None or not isinstance(bay.neighbour, BreakerAndHalfBay):
        parent_group.append(
            draw.Text(
                bay.bus_name,
                x=-55 + xoff,
                y=-8,
                font_size=BUS_LABEL_FONT_SIZE,
                anchor="end",
                stroke_width=0,
            )
        )

    # Draw elementAbove (ABOVE the first busbar)
    if (
        bay.elementAbove is not SwitchType.EMPTY
        and bay.elementAbove is not SwitchType.NOBUS
    ):
        if bay.elementAbove is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(
                draw.Line(
                    xoff,
                    0,
                    xoff,
                    -1 * (params.grid_step + params.cb_size + params.grid_step),
                )
            )
            mark_grid_point(
                sub, xoff, -1 * (params.grid_step + params.cb_size + params.grid_step)
            )

        else:
            # Draw CB or isolator
            # Line from top bus to switch
            parent_group.append(draw.Line(xoff, 0, xoff, -1 * params.grid_step))
            mark_grid_point(sub, xoff, -1 * params.grid_step)
            # Draw the switch
            draw_switch(
                xoff,
                -1 * (params.grid_step + params.cb_size / 2),
                parent_group,
                bay.elementAbove,
                "vertical",
                params=params,
            )
            mark_grid_point(sub, xoff, -1 * (params.grid_step + params.cb_size / 2))
            # Line from switch upward
            parent_group.append(
                draw.Line(
                    xoff,
                    -1 * (params.grid_step + params.cb_size),
                    xoff,
                    -1 * (params.grid_step + params.cb_size + params.grid_step),
                )
            )
            mark_grid_point(
                sub, xoff, -1 * (params.grid_step + params.cb_size + params.grid_step)
            )

    # TOP ELEMENT
    if (
        bay.elementBelow is not SwitchType.NOBUS
        and bay.elementBelow is not SwitchType.EMPTY
    ):
        if bay.elementBelow is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(
                draw.Line(
                    xoff, 0, xoff, params.grid_step + params.cb_size + params.grid_step
                )
            )
            mark_grid_point(
                sub, xoff, params.grid_step + params.cb_size + params.grid_step
            )
        else:
            # Draw CB or isolator
            # Line from top bus to switch
            parent_group.append(draw.Line(xoff, 0, xoff, params.grid_step))
            mark_grid_point(sub, xoff, params.grid_step)
            # Draw the switch
            draw_switch(
                xoff,
                params.grid_step + params.cb_size / 2,
                parent_group,
                bay.elementBelow,
                "vertical",
                params=params,
            )
            mark_grid_point(sub, xoff, params.grid_step + params.cb_size / 2)
            # Line from switch down
            parent_group.append(
                draw.Line(
                    xoff,
                    params.grid_step + params.cb_size,
                    xoff,
                    params.grid_step + params.cb_size + params.grid_step,
                )
            )
            mark_grid_point(
                sub, xoff, params.grid_step + params.cb_size + params.grid_step
            )
            sub.connection_points[bay.elementBelowConnection] = (
                xoff,
                (params.grid_step + params.cb_size + params.grid_step),
            )

    # MIDDLE ELEMENT
    if (
        bay.elementTie is not SwitchType.NOBUS
        and bay.elementTie is not SwitchType.EMPTY
    ):
        top_conn_y = params.grid_step + params.cb_size + params.grid_step
        if bay.elementTie is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(
                draw.Line(
                    xoff,
                    top_conn_y,
                    xoff,
                    top_conn_y + params.grid_step + params.cb_size + params.grid_step,
                )
            )
            mark_grid_point(
                sub,
                xoff,
                top_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )
            sub.connection_points[bay.elementTieConnection] = (
                xoff,
                top_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )
        else:
            # Draw CB or isolator
            # Line to switch
            parent_group.append(
                draw.Line(xoff, top_conn_y, xoff, top_conn_y + params.grid_step)
            )
            mark_grid_point(sub, xoff, top_conn_y + params.grid_step)
            # Draw the switch
            draw_switch(
                xoff,
                top_conn_y + params.grid_step + params.cb_size / 2,
                parent_group,
                bay.elementTie,
                "vertical",
                params=params,
            )
            mark_grid_point(
                sub, xoff, top_conn_y + params.grid_step + params.cb_size / 2
            )
            # Line from switch down
            parent_group.append(
                draw.Line(
                    xoff,
                    top_conn_y + params.grid_step + params.cb_size,
                    xoff,
                    top_conn_y + params.grid_step + params.cb_size + params.grid_step,
                )
            )
            mark_grid_point(
                sub,
                xoff,
                top_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )
            sub.connection_points[bay.elementTieConnection] = (
                xoff,
                top_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )

    # BOTTOM ELEMENT
    if (
        bay.elementOtherBus is SwitchType.NOBUS
        or bay.elementOtherBus is SwitchType.EMPTY
    ):
        # No bottom element or bus
        return parent_group

    middle_conn_y = 2 * (params.grid_step + params.cb_size + params.grid_step)
    if bay.elementOtherBus is SwitchType.DIRECT:
        # Direct connection, just a line
        parent_group.append(
            draw.Line(
                xoff,
                middle_conn_y,
                xoff,
                middle_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )
        )
        mark_grid_point(
            sub,
            xoff,
            middle_conn_y + params.grid_step + params.cb_size + params.grid_step,
        )
    else:
        # Draw CB or isolator
        # Line to switch
        parent_group.append(
            draw.Line(xoff, middle_conn_y, xoff, middle_conn_y + params.grid_step)
        )
        mark_grid_point(sub, xoff, middle_conn_y + params.grid_step)
        # Draw the switch
        draw_switch(
            xoff,
            middle_conn_y + params.grid_step + params.cb_size / 2,
            parent_group,
            bay.elementOtherBus,
            "vertical",
            params=params,
        )
        mark_grid_point(
            sub, xoff, middle_conn_y + params.grid_step + params.cb_size / 2
        )
        # Line from switch to bottom bus
        parent_group.append(
            draw.Line(
                xoff,
                middle_conn_y + params.grid_step + params.cb_size,
                xoff,
                middle_conn_y + params.grid_step + params.cb_size + params.grid_step,
            )
        )
        mark_grid_point(
            sub,
            xoff,
            middle_conn_y + params.grid_step + params.cb_size + params.grid_step,
        )

    # Draw bottom bus with same width as top bus for consistency
    bottom_bus_y = 3 * (params.grid_step + params.cb_size + params.grid_step)
    parent_group.append(
        draw.Line(-50 + xoff, bottom_bus_y, 50 + xoff, bottom_bus_y, stroke_width=5)
    )

    # Mark bottom busbar in grid
    for i in range(int(100 / params.grid_step) + 1):
        mark_grid_point(
            sub, -50 + xoff + i * params.grid_step, bottom_bus_y, weight=BUSBAR_WEIGHT
        )
    mark_grid_point(sub, xoff, bottom_bus_y, weight=BUSBAR_WEIGHT)

    # Handle bottom bus identifier - only draw if its the first bay on the bus
    if (
        bay.neighbour is None
        or not isinstance(bay.neighbour, BreakerAndHalfBay)
        or bay.neighbour.elementOtherBus is SwitchType.NOBUS
    ):
        parent_group.append(
            draw.Text(
                bay.other_bus_name,
                x=-55 + xoff,
                y=bottom_bus_y + 8,
                font_size=BUS_LABEL_FONT_SIZE,
                anchor="end",
                stroke_width=0,
            )
        )

    # Mark grid points for element above and lines only if elementAbove exists
    if (
        bay.elementAbove is not SwitchType.EMPTY
        and bay.elementAbove is not SwitchType.NOBUS
    ):
        top_y = -1 * (params.grid_step + params.cb_size + params.grid_step)
        cb_y = -1 * (params.grid_step + params.cb_size / 2)

        # Always mark the busbar connection
        mark_grid_point(sub, xoff, 0)  # Busbar connection

        if bay.elementAbove is SwitchType.DIRECT:
            # For direct connections, just mark the end point
            mark_grid_point(sub, xoff, top_y)  # End of line
        else:
            # For CB/isolator, mark all points of the switch square:
            # Top of CB
            mark_grid_point(sub, xoff, -1 * params.grid_step)
            # Left side of CB
            mark_grid_point(sub, xoff - params.cb_size / 2, cb_y)
            # CB center
            mark_grid_point(sub, xoff, cb_y)
            # Right side of CB
            mark_grid_point(sub, xoff + params.cb_size / 2, cb_y)
            # Bottom of CB
            mark_grid_point(sub, xoff, -1 * (params.grid_step + params.cb_size))
            # End of line
            mark_grid_point(sub, xoff, top_y)

        # Mark intermediate points between busbar and switch
        for i in range(1, int(abs(cb_y) / params.grid_step)):
            mark_grid_point(sub, xoff, -i * params.grid_step)

        # Mark intermediate points between switch and end of line
        switch_bottom = -1 * (params.grid_step + params.cb_size)
        for i in range(1, int(abs(top_y - switch_bottom) / params.grid_step) + 1):
            y_pos = switch_bottom - i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)
        mark_grid_point(
            sub, xoff, -1 * (params.grid_step + params.cb_size + params.grid_step)
        )

    # Mark grid points for top element and lines only if elementBelow exists
    if (
        bay.elementBelow is not SwitchType.NOBUS
        and bay.elementBelow is not SwitchType.EMPTY
    ):
        bottom_y = params.grid_step + params.cb_size + params.grid_step
        cb_y = params.grid_step + params.cb_size / 2

        # Always mark the busbar connection
        mark_grid_point(sub, xoff, 0)  # Busbar connection

        if bay.elementBelow is SwitchType.DIRECT:
            # For direct connections, just mark the end point
            mark_grid_point(sub, xoff, bottom_y)  # End of line
        else:
            # For CB/isolator, mark all points of the switch square:
            # Top of CB
            mark_grid_point(sub, xoff, params.grid_step)
            # Left side of CB
            mark_grid_point(sub, xoff - params.cb_size / 2, cb_y)
            # CB center
            mark_grid_point(sub, xoff, cb_y)
            # Right side of CB
            mark_grid_point(sub, xoff + params.cb_size / 2, cb_y)
            # Bottom of CB
            mark_grid_point(sub, xoff, params.grid_step + params.cb_size)
            # End of line
            mark_grid_point(sub, xoff, bottom_y)

        # Mark intermediate points between busbar and switch
        for i in range(1, int(cb_y / params.grid_step)):
            mark_grid_point(sub, xoff, i * params.grid_step)

        # Mark intermediate points between switch and end of line
        switch_bottom = params.grid_step + params.cb_size
        for i in range(1, int(abs(bottom_y - switch_bottom) / params.grid_step) + 1):
            y_pos = switch_bottom + i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)
        mark_grid_point(sub, xoff, params.grid_step + params.cb_size + params.grid_step)

    # Mark grid points for middle element and lines only if elementTie exists
    if (
        bay.elementTie is not SwitchType.NOBUS
        and bay.elementTie is not SwitchType.EMPTY
    ):
        switch_y = top_conn_y + params.grid_step + params.cb_size / 2
        bottom_y = top_conn_y + params.grid_step + params.cb_size + params.grid_step

        # Always mark the top connection
        mark_grid_point(sub, xoff, top_conn_y)  # Top connection

        if bay.elementTie is SwitchType.DIRECT:
            # For direct connections, just mark the end point
            mark_grid_point(sub, xoff, bottom_y)  # End of line
        else:
            # For CB/isolator, mark all points of the switch square:
            # Top of CB
            mark_grid_point(sub, xoff, top_conn_y + params.grid_step)
            # Left side of CB
            mark_grid_point(sub, xoff - params.cb_size / 2, switch_y)
            # CB center
            mark_grid_point(sub, xoff, switch_y)
            # Right side of CB
            mark_grid_point(sub, xoff + params.cb_size / 2, switch_y)
            # Bottom of CB
            mark_grid_point(sub, xoff, top_conn_y + params.grid_step + params.cb_size)
            # End of line
            mark_grid_point(sub, xoff, bottom_y)

        # Mark intermediate points between top connection and switch
        for i in range(1, int(abs(switch_y - top_conn_y) / params.grid_step)):
            y_pos = top_conn_y + i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)

        # Mark intermediate points between switch and end of line
        switch_bottom = top_conn_y + params.grid_step + params.cb_size
        for i in range(1, int(abs(bottom_y - switch_bottom) / params.grid_step) + 1):
            y_pos = switch_bottom + i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)
        mark_grid_point(
            sub, xoff, top_conn_y + params.grid_step + params.cb_size + params.grid_step
        )

    # Mark grid points for bottom element and lines only if elementOtherBus exists
    if (
        bay.elementOtherBus is not SwitchType.NOBUS
        and bay.elementOtherBus is not SwitchType.EMPTY
    ):
        switch_y = middle_conn_y + params.grid_step + params.cb_size / 2
        bottom_y = middle_conn_y + params.grid_step + params.cb_size + params.grid_step

        # Always mark the middle connection
        mark_grid_point(sub, xoff, middle_conn_y)  # Middle connection

        if bay.elementOtherBus is SwitchType.DIRECT:
            # For direct connections, just mark the end point
            mark_grid_point(sub, xoff, bottom_y)  # End of line
        else:
            # For CB/isolator, mark all points of the switch square:
            # Top of CB
            mark_grid_point(sub, xoff, middle_conn_y + params.grid_step)
            # Left side of CB
            mark_grid_point(sub, xoff - params.cb_size / 2, switch_y)
            # CB center
            mark_grid_point(sub, xoff, switch_y)
            # Right side of CB
            mark_grid_point(sub, xoff + params.cb_size / 2, switch_y)
            # Bottom of CB
            mark_grid_point(
                sub, xoff, middle_conn_y + params.grid_step + params.cb_size
            )
            # End of line
            mark_grid_point(sub, xoff, bottom_y)

        # Mark intermediate points between middle connection and switch
        for i in range(1, int(abs(switch_y - middle_conn_y) / params.grid_step)):
            y_pos = middle_conn_y + i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)

        # Mark intermediate points between switch and bottom bus
        switch_bottom = middle_conn_y + params.grid_step + params.cb_size
        for i in range(1, int(abs(bottom_y - switch_bottom) / params.grid_step) + 1):
            y_pos = switch_bottom + i * params.grid_step
            mark_grid_point(sub, xoff, y_pos)
        mark_grid_point(
            sub,
            xoff,
            middle_conn_y + params.grid_step + params.cb_size + params.grid_step,
        )

    return parent_group


# make substation group
def get_substation_group(sub: Substation, colour="blue", rotation=0):
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    dg = draw.Group(
        stroke=colour,
        stroke_width=2,
        transform=f"rotate({rotation}, {center_x}, {center_y})",
    )
    for i, bay in enumerate(sub.bays):
        xoff = 50 * i
        if isinstance(bay, SingleSwitchedBay):
            dg = draw_single_switched_bay(
                xoff,
                parent_group=dg,
                bay=bay,
                sub=sub,
                params=drawing_params,
            )
        elif isinstance(bay, BreakerAndHalfBay):
            dg = draw_breaker_and_half_bay(
                xoff,
                parent_group=dg,
                bay=bay,
                sub=sub,
                params=drawing_params,
            )
        elif isinstance(bay, DoubleSwitchedBay):
            dg = draw_double_switched_bay(
                xoff,
                parent_group=dg,
                bay=bay,
                sub=sub,
                params=drawing_params,
            )

    return dg


substation_groups = {
    sub.name: get_substation_group(sub, rotation=sub.rotation) for sub in substations
}

# Scale and position the substations to preserve relative distances
# with maximum distance of approximately 500 units
print("-" * 50)
print("Original UTM coordinates:")
print("-" * 50)
for sub in substations:
    print(f"{sub.name:<5}: ({sub.x:.2f}, {sub.y:.2f})")


# Find min/max values for scaling
min_x = min(sub.x for sub in substations)
max_x = max(sub.x for sub in substations)
min_y = min(sub.y for sub in substations)
max_y = max(sub.y for sub in substations)

# Calculate ranges
x_range = max_x - min_x
y_range = max_y - min_y

# Determine scaling factor for both dimensions to fit within 1800 units (leaving 100px margin on each side)
scale_factor_x = MAP_DIMS * 0.9 / x_range if x_range > 0 else 1
scale_factor_y = MAP_DIMS * 0.9 / y_range if y_range > 0 else 1

# Use the smaller scaling factor to maintain aspect ratio
scale_factor = min(scale_factor_x, scale_factor_y)

print("\n" + "-" * 50)
print(f"Scaling Information:")
print("-" * 50)
print(f"Scaling factor: {scale_factor:.6f}")
print(f"X range: {x_range:.2f}, Y range: {y_range:.2f}")
print(f"Max dimension: {max(x_range, y_range):.2f}")

# Apply scaling and translation with y-axis flipped (since SVG y increases downward)
print("\nScaled coordinates:")
for sub in substations:
    # Scale coordinates to use more of the MAP_DIMSxMAP_DIMS area
    # Use a larger scaling factor to spread out across the canvas
    # Use 20% margin on all sides ( MAP_DIMS * 0.2)
    sub.scaled_x = (sub.x - min_x) * scale_factor * 4 + (MAP_DIMS * 0.2)
    # Invert the y-axis to match geographic orientation (north at top)
    sub.scaled_y = (sub.y - min_y) * scale_factor * 4 + (MAP_DIMS * 0.2)
    print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")


# Use NetworkX to locate the substations with balanced spacing
import networkx as nx

# Create a dictionary of our substations with their scaled coordinates
nodes = [sub.name for sub in substations]
initial_pos = {sub.name: (sub.scaled_x, sub.scaled_y) for sub in substations}

# Create a graph for the spring layout
G = nx.Graph()
G.add_nodes_from(nodes)

# Find connections between substations to create edges
connection_map = {}  # dict of conn_name -> list of sub_names
for sub in substations:
    for bay in sub.bays:
        connections = []
        if hasattr(bay, "elementAboveConnection"):
            connections.append(bay.elementAboveConnection)
        if hasattr(bay, "elementBelowConnection"):
            connections.append(bay.elementBelowConnection)
        if hasattr(bay, "elementOtherBusConnection"):
            connections.append(bay.elementOtherBusConnection)
        if hasattr(bay, "elementTieConnection"):
            connections.append(bay.elementTieConnection)

        for conn_name in connections:
            if conn_name:
                if conn_name not in connection_map:
                    connection_map[conn_name] = []
                if sub.name not in connection_map[conn_name]:
                    connection_map[conn_name].append(sub.name)

# Add edges for substations sharing a connection
for conn_name, sub_names in connection_map.items():
    if len(sub_names) > 1:
        for i in range(len(sub_names)):
            for j in range(i + 1, len(sub_names)):
                G.add_edge(sub_names[i], sub_names[j])

# Calculate average distance for setting spring layout parameter `k`
if G.number_of_edges() > 0:
    distances = [
        np.linalg.norm(np.array(initial_pos[u]) - np.array(initial_pos[v]))
        for u, v in G.edges()
    ]
    avg_distance = np.mean(distances)
else:
    avg_distance = 0

# Apply a weak spring force to gently adjust positions
if G.number_of_edges() > 0 and avg_distance > 0:
    print("\nApplying spring layout to adjust substation positions...")
    # Run the spring layout to get an idealized layout
    spring_pos = nx.spring_layout(
        G, pos=initial_pos, iterations=3, k=avg_distance, seed=0
    )
    final_pos = spring_pos
else:
    final_pos = initial_pos

substation_map = {sub.name: sub for sub in substations}

# # Find min/max values to normalize the layout results
min_x_pos = min(pos[0] for pos in final_pos.values())
max_x_pos = max(pos[0] for pos in final_pos.values())
min_y_pos = min(pos[1] for pos in final_pos.values())
max_y_pos = max(pos[1] for pos in final_pos.values())

# # Calculate ranges for normalization
x_pos_range = max_x_pos - min_x_pos
y_pos_range = max_y_pos - min_y_pos

# Scale the positions to fit within a MAP_DIMSxMAP_DIMS area with margins
print("\nNormalizing coordinates to MAP_DIMSxMAP_DIMS area")
for name, coords in final_pos.items():
    sub = substation_map[name]
    drawing_params = DrawingParams()

    # Normalize positions to 0-1 range
    norm_x = (coords[0] - min_x_pos) / x_pos_range if x_pos_range > 0 else 0.5
    norm_y = (coords[1] - min_y_pos) / y_pos_range if y_pos_range > 0 else 0.5

    # Scale to fit within the full MAP_DIMSxMAP_DIMS canvas with 10% unit margins on each side
    _15 = MAP_DIMS * 0.15
    _90 = MAP_DIMS * 0.9
    raw_x = norm_x * (_90 - _15) + _15
    raw_y = norm_y * (_90 - _15) + _15

    # Get the bounding box to calculate busbar position offset
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)
    height = max_y - min_y

    # Calculate the offset from center to busbar position (y=0 in local coordinates)
    # The busbar is at y=0, and the center of bounding box is at (min_y + height/2)
    busbar_to_center_offset = 0 - (min_y + height / 2)

    # Snap to grid - ensure the busbar position aligns exactly with the grid
    _test = drawing_params.grid_step
    sub.scaled_x = round(raw_x / _test) * _test

    # First snap the center to grid
    snapped_center_y = round(raw_y / _test) * _test

    # Then adjust to ensure the busbar (not the center) aligns with the grid
    # Calculate where the busbar would be after snapping the center
    busbar_y = snapped_center_y + busbar_to_center_offset

    # Snap the busbar position to grid
    snapped_busbar_y = round(busbar_y / _test) * _test

    # Adjust the center position to make the busbar position exactly on grid
    sub.scaled_y = snapped_busbar_y - busbar_to_center_offset


print("\nFinal SVG coordinates after network balancing:")
for sub in substations:
    print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")


# draw at the scaled x and y locations
# In SVG, y increases downward, so we'll invert the y-coordinate
for sub in substations:
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)
    width = max_x - min_x
    height = max_y - min_y

    # The desired center is (sub.scaled_x, MAP_DIMS - sub.scaled_y)
    # The group is drawn relative to (0,0).
    # The group's content bounding box is (min_x, min_y) to (max_x, max_y).
    # The center of the group's content is (min_x + width/2, min_y + height/2).
    # We want to place the group such that its center aligns with the desired center.
    # The `draw.Use` x,y is the top-left of the group's coordinate system (0,0).
    # So we need to offset the placement.
    use_x = sub.scaled_x - (min_x + width / 2)
    use_y = (MAP_DIMS - sub.scaled_y) - (min_y + height / 2)
    # Set the use_x and use_y to the scaled_x and scaled_y
    sub.use_x = use_x
    sub.use_y = use_y


# # Iteratively adjust substation positions to avoid overlaps
# MAX_ITERATIONS = 100
# MIN_DISTANCE = 200
# for i in range(MAX_ITERATIONS):
#     moved = False
#     for sub1, sub2 in combinations(substations, 2):
#         # get bboxes and account for rotation
#         min_x1, min_y1, max_x1, max_y1 = sub1.get_drawing_bbox(drawing_params)
#         if abs(sub1.rotation % 180) == 90:
#             w, h = max_x1 - min_x1, max_y1 - min_y1
#             cx, cy = (min_x1 + max_x1) / 2, (min_y1 + max_y1) / 2
#             min_x1, max_x1 = cx - h / 2, cx + h / 2
#             min_y1, max_y1 = cy - w / 2, cy + w / 2
#         bbox1 = {
#             "min_x": sub1.use_x + min_x1,
#             "max_x": sub1.use_x + max_x1,
#             "min_y": sub1.use_y + min_y1,
#             "max_y": sub1.use_y + max_y1,
#         }

#         min_x2, min_y2, max_x2, max_y2 = sub2.get_drawing_bbox(drawing_params)
#         if abs(sub2.rotation % 180) == 90:
#             w, h = max_x2 - min_x2, max_y2 - min_y2
#             cx, cy = (min_x2 + max_x2) / 2, (min_y2 + max_y2) / 2
#             min_x2, max_x2 = cx - h / 2, cx + h / 2
#             min_y2, max_y2 = cy - w / 2, cy + w / 2
#         bbox2 = {
#             "min_x": sub2.use_x + min_x2,
#             "max_x": sub2.use_x + max_x2,
#             "min_y": sub2.use_y + min_y2,
#             "max_y": sub2.use_y + max_y2,
#         }

#         # calculate distance between bounding boxes
#         dx = max(0, bbox1["min_x"] - bbox2["max_x"]) + max(
#             0, bbox2["min_x"] - bbox1["max_x"]
#         )
#         dy = max(0, bbox1["min_y"] - bbox2["max_y"]) + max(
#             0, bbox2["min_y"] - bbox1["max_y"]
#         )
#         distance = math.sqrt(dx**2 + dy**2)

#         if distance < MIN_DISTANCE:
#             moved = True
#             # Vector between centers
#             center1_x = (bbox1["min_x"] + bbox1["max_x"]) / 2
#             center1_y = (bbox1["min_y"] + bbox1["max_y"]) / 2
#             center2_x = (bbox2["min_x"] + bbox2["max_x"]) / 2
#             center2_y = (bbox2["min_y"] + bbox2["max_y"]) / 2

#             vec_x = center2_x - center1_x
#             vec_y = center2_y - center1_y
#             dist_centers = math.sqrt(vec_x**2 + vec_y**2)

#             # Avoid division by zero if centers are identical
#             if dist_centers == 0:
#                 vec_x, vec_y, dist_centers = 1, 0, 1

#             # Amount to move
#             move_amount = MIN_DISTANCE - distance

#             # Normalize vector and move substations
#             norm_vec_x = vec_x / dist_centers
#             norm_vec_y = vec_y / dist_centers
#             move_x = norm_vec_x * move_amount / 2
#             move_y = norm_vec_y * move_amount / 2

#             sub1.use_x -= move_x
#             sub1.use_y -= move_y
#             sub2.use_x += move_x
#             sub2.use_y += move_y

#     if not moved:
#         print(f"Overlap resolution converged after {i + 1} iterations.")
#         break
# else:
#     print(f"Overlap resolution did not converge after {MAX_ITERATIONS} iterations.")


# Re-snap positions to the grid after overlap avoidance by snapping the center
grid_step = drawing_params.grid_step
for sub in substations:
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)
    width = max_x - min_x
    height = max_y - min_y
    local_center_x = min_x + width / 2
    local_center_y = min_y + height / 2

    # Get current global center
    global_center_x = sub.use_x + local_center_x
    global_center_y = sub.use_y + local_center_y

    # Snap global center to grid
    snapped_center_x = round(global_center_x / grid_step) * grid_step
    snapped_center_y = round(global_center_y / grid_step) * grid_step

    # Recalculate use_x/y from snapped center
    sub.use_x = snapped_center_x - local_center_x
    sub.use_y = snapped_center_y - local_center_y


# Now draw the substations at their final, adjusted positions
for sub in substations:
    d.append(draw.Use(substation_groups[sub.name], sub.use_x, sub.use_y))


# TESTING - drawing connections via a permissive grid and shortest path
# Create a 2D grid of points with 25 step size
grid_size = MAP_DIMS
step = 25
num_steps = grid_size // step + 1
points = [[0 for _ in range(num_steps)] for _ in range(num_steps)]

# Transfer grid points from each substation to the global MAP_DIMSxMAP_DIMS grid
for sub in substations:
    # Calculate top-left position of the substation in SVG coordinates
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)

    # Calculate center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Get rotation angle in radians
    rotation_rad = math.radians(sub.rotation)

    # Convert local substation coordinates to global SVG coordinates
    for local_x, local_y in sub.grid_points:
        # Apply rotation around the center of the substation's bounding box
        # First, translate point to origin (relative to center)
        rel_x = local_x - center_x
        rel_y = local_y - center_y

        # Then rotate
        rotated_x = rel_x * math.cos(rotation_rad) - rel_y * math.sin(rotation_rad)
        rotated_y = rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)

        # Translate back
        rotated_local_x = rotated_x + center_x
        rotated_local_y = rotated_y + center_y

        # Calculate global coordinates (SVG space) using the final use_x and use_y
        global_x = sub.use_x + rotated_local_x
        global_y = sub.use_y + rotated_local_y

        # Convert to grid indices
        grid_x = int(round(global_x / step))
        grid_y = int(round(global_y / step))

        # Mark the point as True if within bounds
        if 0 <= grid_x < num_steps and 0 <= grid_y < num_steps:
            points[grid_x][grid_y] = sub.grid_points.get((local_x, local_y))


# for each substation, find the two tuples which have the same connection key value
all_connections = {}
for sub in substations:
    # Get substation's drawing bounding box for coordinate transformation
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)

    # Calculate center of the bounding box for rotation
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    rotation_rad = math.radians(sub.rotation)

    for linedef, local_coords in sub.connection_points.items():
        if linedef == "":
            # todo raise warning
            continue

        local_x, local_y = local_coords

        # Apply rotation to connection points, same as for grid points
        rel_x = local_x - center_x
        rel_y = local_y - center_y
        rotated_x = rel_x * math.cos(rotation_rad) - rel_y * math.sin(rotation_rad)
        rotated_y = rel_x * math.sin(rotation_rad) + rel_y * math.cos(rotation_rad)
        rotated_local_x = rotated_x + center_x
        rotated_local_y = rotated_y + center_y

        # Transform local coordinates to global MAP_DIMSxMAP_DIMS reference using final positions
        global_coords = (sub.use_x + rotated_local_x, sub.use_y + rotated_local_y)

        if linedef not in all_connections:
            all_connections[linedef] = [global_coords]
        else:
            all_connections[linedef].append(global_coords)


# for each connection, calculate the euclidian distance then convert into a list of dicts,
# ... from shortest to longest
def _distance(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


# Sort connections by distance for better visualization
# filter out connections with less than 2 points
all_connections = {key: val for key, val in all_connections.items() if len(val) == 2}
all_connections = {
    k: v
    for k, v in sorted(all_connections.items(), key=lambda item: _distance(*item[1]))
    if len(v) == 2
}

for connection in all_connections.values():
    # Only process connections with exactly 2 points
    if len(connection) != 2:
        print(f"Skipping connection with {len(connection)} points instead of 2")
        continue

    # Convert global coordinates to grid indices
    start_point = (int(connection[0][0] // step), int(connection[0][1] // step))
    end_point = (int(connection[1][0] // step), int(connection[1][1] // step))

    # Ensure grid indices are within bounds
    start_point = (
        max(0, min(start_point[0], num_steps - 1)),
        max(0, min(start_point[1], num_steps - 1)),
    )
    end_point = (
        max(0, min(end_point[0], num_steps - 1)),
        max(0, min(end_point[1], num_steps - 1)),
    )

    # set start and end coord to a weight of 0
    points[start_point[0]][start_point[1]] = 0
    points[end_point[0]][end_point[1]] = 0

    print(f"Finding path from {start_point} to {end_point}")

    try:
        # Find path using A* with weights - convert tuples to ensure right format
        start_node = (int(start_point[0]), int(start_point[1]))
        end_node = (int(end_point[0]), int(end_point[1]))
        path, points = findpath.run_gridsearch(start_node, end_node, points)

        # Draw a line between each pair of consecutive path points, converting grid indices back to drawing coordinates
        if len(path) > 1:
            print(f"Drawing path with {len(path)} points")
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                d.append(
                    draw.Line(
                        start[0] * step,
                        start[1] * step,
                        end[0] * step,
                        end[1] * step,
                        stroke="blue",
                        stroke_width=2,
                    )
                )
    except Exception as e:
        print(f"Error finding path: {e}")
        continue


# Draw circles at each grid point
# for i in range(num_steps):
#     for j in range(num_steps):
#         if points[i][j] >= 25:
#             col = "red"
#         elif points[i][j] > 0:
#             col = "orange"
#         else:
#             # col = "green"
#             continue
#         # print(points[i][j])
#         x, y = i * step, j * step
#         d.append(draw.Circle(x, y, 3, fill=col))

# and draw circles at connection pointsd
for connection in all_connections.values():
    for point in connection:
        d.append(draw.Circle(point[0], point[1], 5, fill="blue"))


# Draw substation titles
max_search_radius_grid = TITLE_MAX_SEARCH_RADIUS_PX // step

for sub in substations:
    if not sub.title:
        continue

    # Estimate text bounding box size in grid units
    # Using a heuristic for average character width (font size * 0.6)
    text_width_px = len(sub.title) * TITLE_FONT_SIZE * 0.6
    text_height_px = TITLE_FONT_SIZE
    text_width_grid = int(math.ceil(text_width_px / step)) + 1
    text_height_grid = int(math.ceil(text_height_px / step)) + 1

    # Calculate substation center in global SVG coordinates
    min_x, min_y, max_x, max_y = sub.get_drawing_bbox(drawing_params)
    width = max_x - min_x
    height = max_y - min_y
    local_center_x = min_x + width / 2
    local_center_y = min_y + height / 2
    center_x = sub.use_x + local_center_x
    center_y = sub.use_y + local_center_y

    center_gx = int(round(center_x / step))
    center_gy = int(round(center_y / step))

    # Search for a free spot for the title
    found_spot = False
    # Start search from a small radius to find closest spot
    for r_grid in range(1, max_search_radius_grid + 1):
        # Check points on the circle of radius r_grid
        for angle_deg in range(0, 360, 15):  # Check every 15 degrees
            angle_rad = math.radians(angle_deg)
            # This is the anchor point for the text (bottom-left)
            gx = center_gx + int(r_grid * math.cos(angle_rad))
            gy = center_gy + int(r_grid * math.sin(angle_rad))

            # Check if the bounding box for the text is clear
            is_clear = True
            for i in range(text_width_grid):
                for j in range(text_height_grid):
                    check_gx = gx + i
                    check_gy = (
                        gy - j
                    )  # Text is drawn with y as baseline, so box goes up
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
                d.append(
                    draw.Text(
                        sub.title,
                        font_size=TITLE_FONT_SIZE,
                        x=title_x,
                        y=title_y,
                        fill="black",
                    )
                )
                # Mark the area as occupied
                for i in range(text_width_grid):
                    for j in range(text_height_grid):
                        mark_gx = gx + i
                        mark_gy = gy - j
                        if 0 <= mark_gx < num_steps and 0 <= mark_gy < num_steps:
                            points[mark_gx][mark_gy] = 1

                found_spot = True
                break
        if found_spot:
            break

    if not found_spot:
        print(f"Warning: Could not find a free spot for title of substation {sub.name}")


# draw bus
d.save_svg("example.svg")

# Generate locations data for javascript
locations_data = []
for sub in substations:
    # Use the substation's title for the Leaflet map
    title = sub.title if sub.title else sub.name
    # The center of the substation in Leaflet coordinates [y, x]
    leaflet_y = MAP_DIMS - sub.use_y
    leaflet_x = sub.use_x
    locations_data.append(f'{{ title: "{title}", coords: [{leaflet_y}, {leaflet_x}] }}')

locations_json_string = "[\n        " + ",\n        ".join(locations_data) + "\n    ]"

# Generate index.html from template
with open("example.svg", "r", encoding="utf-8") as f:
    svg_content = f.read()

with open(
    r"C:\Users\DamienVermeer\Downloads\index.template.html", "r", encoding="utf-8"
) as f:
    template_content = f.read()

# The SVG content needs to be embedded within Javascript backticks, so we need to escape backticks in the SVG content itself.
svg_content_escaped = svg_content.replace("`", "\\`")

html_content = template_content.replace("%%SVG_CONTENT%%", svg_content_escaped)
html_content = html_content.replace("%%LOCATIONS_DATA%%", locations_json_string)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("\nGenerated index.html with embedded SVG.")
