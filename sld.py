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
import utm

# modifying so bus is included in bay definition


class SwitchType(Enum):
    NOBUS = auto()
    BUSTIE = auto()
    EMPTY = auto()
    DIRECT = auto()
    CB = auto()
    ISOL = auto()


@dataclass
class BaseBay:
    bus_name: str
    element: SwitchType = SwitchType.CB
    flip: bool = True
    neighbour: "BaseBay" = None


@dataclass
class SingleSwitchedBay(BaseBay):
    pass


@dataclass
class DoubleSwitchedBay(BaseBay):
    other_bus_name: str = ""
    element2: SwitchType = SwitchType.CB


@dataclass
class BreakerAndHalfBay(DoubleSwitchedBay):
    element3: SwitchType = SwitchType.CB


@dataclass
class BusTieBay(BaseBay):
    tie_bus_name: str = ""


@dataclass
class Substation:
    name: str
    lat: float
    long: float
    bays: list[BaseBay] = field(default_factory=list)
    scaled_x: float = 0
    scaled_y: float = 0

    def __post_init__(self):
        self.x, self.y, _, _ = utm.from_latlon(self.lat, self.long)

    def add_bay(self, bay: BaseBay) -> None:
        self.bays.append(bay)
        # set each bay to its left neighbour
        for i, _ in enumerate(self.bays):
            if i == 0:
                continue
            self.bays[i].neighbour = self.bays[i - 1]


# testing - single switched sub
arts_220kv = Substation("ARTS", lat=-37.163788, long=143.245555)
arts_220kv.add_bay(
    BreakerAndHalfBay(
        "ARTS",
        element=SwitchType.ISOL,
        element2=SwitchType.CB,
        element3=SwitchType.ISOL,
        other_bus_name="?",
    )
)
arts_220kv.add_bay(
    BreakerAndHalfBay(
        "?",
        element=SwitchType.ISOL,
        element2=SwitchType.CB,
        element3=SwitchType.ISOL,
        other_bus_name="?",
    )
)
wbts_220kv = Substation("WBTS", lat=-37.355738, long=143.606139)
wbts_220kv.add_bay(
    SingleSwitchedBay(
        "WBTS",
        element=SwitchType.ISOL,
    )
)
wbts_220kv.add_bay(
    SingleSwitchedBay(
        "?",
        element=SwitchType.CB,
    )
)
wbts_220kv.add_bay(
    SingleSwitchedBay(
        "?",
        element=SwitchType.CB,
    )
)
wbts_220kv.add_bay(
    SingleSwitchedBay(
        "?",
        element=SwitchType.ISOL,
    )
)
cwts_220kv = Substation("CWTS", lat=-37.120842, long=143.154319)
cwts_220kv.add_bay(
    BreakerAndHalfBay(
        "CWTS",
        element=SwitchType.CB,
        element2=SwitchType.CB,
        element3=SwitchType.CB,
        other_bus_name="?",
    )
)
cwts_220kv.add_bay(
    BreakerAndHalfBay(
        "?",
        element=SwitchType.CB,
        element2=SwitchType.DIRECT,
        element3=SwitchType.CB,
        other_bus_name="?",
    )
)
bats_220kv = Substation("BATS", lat=-37.567543, long=143.921133)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2 (BATS)",
        element=SwitchType.CB,
        element2=SwitchType.ISOL,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.ISOL,
        element2=SwitchType.CB,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.CB,
        element2=SwitchType.CB,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.CB,
        element2=SwitchType.CB,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.CB,
        element2=SwitchType.EMPTY,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.EMPTY,
        element2=SwitchType.CB,
        other_bus_name="1",
    )
)
bats_220kv.add_bay(
    DoubleSwitchedBay(
        "2",
        element=SwitchType.CB,
        element2=SwitchType.CB,
        other_bus_name="1",
    )
)

# ss_sub.add_bay(
#     DoubleSwitchedBay(
#         "2", element=SwitchType.NOBUS, element2=SwitchType.ISOL, other_bus_name="3"
#     )
# )
# ss_sub.add_bay(
#     DoubleSwitchedBay(
#         "2", element=SwitchType.NOBUS, element2=SwitchType.NOBUS, other_bus_name="3"
#     )
# )
# ss_sub.add_bay(
#     DoubleSwitchedBay(
#         "2", element=SwitchType.NOBUS, element2=SwitchType.CB, other_bus_name="4A"
#     )
# )


# # create a simple single switched sub
# new_sub = Substation(name="sub1")
# bus1 = Bus(name="4A")
# new_sub.add_bus(bus1)
# # add 3 CBs and an isol
# bus1.add_bay(SingleSwitchedBay(name="bay1"))
# bus1.add_bay(SingleSwitchedBay(name="bay2"))
# bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
# bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
# bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
# bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))

# bus2 = Bus(name="2", is_planar=True)
# new_sub.add_bus(bus2)
# bus2.add_bay(SingleSwitchedBay(name="bay4", switch=SwitchType.ISOL))
# bus3 = Bus(name="3", is_planar=True)
# new_sub.add_bus(bus3)
# # testing planar bustie
# bus1.add_bay(BusTieBay(name="bay3", tie_bus=bus2, switch=SwitchType.ISOL))
# bus2.add_bay(BusTieBay(name="bay5", tie_bus=bus3, switch=SwitchType.ISOL))
# bus3.add_bay(SingleSwitchedBay(name="bay6", switch=SwitchType.CB))

# # ------------------------------------------------

# double_switched_sub = Substation(name="sub2")
# dbus1 = Bus(name="4A")
# dbus2 = Bus(name="1")
# double_switched_sub.add_bus(dbus1)
# double_switched_sub.add_bus(dbus2)
# dbus1.add_bay(
#     DoubleSwitchedBay(
#         name="bay1", switch=SwitchType.ISOL, switch2=SwitchType.CB, other_bus=dbus2
#     )
# )
# dbus1.add_bay(
#     DoubleSwitchedBay(
#         name="bay2", switch=SwitchType.CB, switch2=SwitchType.ISOL, other_bus=dbus2
#     )
# )
# dbus1.add_bay(
#     DoubleSwitchedBay(
#         name="bay2", switch=SwitchType.CB, switch2=SwitchType.ISOL, other_bus=dbus2
#     )
# )


import drawsvg as draw

d = draw.Drawing(1500, 1500, origin="center")


def draw_switch(
    x: float,
    y: float,
    parent_group: draw.Group,
    switch_type: SwitchType,
    orientation: str = "vertical",
    rotation_angle: int = 45,
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
                draw.Rectangle(x - 10, y - 10, 20, 20, fill="white")
            )  # cb
        else:  # horizontal
            parent_group.append(
                draw.Rectangle(x - 10, y - 10, 20, 20, fill="white")
            )  # cb
    elif switch_type == SwitchType.ISOL:
        # Isolator is drawn as a rotated line
        if orientation == "vertical":
            parent_group.append(
                draw.Line(
                    x,
                    y - 15,
                    x,
                    y + 15,
                    stroke_width=2,
                    transform=f"rotate({rotation_angle}, {x}, {y})",
                )
            )
        else:  # horizontal
            parent_group.append(
                draw.Line(
                    x - 15,
                    y,
                    x + 15,
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
    if bay.element is SwitchType.NOBUS:
        return parent_group
    parent_group.append(draw.Line(-25 + xoff, 0, 25 + xoff, 0, stroke_width=5))

    # handle bus identifier - only draw if its the first bay on the bus
    if bay.neighbour is None or bay.neighbour.element is SwitchType.NOBUS:
        parent_group.append(
            draw.Text(bay.bus_name, x=-25 + xoff, y=-8, font_size=25, anchor="end")
        )

    if bay.element is SwitchType.EMPTY:
        # nothing else to do
        return parent_group

    elif bay.element is SwitchType.DIRECT:
        # Line off bus - but is longer than normal as no cb/isol
        parent_group.append(draw.Line(xoff, 0, xoff, ys * 60))
        return parent_group

    # else its either CB or isolator

    # Line off bus towards switch
    parent_group.append(draw.Line(xoff, 0, xoff, ys * 20))
    # Draw the switch at the correct position
    draw_switch(xoff, ys * 30, parent_group, bay.element, "vertical")
    # Line from switch towards feeder
    parent_group.append(draw.Line(xoff, ys * 40, xoff, ys * 60))

    return parent_group


def draw_double_switched_bay(
    xoff: float,
    parent_group: draw.Group,
    bay: DoubleSwitchedBay,
    first_bay: bool = False,
) -> draw.Group:
    """Draw a double switched bay at the given x offset and y direction.

    Args:
        xoff (float): x offset from the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (DoubleSwitchedBay): bay to draw the double switched bay for

    Returns:
        draw.Group: parent group with the double switched bay added
    """
    # top element is the same as single switched bay
    parent_group = draw_single_switched_bay(
        xoff,
        parent_group,
        bay,
    )

    # handle bottom element
    ys = 1 if bay.flip else -1
    # handle bus - only draw if its of type other than NOBUS
    if bay.element2 is SwitchType.NOBUS:
        return parent_group
    parent_group.append(draw.Line(-25 + xoff, 120, 25 + xoff, 120, stroke_width=5))

    # handle bus identifier - only draw if its the first bay on the bus
    if (
        bay.neighbour is None
        or not hasattr(bay.neighbour, "element2")
        or bay.neighbour.element2 is SwitchType.NOBUS
    ):
        parent_group.append(
            draw.Text(
                bay.other_bus_name, x=-25 + xoff, y=-8 + 150, font_size=25, anchor="end"
            )
        )

    if bay.element is SwitchType.EMPTY:
        # nothing else to do
        return parent_group

    elif bay.element is SwitchType.DIRECT:
        # Line off bus - but is longer than normal as no cb/isol
        parent_group.append(draw.Line(xoff, ys * 60, xoff, ys * 120))
        return parent_group

    # else its either CB or isolator

    # Line off bus towards switch
    parent_group.append(draw.Line(xoff, 40, xoff, ys * 80))
    # Draw the switch at the correct position
    draw_switch(xoff, ys * 90, parent_group, bay.element2, "vertical")
    # Line from switch towards feeder
    parent_group.append(draw.Line(xoff, ys * 100, xoff, ys * 120))

    return parent_group


def draw_cb_and_a_half_bay(
    xoff: float,
    parent_group: draw.Group,
    bay: BreakerAndHalfBay,
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

    # Add bus identifier for top bus if needed
    if bay.neighbour is None or not isinstance(bay.neighbour, BreakerAndHalfBay):
        parent_group.append(
            draw.Text(bay.bus_name, x=-55 + xoff, y=-8, font_size=25, anchor="end")
        )

    # TOP ELEMENT
    if bay.element is not SwitchType.NOBUS and bay.element is not SwitchType.EMPTY:
        if bay.element is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(draw.Line(xoff, 0, xoff, 40))
        else:
            # Draw CB or isolator
            # Line from top bus to switch
            parent_group.append(draw.Line(xoff, 0, xoff, 20))
            # Draw the switch
            draw_switch(xoff, 30, parent_group, bay.element, "vertical")
            # Line from switch down
            parent_group.append(draw.Line(xoff, 40, xoff, 60))

    # MIDDLE ELEMENT
    if bay.element2 is not SwitchType.NOBUS and bay.element2 is not SwitchType.EMPTY:
        if bay.element2 is SwitchType.DIRECT:
            # Direct connection, just a line
            parent_group.append(draw.Line(xoff, 60, xoff, 120))
        else:
            # Draw CB or isolator
            # Line to switch
            parent_group.append(draw.Line(xoff, 60, xoff, 80))
            # Draw the switch
            draw_switch(xoff, 90, parent_group, bay.element2, "vertical")
            # Line from switch down
            parent_group.append(draw.Line(xoff, 100, xoff, 120))

    # BOTTOM ELEMENT
    if bay.element3 is SwitchType.NOBUS or bay.element3 is SwitchType.EMPTY:
        # No bottom element or bus
        return parent_group

    if bay.element3 is SwitchType.DIRECT:
        # Direct connection, just a line
        parent_group.append(draw.Line(xoff, 120, xoff, 180))
    else:
        # Draw CB or isolator
        # Line to switch
        parent_group.append(draw.Line(xoff, 120, xoff, 140))
        # Draw the switch
        draw_switch(xoff, 150, parent_group, bay.element3, "vertical")
        # Line from switch to bottom bus
        parent_group.append(draw.Line(xoff, 160, xoff, 180))

    # Draw bottom bus with same width as top bus for consistency
    parent_group.append(draw.Line(-50 + xoff, 180, 50 + xoff, 180, stroke_width=5))

    # Handle bottom bus identifier - only draw if its the first bay on the bus
    if (
        bay.neighbour is None
        or not isinstance(bay.neighbour, BreakerAndHalfBay)
        or bay.neighbour.element3 is SwitchType.NOBUS
    ):
        parent_group.append(
            draw.Text(
                bay.other_bus_name, x=-55 + xoff, y=188, font_size=25, anchor="end"
            )
        )

    return parent_group


def flipline_bustie(
    busend_x: float, busend_y: float, parent_group: draw.Group, bay: BaseBay
) -> draw.Group:
    """Draw a CB or isolator in-line with a planar bus

    Args:
        busend_x (float): x coordinate of the end of the bus
        busend_y (float): y coordinate of the end of the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (Bay): bay to draw the CB or isolator for

    Returns:
        draw.Group: parent group with the CB or isolator added
    """
    # Line off bus towards switch
    parent_group.append(
        draw.Line(busend_x, busend_y, busend_x + 40, busend_y, stroke_width=5)
    )

    # Draw the switch
    switch_x = busend_x + 50  # Center point of the switch
    draw_switch(switch_x, busend_y, parent_group, bay.element, "horizontal")

    # Line from switch towards next bus section
    parent_group.append(
        draw.Line(busend_x + 60, busend_y, busend_x + 100, busend_y, stroke_width=5)
    )

    return parent_group


# make substation group
def get_substation_group(sub: Substation, colour="blue", rotation=0):
    dg = draw.Group(stroke=colour, stroke_width=2, transform=f"rotate({rotation})")
    for i, bay in enumerate(sub.bays):
        xoff = 50 * i
        if isinstance(bay, SingleSwitchedBay):
            dg = draw_single_switched_bay(xoff, dg, bay)
        elif isinstance(bay, BreakerAndHalfBay):
            dg = draw_cb_and_a_half_bay(xoff, dg, bay)
        elif isinstance(bay, DoubleSwitchedBay):
            dg = draw_double_switched_bay(xoff, dg, bay)

        # elif isinstance(bay, BusTieBay):
        #     dg = flipline_bustie(xoff, 0, dg, bay)

    # for i, bus in enumerate(k for k in sub.buses if k.is_planar):
    #     xoff += 100
    #     # draw main busbar
    #     bus_length = (bus.num_normal_bays - 1) * 50
    #     # draw small text at end of bus saying name
    #     dg.append(draw.Text(bus.name, x=-20 + xoff, y=-8, font_size=25, anchor="end"))
    #     dg.append(draw.Line(-20 + xoff, 0, bus_length + 20 + xoff, 0, stroke_width=5))

    #     if any(x for x in bus.bays if isinstance(x, DoubleSwitchedBay)):
    #         # draw 2nd busbar
    #         other_bus = [x for x in bus.bays if isinstance(x, DoubleSwitchedBay)][
    #             0
    #         ].other_bus
    #         bus_length = (bus.num_normal_bays - 1) * 50
    #         # draw small text at end of bus saying name
    #         dg.append(
    #             draw.Text(
    #                 other_bus.name,
    #                 x=-20 + xoff,
    #                 y=+120 + 32 - 8,
    #                 font_size=25,
    #                 anchor="end",
    #             )
    #         )
    #         dg.append(
    #             draw.Line(-20 + xoff, 120, bus_length + 20 + xoff, 120, stroke_width=5)
    #         )

    #     # draw bays
    #     for j, bay in enumerate(x for x in bus.bays if not isinstance(x, BusTieBay)):
    #         xoff += 50 if j > 0 else 0
    #         ys = 1 if bay.flip else -1
    #         # check type
    #         if isinstance(bay, SingleSwitchedBay):
    #             dg = draw_single_switched_bay(xoff, ys, dg, bay)
    #         elif isinstance(bay, DoubleSwitchedBay):
    #             dg = draw_double_switched_bay(xoff, dg, bay)
    #             bottom_buses.append(bay.other_bus)

    #     # TODO handle non planar bus tie
    #     for j, bustie in enumerate(x for x in bus.bays if isinstance(x, BusTieBay)):
    #         if bus.is_planar and bustie.tie_bus.is_planar:
    #             dg = flipline_bustie(xoff, 0, dg, bustie)

    # draw bus tie which is a CB in line with this bus
    # fill in non-planar buses

    return dg


d_arts = get_substation_group(arts_220kv, rotation=0)
d_wbts = get_substation_group(wbts_220kv, rotation=0)
d_cwts = get_substation_group(cwts_220kv, rotation=0)
d_bats = get_substation_group(bats_220kv, rotation=0)

# Scale and position the substations to preserve relative distances
# with maximum distance of approximately 500 units
print("-" * 50)
print("Original UTM coordinates:")
print("-" * 50)
for sub in [arts_220kv, wbts_220kv, cwts_220kv, bats_220kv]:
    print(f"{sub.name:<5}: ({sub.x:.2f}, {sub.y:.2f})")

# Find min/max values for scaling
min_x = min(arts_220kv.x, wbts_220kv.x, cwts_220kv.x, bats_220kv.x)
max_x = max(arts_220kv.x, wbts_220kv.x, cwts_220kv.x, bats_220kv.x)
min_y = min(arts_220kv.y, wbts_220kv.y, cwts_220kv.y, bats_220kv.y)
max_y = max(arts_220kv.y, wbts_220kv.y, cwts_220kv.y, bats_220kv.y)

# Calculate ranges
x_range = max_x - min_x
y_range = max_y - min_y

# Determine scaling factor for max range to be 500 units
max_range = max(x_range, y_range)
scale_factor = 500 / max_range

print("\n" + "-" * 50)
print(f"Scaling Information:")
print("-" * 50)
print(f"Scaling factor: {scale_factor:.6f}")
print(f"X range: {x_range:.2f}, Y range: {y_range:.2f}")
print(f"Max range: {max_range:.2f}")

# Apply scaling and translation with y-axis flipped (since SVG y increases downward)
print("\nScaled coordinates:")
for sub in [arts_220kv, wbts_220kv, cwts_220kv, bats_220kv]:
    # Scale coordinates and add offset to keep everything visible
    sub.scaled_x = (sub.x - min_x) * scale_factor + 100
    # Invert the y-axis to match geographic orientation (north at top)
    sub.scaled_y = 600 - ((sub.y - min_y) * scale_factor + 100)
    print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")


# Use NetworkX to locate the substations with balanced spacing
import networkx as nx

# Create a dictionary of our substations with their scaled coordinates
nodes = [arts_220kv.name, wbts_220kv.name, cwts_220kv.name, bats_220kv.name]
initial_pos = {
    arts_220kv.name: (arts_220kv.scaled_x, arts_220kv.scaled_y),
    wbts_220kv.name: (wbts_220kv.scaled_x, wbts_220kv.scaled_y),
    cwts_220kv.name: (cwts_220kv.scaled_x, cwts_220kv.scaled_y),
    bats_220kv.name: (bats_220kv.scaled_x, bats_220kv.scaled_y),
}

print("\n" + "-" * 50)
print("Network Layout Calculation")
print("-" * 50)
print("Initial positions:")
for node, pos in initial_pos.items():
    print(f"{node}: (x={pos[0]:.4f}, y={pos[1]:.4f})")

# Create a graph and add nodes
G = nx.Graph()
G.add_nodes_from(nodes)

# Add edges connecting ARTS-CWTS and ARTS-WBTS as specified
G.add_edges_from(
    [
        (arts_220kv.name, cwts_220kv.name),
        (arts_220kv.name, wbts_220kv.name),
        (wbts_220kv.name, bats_220kv.name),
    ]
)

# Run the physics simulation to balance the layout
# Use appropriate parameters for 0-500 range positions
final_pos = nx.kamada_kawai_layout(
    G,
    pos=initial_pos,  # Using our original 0-500 range positions
    scale=0.5,  # No additional scaling needed
)

print("\nBalanced positions after network simulation:")
for node, coords in final_pos.items():
    print(f"{node}: (x={coords[0]:.4f}, y={coords[1]:.4f})")

# Apply the balanced positions back to our substations
substation_map = {
    arts_220kv.name: arts_220kv,
    wbts_220kv.name: wbts_220kv,
    cwts_220kv.name: cwts_220kv,
    bats_220kv.name: bats_220kv,
}

# Scale the positions back to our SVG coordinate space
for name, coords in final_pos.items():
    sub = substation_map[name]
    # Scale from 0-1 range to our SVG size (adding margins)
    sub.scaled_x = coords[0] * 800 + 100  # Scale to 800 width + 100 margin
    sub.scaled_y = coords[1] * 600 + 100  # Scale to 600 height + 100 margin

print("\nFinal SVG coordinates after network balancing:")
for sub in [arts_220kv, wbts_220kv, cwts_220kv, bats_220kv]:
    print(f"{sub.name}: ({sub.scaled_x:.1f}, {sub.scaled_y:.1f})")


# draw at the scaled x and y locations
d.append(draw.Use(d_arts, arts_220kv.scaled_x, arts_220kv.scaled_y))
d.append(draw.Use(d_wbts, wbts_220kv.scaled_x, wbts_220kv.scaled_y))
d.append(draw.Use(d_cwts, cwts_220kv.scaled_x, cwts_220kv.scaled_y))
d.append(draw.Use(d_bats, bats_220kv.scaled_x, bats_220kv.scaled_y))


# Function to connect points between substations
def connect_substation_points(
    drawing: draw.Drawing,
    source_sub,
    source_bay_idx: int,
    source_y_offset: float,
    target_sub,
    target_bay_idx: int,
    target_y_offset: float,
    color: str = "red",
    stroke_width: float = 1.5,
    stroke_dasharray: str = "",
    add_arrow: bool = False,
):
    """Draw a line connecting specific points between two substations using right angles."""
    # Calculate bay x-offset (50 units per bay)
    source_local_x = 50 * source_bay_idx
    target_local_x = 50 * target_bay_idx

    # Calculate global coordinates by adding substation position
    source_x = source_sub.scaled_x + source_local_x
    source_y = source_sub.scaled_y + source_y_offset

    target_x = target_sub.scaled_x + target_local_x
    target_y = target_sub.scaled_y + target_y_offset

    # Create a path with right-angle bends
    path = draw.Path(
        stroke=color,
        stroke_width=stroke_width,
        stroke_dasharray=stroke_dasharray,
        fill="none",
    )
    path.M(source_x, source_y)

    # Decide on V-H-V or H-V-H path to make the connection look balanced.
    if abs(source_x - target_x) > abs(source_y - target_y):
        # Horizontal-Vertical-Horizontal path
        mid_x = (source_x + target_x) / 2
        path.L(mid_x, source_y).L(mid_x, target_y).L(target_x, target_y)
    else:
        # Vertical-Horizontal-Vertical path
        mid_y = (source_y + target_y) / 2
        path.L(source_x, mid_y).L(target_x, mid_y).L(target_x, target_y)

    # Create arrow marker if requested
    if add_arrow:
        arrow = draw.Marker(-0.1, -0.5, 0.9, 0.5, scale=4, orient="auto")
        arrow.append(draw.Lines(-0.1, 0.5, -0.1, -0.5, 0.9, 0, fill=color, close=True))
        path.marker_end = arrow

    # Add the path to the drawing
    drawing.append(path)
    return drawing


# Example: Connect CWTS bay 1 between element2 and element3 to ARTS bay 0 between element2 and element3
# For breaker-and-a-half bays:
# - element2 (middle switch) is at y=90
# - element3 (bottom switch) is at y=150
# - so between them is around y=120
connect_substation_points(
    d,  # SVG drawing
    cwts_220kv,  # Source substation
    1,  # Source bay index (second bay)
    120,  # Source y (between element2 and element3)
    arts_220kv,  # Target substation
    0,  # Target bay index (first bay)
    120,  # Target y (between element2 and element3)
    color="green",  # Line color
    stroke_width=2,  # Line width
)
connect_substation_points(
    d,  # SVG drawing
    arts_220kv,  # Source substation
    1,  # Source bay index (second bay)
    120,  # Source y (between element2 and element3)
    wbts_220kv,  # Target substation
    0,  # Target bay index (first bay)
    60,  # Target y (between element2 and element3)
    color="green",  # Line color
    stroke_width=2,  # Line width
)
connect_substation_points(
    d,  # SVG drawing
    wbts_220kv,  # Source substation
    3,  # Source bay index (second bay)
    60,  # Source y (between element2 and element3)
    bats_220kv,  # Target substation
    0,  # Target bay index (first bay)
    60,  # Target y (between element2 and element3)
    color="green",  # Line color
    stroke_width=2,  # Line width
)

# draw bus
# d.append(get_substation_group(new_sub, rotation=0))
# d.append(get_substation_group(arts_220kv, rotation=0))
# d.append(get_substation_group(wbts_220kv, rotation=0))
# d.append(dget_substation_group(cwts_220kv, rotation=0))
d.save_svg("example.svg")
