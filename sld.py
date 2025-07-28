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
    bays: list[BaseBay] = field(default_factory=list)

    def add_bay(self, bay: BaseBay) -> None:
        self.bays.append(bay)
        # set each bay to its left neighbour
        for i, _ in enumerate(self.bays):
            if i == 0:
                continue
            self.bays[i].neighbour = self.bays[i - 1]


# testing - single switched sub
ss_sub = Substation("test")
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.CB))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.ISOL))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.CB))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.DIRECT))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.EMPTY))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.CB))
ss_sub.add_bay(SingleSwitchedBay("1", element=SwitchType.NOBUS))
ss_sub.add_bay(SingleSwitchedBay("2", element=SwitchType.CB))
ss_sub.add_bay(SingleSwitchedBay("2", element=SwitchType.CB))
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.CB, element2=SwitchType.CB, other_bus_name="3"
    )
)
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.CB, element2=SwitchType.DIRECT, other_bus_name="3"
    )
)
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.NOBUS, element2=SwitchType.ISOL, other_bus_name="3"
    )
)
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.NOBUS, element2=SwitchType.ISOL, other_bus_name="3"
    )
)
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.NOBUS, element2=SwitchType.NOBUS, other_bus_name="3"
    )
)
ss_sub.add_bay(
    DoubleSwitchedBay(
        "2", element=SwitchType.NOBUS, element2=SwitchType.CB, other_bus_name="4"
    )
)


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

    # # Line off bus towards first switch
    # parent_group.append(draw.Line(xoff, 0, xoff, 20))

    # # Draw first switch
    # draw_switch(xoff, 30, parent_group, bay.switch, "vertical")

    # # Line between switches
    # parent_group.append(draw.Line(xoff, 40, xoff, 80))

    # # Draw second switch
    # draw_switch(xoff, 90, parent_group, bay.switch2, "vertical")

    # # Line towards busbar2
    # parent_group.append(draw.Line(xoff, 100, xoff, 120))

    # return parent_group


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
    draw_switch(switch_x, busend_y, parent_group, bay.switch, "horizontal")

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


# draw bus
# d.append(get_substation_group(new_sub, rotation=0))
d.append(get_substation_group(ss_sub, rotation=0))
d.save_svg("example.svg")
