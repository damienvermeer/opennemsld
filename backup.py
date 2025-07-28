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
from typing import Union


class BayType(Enum):
    SINGLE = "single"
    DOUBLE = "double"


class SwitchType(Enum):
    CB = auto()
    ISOL = auto()


@dataclass
class Bus:
    name: str
    bays: list["BaseBay"] = field(default_factory=list)
    is_planar: bool = True

    def add_bay(self, bay: "BaseBay") -> None:
        self.bays.append(bay)

    @property
    def num_normal_bays(self) -> int:
        return len([x for x in self.bays if not isinstance(x, BusTieBay)])


@dataclass
class BaseBay:
    name: str
    switch: SwitchType = SwitchType.CB


@dataclass
class SingleSwitchedBay(BaseBay):
    draw_in: bool = True


@dataclass
class DoubleSwitchedBay(BaseBay):
    draw_in: bool = True
    switch2: SwitchType = SwitchType.CB
    other_bus: Bus = None


@dataclass
class BusTieBay(BaseBay):
    tie_bus: Bus = None


@dataclass
class Substation:
    name: str
    buses: list[Bus] = field(default_factory=list)

    def add_bus(self, bus: Bus) -> None:
        self.buses.append(bus)

    def get_bus_from_name(self, name: str) -> Bus:
        for bus in self.buses:
            if bus.name == name:
                return bus
        raise ValueError(f"Bus {name} not found")


# create a simple single switched sub
new_sub = Substation(name="sub1")
bus1 = Bus(name="4A")
new_sub.add_bus(bus1)
# add 3 CBs and an isol
bus1.add_bay(SingleSwitchedBay(name="bay1"))
bus1.add_bay(SingleSwitchedBay(name="bay2"))
bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))
bus1.add_bay(SingleSwitchedBay(name="bay2", switch=SwitchType.ISOL))

bus2 = Bus(name="2", is_planar=True)
new_sub.add_bus(bus2)
bus2.add_bay(SingleSwitchedBay(name="bay4", switch=SwitchType.ISOL))
bus3 = Bus(name="3", is_planar=True)
new_sub.add_bus(bus3)
# testing planar bustie
bus1.add_bay(BusTieBay(name="bay3", tie_bus=bus2, switch=SwitchType.ISOL))
bus2.add_bay(BusTieBay(name="bay5", tie_bus=bus3, switch=SwitchType.ISOL))
bus3.add_bay(SingleSwitchedBay(name="bay6", switch=SwitchType.CB))

# ------------------------------------------------

double_switched_sub = Substation(name="sub2")
dbus1 = Bus(name="4A")
dbus2 = Bus(name="1")
double_switched_sub.add_bus(dbus1)
double_switched_sub.add_bus(dbus2)
dbus1.add_bay(
    DoubleSwitchedBay(
        name="bay1", switch=SwitchType.ISOL, switch2=SwitchType.CB, other_bus=dbus2
    )
)
dbus1.add_bay(
    DoubleSwitchedBay(
        name="bay2", switch=SwitchType.CB, switch2=SwitchType.ISOL, other_bus=dbus2
    )
)


import drawsvg as draw

d = draw.Drawing(1000, 1000, origin="center")


def draw_cb_or_isol(
    xoff: float, ys: float, parent_group: draw.Group, bay: BaseBay
) -> draw.Group:
    """Draw a CB or isolator at the given x offset and y direction.

    Args:
        xoff (float): x offset from the bus
        ys (float): y direction (1 for up, -1 for down)
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (Bay): bay to draw the CB or isolator for

    Returns:
        draw.Group: parent group with the CB or isolator added
    """
    parent_group.append(draw.Line(xoff, 0, xoff, ys * 20))  # line off bus towards CB
    if bay.switch == SwitchType.CB:
        # Adjust y-coordinate based on ys direction
        y_pos = ys * 20 if ys == 1 else ys * 20 - 20
        parent_group.append(
            draw.Rectangle(-10 + xoff, y_pos, 20, 20, fill="white")
        )  # cb
    elif bay.switch == SwitchType.ISOL:
        parent_group.append(
            draw.Line(
                xoff,
                ys * 15,
                xoff,
                ys * 45,
                stroke_width=2,
                transform=f"rotate(45, {xoff}, {ys * 30})",
            )
        )
    parent_group.append(
        draw.Line(xoff, ys * 40, xoff, ys * 60)
    )  # line off bus towards feeder
    return parent_group


def draw_double_switched_bay(
    xoff: float, parent_group: draw.Group, bay: DoubleSwitchedBay
) -> draw.Group:
    """Draw a double switched bay at the given x offset and y direction.

    Args:
        xoff (float): x offset from the bus
        parent_group (draw.Group): parent group to add the CB or isolator to
        bay (DoubleSwitchedBay): bay to draw the double switched bay for

    Returns:
        draw.Group: parent group with the double switched bay added
    """
    parent_group.append(draw.Line(xoff, 0, xoff, 20))  # line off bus towards CB
    # draw first switch
    if bay.switch == SwitchType.CB:
        parent_group.append(draw.Rectangle(-10 + xoff, 20, 20, 20, fill="white"))  # cb
    elif bay.switch == SwitchType.ISOL:
        parent_group.append(
            draw.Line(
                xoff,
                15,
                xoff,
                45,
                stroke_width=2,
                transform=f"rotate(45, {xoff}, {30})",
            )
        )
    parent_group.append(draw.Line(xoff, 40, xoff, 80))  # line off bus towards feeder
    # ... which is longer in a double switched bay
    # draw second switch
    if bay.switch2 == SwitchType.CB:
        parent_group.append(draw.Rectangle(-10 + xoff, 80, 20, 20, fill="white"))  # cb
    elif bay.switch2 == SwitchType.ISOL:
        parent_group.append(
            draw.Line(
                xoff,
                75,
                xoff,
                105,
                stroke_width=2,
                transform=f"rotate(45, {xoff}, {90})",
            )
        )
    # draw line towards busbar2
    parent_group.append(draw.Line(xoff, 100, xoff, 120))
    return parent_group


def draw_inline_bustie(
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
    parent_group.append(
        draw.Line(busend_x, busend_y, busend_x + 40, busend_y, stroke_width=5)
    )  # line off bus towards CB
    if bay.switch == SwitchType.CB:
        parent_group.append(
            draw.Rectangle(busend_x + 40, busend_y - 10, 20, 20, fill="white")
        )  # cb
    elif bay.switch == SwitchType.ISOL:
        parent_group.append(
            draw.Line(
                busend_x + 35,
                busend_y,
                busend_x + 65,
                busend_y,
                stroke_width=2,
                # -45 to make it look similar to non-bus-tie ones
                transform=f"rotate(-45, {busend_x + 50}, {busend_y})",
            )
        )
    parent_group.append(
        draw.Line(busend_x + 60, busend_y, busend_x + 100, busend_y, stroke_width=5)
    )  # line off bus towards feeder
    return parent_group


# make substation group
def get_substation_group(sub: Substation, colour="blue", rotation=0):
    dg = draw.Group(stroke=colour, stroke_width=2, transform=f"rotate({rotation})")
    xoff = -100
    for i, bus in enumerate(k for k in sub.buses if k.is_planar):
        xoff += 100
        # draw main busbar
        bus_length = (bus.num_normal_bays - 1) * 50
        # draw small text at end of bus saying name
        dg.append(draw.Text(bus.name, x=-20 + xoff, y=-8, font_size=25, anchor="end"))
        dg.append(draw.Line(-20 + xoff, 0, bus_length + 20 + xoff, 0, stroke_width=5))

        # draw bays
        for j, bay in enumerate(x for x in bus.bays if not isinstance(x, BusTieBay)):
            xoff += 50 if j > 0 else 0
            ys = 1 if bay.draw_in else -1
            # check type
            if isinstance(bay, SingleSwitchedBay):
                dg = draw_cb_or_isol(xoff, ys, dg, bay)
            elif isinstance(bay, DoubleSwitchedBay):
                dg = draw_double_switched_bay(xoff, dg, bay)

        # TODO handle non planar bus tie
        for j, bustie in enumerate(x for x in bus.bays if isinstance(x, BusTieBay)):
            if bus.is_planar and bustie.tie_bus.is_planar:
                dg = draw_inline_bustie(xoff, 0, dg, bustie)

        # draw bus tie which is a CB in line with this bus
    # fill in non-planar buses

    return dg


# draw bus
# d.append(get_substation_group(new_sub, rotation=0))
d.append(get_substation_group(double_switched_sub, rotation=0))
d.save_svg("example.svg")
