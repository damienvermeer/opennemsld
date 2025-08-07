# NEM SLD Language Reference

This document provides a comprehensive reference for the NEM SLD language, which is used to define electrical substations and their components in a compact, text-based format.

## Table of Contents

- [Introduction](#introduction)
- [Substation Structure](#substation-structure)
- [Bay Definition Language](#bay-definition-language)
  - [Busbar Symbols](#busbar-symbols)
  - [Element Symbols](#element-symbols)
  - [Connection References](#connection-references)
- [Objects](#objects)
  - [Transformers](#transformers)
  - [Generators](#generators)
- [Child Definitions](#child-definitions)
- [Complete Example](#complete-example)

## Introduction

NEM SLD provides a compact, text-based language for defining electrical substations and their components. It allows for the representation of complex electrical networks using a combination of symbols and connection references. The language is designed to be readable by humans and parsable by software to generate visual single-line diagrams.

## Substation Structure

A substation is defined in YAML with the following structure:

```yaml
- name: "Substation Name"
  lat: -35.123456           # Latitude
  long: 142.123456          # Longitude
  voltage_kv: 220           # Primary voltage level in kV
  def: |                    # Bay definitions using the symbolic language
    |/1x2x||d5
    |x3x4x||
  buses:                    # Bus definitions
    1: "Bus Name 1"         # Bus ID: Bus Name
    2: "Bus Name 2"
  connections:              # External connection definitions
    1: "Connection Name 1"  # Connection ID: Connection Name
    2: "Connection Name 2"
  objects:                  # Additional objects (transformers, generators, etc.)
    # Object definitions
  child_definitions:        # Nested substation components
    # Child definitions
```

## Bay Definition Language

The bay definition language uses a compact, symbolic representation to define the layout of electrical components. Each line in the `def` field represents a horizontal bay, and each character or sequence of characters represents a specific component.

### Busbar Symbols

| Symbol | Description |
|--------|-------------|
| `\|`   | Standard busbar. Multiple consecutive `\|` characters indicate busbar ID. |
| `s`    | String busbar (thin line). |
| `N`    | Null busbar (no line drawn - used indicating a split in the busbar). |
| `t`    | Busbar with circuit breaker tie. |
| `ts`   | Busbar with circuit breaker tie (thin line). |
| `i`    | Busbar with isolator tie. |
| `is`   | Busbar with isolator tie (thin line). |

### Element Symbols

| Symbol | Description |
|--------|-------------|
| `x`    | Circuit breaker. |
| `?`    | Unknown switch type (drawn as a question mark). |
| `/`    | Isolator (diagonal line). |
| `d`    | Direct connection (no switch). |
| `E`    | Empty bay position (used for spacing). |

### Connection References

Numbers in the bay definition represent external connections. For example, `1` refers to connection ID 1 as defined in the `connections` field - which is a list like the below:
```yaml
connections:
  1: "Connection Name 1"
  2: "Connection Name 2"
```

## Objects

Objects are additional components that can be defined separately from the bay structure. These include transformers, generators, and other specialized equipment.

### Transformers

Transformers are defined with the following attributes:

```yaml
- type: tx-ud  # Up-down transformer, winding 1 at top
  metadata:
    w1: 220     # Winding 1 voltage in kV
    w2: 66      # Winding 2 voltage in kV
  rel_x: 1      # Relative x position (grid steps)
  rel_y: 3      # Relative y position (grid steps)
  connections:
    1: "HV Connection"   # Winding 1 connection
    2: "LV Connection"   # Winding 2 connection
```

Transformer types:
- `tx-ud`: Up-down transformer (vertical orientation)
- `tx-lr`: Left-right transformer (horizontal orientation)

### Generators

Generators are defined with the following attributes:

```yaml
- type: gen
  metadata:
    voltage: 66      # Generator voltage in kV
    text: "WF"       # Text to display inside the generator symbol
    info: |          # Optional info for popups
      Wind Farm Name
      39 x 2.05 = 80 MW
  rel_x: 1           # Relative x position (grid steps) relative to the parent substation origin
  rel_y: 4           # Relative y position (grid steps) relative to the parent substation origin
```

## Child Definitions

Child definitions allow for nested substation components, such as lower voltage sections connected through transformers. They follow the same structure as the main substation definition but are included within the parent substation. Multiple child definitions can be included in a single substation definition.

```yaml
child_definitions:
  - def: |
      1dsx
      3dsx
    voltage_kv: 33
    rel_x: -2
    rel_y: 19
    connections:
      1: "Connection Name"
    objects:
      # Objects specific to this child definition
```

## Complete Example

Here's a complete example of a simple substation definition:

```yaml
- name: "Kiamal (KMTS)"
  lat: -35.029758
  long: 142.293822
  voltage_kv: 220
  def: |
    |/1x2x||d5
    |x3x4x||
  buses:
    1: ''
    2: ''
  connections:
    1: VIC-220-MRTS-KMTS-1
    2: VIC-220-KIAMALSF-TX1
    3: VIC-220-RCTS-KMTS-1
    4: VIC-220-KIAMALSC
    5: VIC-220-KIAMALSF-TX2
  objects:
    - type: tx-ud
      metadata:
        w1: 220
        w2: 33
      rel_x: -1
      rel_y: 12
      connections:
        1: VIC-220-KIAMALSF-TX1
        2: VIC-33-KIAMALSF-BUS1
    - type: gen
      metadata:
        voltage: 33
        text: PV
        info: |
          Kiamal Solar Farm
          150 x 1.598 = 239.7 MVA (200 MW)
      rel_x: 1
      rel_y: 4
```

This definition represents a substation named "Kiamal" with:
- Two busbars in a breaker and a half arrangement (with one CB replaced with an isolator)
- A direct line connection at the bottom of the first bay
- External connections to other substations
- A transformer and generator

The resulting diagram would show these components with appropriate connections between them.

## Diagram Visualization

When the bay definition is parsed, it generates a single-line diagram with:

1. Busbars represented as horizontal lines
2. Circuit breakers as squares
3. Isolators as diagonal lines
4. Direct connections as vertical or horizontal lines
5. External connections labeled with their names
6. Transformers and generators with standard industry symbols
7. Color-coding based on voltage levels

The grid-based layout ensures consistent spacing and alignment of components, while the connection references maintain logical relationships between different parts of the network.
