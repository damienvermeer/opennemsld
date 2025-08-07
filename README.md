# NEM SLD

## Overview

NEM SLD is an interactive geographic schematic of the Australian electrical system (transmission and sub-transmission). This project visualizes the National Electricity Market (NEM) Single Line Diagrams in an accessible web format.

## Features

- **Geographically Relative**: Substations are positioned at their approximate real-world locations relative to each other (though transmission lines may not follow exact geographic routes)
- **Realistic Switchgear Layout**: The layout of switchgear is based on the real-world arrangement of substations, with some simplifications for clarity
- **Interactive Navigation**: Click to pan around, scroll to zoom in/out, or use the search bar (Ctrl+F) to find substations by name
- **Asset Information**: Hover on generators and assets to view additional information

## Current Status

**⚠️ Work in Progress ⚠️**

Currently, only parts of Victoria have been implemented. More regions of the National Electricity Market will be added in future updates.

## Data Sources

- **AEMO NEM SLDs**: The Australian Energy Market Operator used to publish [NEM SLDs](https://web.archive.org/web/20220119012004if_/https://aemo.com.au/-/media/files/electricity/nem/planning_and_forecasting/maps/nem-slds.pdf) but these haven't been updated since 2019
- **Open Infrastructure Map**: [Open Infrastructure Map](https://openinframap.org/#6.89/-36.804/144.698) has been used for some geographic data

## Usage

1. Open the index.html file in a web browser
2. Navigate the map using your mouse:
   - Click and drag to pan
   - Scroll to zoom in/out
   - Use Ctrl+F to search for specific substations
3. Hover over generators and assets to view additional information

## Contributing

This diagram is community-sourced and contributions are welcome! You can help by:

1. **Reporting Errors**: If you notice any inaccuracies or have updates, please:
   - Email [updates@nemsld.com](mailto:updates@nemsld.com), or
   - [Post an issue on GitHub](https://github.com/damienvermeer/opennemsld)

2. **Providing Information**: Share:
   - Details about errors or changes (references are helpful for verification)
   - Public information about substation layouts or locations
   - Information about new generating systems (once built, not just at committed stage)

3. **Code Contributions**: For those wanting to modify the codebase directly:
   - Fork the repository
   - Make your changes (substations are currently defined in `substation_definitions.yaml`)
   - Submit a pull request

## License

This project is free and open source. See the license file for details.