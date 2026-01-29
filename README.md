# 3D Room Detection

A project for experimenting with automatically producing room geometry from OBJ files (coming from BIM models).

## Features

- Automatic room detection from OBJ files exported from IFC/Revit
- Multi-floor building support
- Configurable parameters for grid spacing, room height thresholds, and minimum room dimensions
- Outputs candidate points ready for flood fill polygon generation

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Place your OBJ files in the `sample_models/` folder, then run:

```bash
# Activate virtual environment first
source venv/bin/activate

# Process a single OBJ file
python main.py sample_models/building.obj

# Process all OBJ files in the folder
python main.py sample_models/
```

### Command Line Options

```bash
python main.py [OPTIONS] <input_path>

Options:
  --grid-spacing FLOAT    Grid sampling spacing in cm (default: 25.0)
  --min-height FLOAT      Minimum room height in cm (default: 200.0)
  --max-height FLOAT      Maximum room height in cm (default: 500.0)
  --min-room-size FLOAT   Minimum room dimension in cm (default: 100.0)
  --by-floor              Group results by floor level
  --quiet                 Suppress progress output
```

### Examples

```bash
# Standard run with default parameters
python main.py sample_models/building.obj

# Coarser grid for faster processing
python main.py --grid-spacing 50 sample_models/building.obj

# Custom height thresholds
python main.py --min-height 220 --max-height 400 sample_models/building.obj

# Show results grouped by floor
python main.py --by-floor sample_models/building.obj
```

### Python API

```python
from room_detection import (
    detect_rooms,
    detect_rooms_by_floor,
    RoomDetectionConfig,
)

# With default configuration
candidates = detect_rooms("sample_models/building.obj")

# With custom configuration
config = RoomDetectionConfig(
    grid_spacing_cm=50.0,
    min_room_height_cm=220.0,
    max_room_height_cm=400.0,
    min_room_dimension_cm=150.0,
)
candidates = detect_rooms("sample_models/building.obj", config=config)

# Get results grouped by floor
floors = detect_rooms_by_floor("sample_models/building.obj")
for floor_z, floor_candidates in floors.items():
    print(f"Floor at Z={floor_z}: {len(floor_candidates)} candidates")
```

## Algorithm Overview

The room detection pipeline works as follows:

1. **Mesh Loading**: Load OBJ file preserving group information
2. **Grid Sampling**: Generate sample points across the building footprint at each floor level
3. **Vertical Ray Casting**: Cast rays up and down to find interior points (both directions must hit)
4. **Height Filtering**: Remove points with unrealistic room heights
5. **Horizontal Ray Casting**: Validate room dimensions with 6 horizontal rays (3 opposing pairs)
6. **Candidate Generation**: Create candidate points with ceiling plane and mesh information

The candidate points are ready for flood fill processing to generate room polygons.

## Project Structure

```
.
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── sample_models/             # Place your OBJ files here
├── room_detection/            # Core package
│   ├── __init__.py           # Package exports
│   ├── types.py              # Data classes (CandidatePoint, Plane, etc.)
│   ├── config.py             # Configuration re-exports
│   ├── mesh_loader.py        # OBJ loading with group preservation
│   ├── grid_sampler.py       # Grid point generation
│   ├── ray_caster.py         # Vertical and horizontal ray casting
│   ├── point_filter.py       # Height and room size filtering
│   └── pipeline.py           # Main detection pipeline
└── README.md
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_spacing_cm` | 25.0 | Distance between sample points |
| `min_room_height_cm` | 200.0 | Minimum valid room height |
| `max_room_height_cm` | 500.0 | Maximum valid room height |
| `min_room_dimension_cm` | 100.0 | Minimum room width/depth |
| `horizontal_ray_offset_cm` | 10.0 | Distance below ceiling for horizontal rays |
| `wall_thickness_cm` | 15.0 | Minimum distance to be considered inside room |

## Notes

- Z axis is assumed to be "up" (vertical)
- Coordinates are expected in centimeters
- OBJ files should be exported with groups preserved (each BIM element as a group)
