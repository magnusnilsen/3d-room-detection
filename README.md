# 3D Room Detection

A project for experimenting with automatically producing room geometry from OBJ files (coming from BIM models).

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

Place your OBJ files in the `sample_models/` folder, then run:

```bash
python main.py <path_to_obj_file>
```

Or process all OBJ files in the sample_models folder:

```bash
python main.py sample_models/
```

## Project Structure

```
.
├── main.py              # Main entry point
├── room_detection.py    # Core room detection algorithm
├── requirements.txt     # Python dependencies
├── sample_models/       # Place your OBJ files here
└── README.md           # This file
```
