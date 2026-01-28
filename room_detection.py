"""
Core room detection algorithm for processing OBJ files from BIM models.
"""

import numpy as np
import trimesh
from typing import List, Dict, Any


def detect_rooms(obj_file_path: str) -> List[Dict[str, Any]]:
    """
    Detect rooms from an OBJ file.
    
    Args:
        obj_file_path: Path to the OBJ file
        
    Returns:
        List of dictionaries containing room information
    """
    # Load the mesh
    mesh = trimesh.load(obj_file_path)
    
    # Ensure we have a scene or mesh
    if isinstance(mesh, trimesh.Scene):
        # If it's a scene, get the first mesh or combine all meshes
        if len(mesh.geometry) > 0:
            mesh = list(mesh.geometry.values())[0]
        else:
            raise ValueError("Scene contains no geometry")
    
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
    print(f"Bounds: {mesh.bounds}")
    print(f"Volume: {mesh.volume:.2f}")
    
    # TODO: Implement room detection algorithm
    # This is a placeholder that returns basic mesh information
    rooms = [{
        "id": 1,
        "bounds": mesh.bounds.tolist(),
        "volume": float(mesh.volume),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }]
    
    return rooms
