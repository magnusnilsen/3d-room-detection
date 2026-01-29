"""
Mesh loading utilities for OBJ files with group preservation.
"""

import trimesh
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


class MeshData:
    """Container for loaded mesh data with group information."""
    
    def __init__(
        self,
        combined_mesh: trimesh.Trimesh,
        face_to_group: np.ndarray,
        group_names: Dict[int, str],
        group_bounds: Dict[str, np.ndarray],
        individual_meshes: Dict[str, trimesh.Trimesh]
    ):
        """
        Initialize MeshData.
        
        Args:
            combined_mesh: Single combined mesh for ray casting
            face_to_group: Array mapping face index to group ID
            group_names: Dict mapping group ID to group name
            group_bounds: Dict mapping group name to bounding box [min, max]
            individual_meshes: Dict mapping group name to individual mesh
        """
        self.mesh = combined_mesh
        self.face_to_group = face_to_group
        self.group_names = group_names
        self.group_bounds = group_bounds
        self.individual_meshes = individual_meshes
    
    def get_group_name(self, face_idx: int) -> str:
        """Get the group name for a given face index."""
        if face_idx < 0 or face_idx >= len(self.face_to_group):
            return "unknown"
        group_id = self.face_to_group[face_idx]
        return self.group_names.get(group_id, "unknown")
    
    def get_group_bounds(self, group_name: str) -> Optional[np.ndarray]:
        """Get the bounding box for a group."""
        return self.group_bounds.get(group_name)
    
    def get_face_normal(self, face_idx: int) -> np.ndarray:
        """Get the normal vector for a face."""
        if face_idx < 0 or face_idx >= len(self.mesh.face_normals):
            return np.array([0, 0, 1])
        return self.mesh.face_normals[face_idx]


def load_building_mesh(obj_path: str) -> MeshData:
    """
    Load an OBJ file and return combined mesh with group mapping.
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        MeshData object containing the combined mesh and group information
    """
    path = Path(obj_path)
    if not path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    
    # Load as scene to preserve groups
    scene = trimesh.load(str(obj_path), group_material=False, force='scene')
    
    if isinstance(scene, trimesh.Trimesh):
        # Single mesh, no groups
        return MeshData(
            combined_mesh=scene,
            face_to_group=np.zeros(len(scene.faces), dtype=np.int32),
            group_names={0: "default"},
            group_bounds={"default": scene.bounds},
            individual_meshes={"default": scene}
        )
    
    if not isinstance(scene, trimesh.Scene) or len(scene.geometry) == 0:
        raise ValueError("Failed to load OBJ file or file contains no geometry")
    
    # Build group mapping
    geometries = list(scene.geometry.items())
    
    face_to_group_list = []
    group_names = {}
    group_bounds = {}
    individual_meshes = {}
    
    for group_id, (name, mesh) in enumerate(geometries):
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        
        # Store mapping from faces to group
        num_faces = len(mesh.faces)
        face_to_group_list.append(np.full(num_faces, group_id, dtype=np.int32))
        
        # Store group info
        group_names[group_id] = name
        group_bounds[name] = mesh.bounds.copy()
        individual_meshes[name] = mesh
    
    # Concatenate all meshes
    mesh_list = [m for m in scene.geometry.values() if isinstance(m, trimesh.Trimesh)]
    if not mesh_list:
        raise ValueError("No valid meshes found in scene")
    
    combined_mesh = trimesh.util.concatenate(mesh_list)
    face_to_group = np.concatenate(face_to_group_list)
    
    print(f"Loaded mesh with {len(combined_mesh.vertices)} vertices, "
          f"{len(combined_mesh.faces)} faces, {len(group_names)} groups")
    print(f"Bounds: {combined_mesh.bounds}")
    
    return MeshData(
        combined_mesh=combined_mesh,
        face_to_group=face_to_group,
        group_names=group_names,
        group_bounds=group_bounds,
        individual_meshes=individual_meshes
    )
