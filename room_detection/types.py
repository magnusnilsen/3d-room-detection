"""
Data types for room detection algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Plane:
    """Represents a plane in 3D space."""
    point: np.ndarray   # Point on the plane
    normal: np.ndarray  # Unit normal vector
    
    def __post_init__(self):
        # Ensure normal is normalized
        norm = np.linalg.norm(self.normal)
        if norm > 0:
            self.normal = self.normal / norm


@dataclass
class VerticalHitResult:
    """Result of vertical ray casting for a single sample point."""
    sample_point: np.ndarray      # Original sample point (x, y, z)
    top_hit: np.ndarray           # Top intersection point
    bottom_hit: np.ndarray        # Bottom intersection point
    room_height: float            # Distance between top and bottom hits
    top_face_idx: int             # Face index of top hit (for plane extraction)
    top_mesh_name: str            # Group/object name of the hit mesh
    
    @property
    def ceiling_z(self) -> float:
        """Z coordinate of the ceiling hit."""
        return float(self.top_hit[2])
    
    @property
    def floor_z(self) -> float:
        """Z coordinate of the floor hit."""
        return float(self.bottom_hit[2])


@dataclass
class CandidatePoint:
    """A validated point ready for flood fill processing."""
    sample_point: np.ndarray        # Original grid sample point
    top_hit_point: np.ndarray       # Ceiling intersection point
    bottom_hit_point: np.ndarray    # Floor intersection point
    room_height: float              # Vertical distance (ceiling to floor)
    ceiling_plane: Plane            # Plane for flood fill domain
    ceiling_mesh_bounds: np.ndarray # Bounding box of hit mesh object [min, max]
    ceiling_mesh_name: str          # Group name (e.g., "Obj.157")
    
    # Horizontal ray validation results (optional, for debugging)
    horizontal_distances: Optional[dict] = field(default=None)
    
    @property
    def floor_z(self) -> float:
        """Z coordinate of the floor."""
        return float(self.bottom_hit_point[2])
    
    @property
    def ceiling_z(self) -> float:
        """Z coordinate of the ceiling."""
        return float(self.top_hit_point[2])


@dataclass
class RoomDetectionConfig:
    """Configuration parameters for room detection algorithm."""
    
    # Grid sampling
    grid_spacing_cm: float = 25.0
    
    # Multi-floor sampling
    floor_height_cm: float = 300.0  # Expected floor-to-floor height
    sample_offset_from_floor_cm: float = 150.0  # Sample at mid-height of each floor
    
    # Height filtering
    min_room_height_cm: float = 200.0  # Minimum valid room height
    max_room_height_cm: float = 500.0  # Maximum valid room height (filter atriums)
    
    # Horizontal validation
    min_room_dimension_cm: float = 100.0  # Minimum room width/depth
    horizontal_ray_offset_cm: float = 10.0  # Cast rays this far below ceiling
    wall_thickness_cm: float = 15.0  # Minimum distance to consider as "inside room"
    
    # Performance
    batch_size: int = 10000  # Batch size for ray casting operations
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.grid_spacing_cm <= 0:
            raise ValueError("grid_spacing_cm must be positive")
        if self.min_room_height_cm >= self.max_room_height_cm:
            raise ValueError("min_room_height_cm must be less than max_room_height_cm")
        if self.min_room_dimension_cm <= 0:
            raise ValueError("min_room_dimension_cm must be positive")
