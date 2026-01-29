"""
3D visualization for room detection results.
"""

import numpy as np
import trimesh
from typing import List, Optional
from .types import CandidatePoint
from .mesh_loader import MeshData


def create_point_cloud(candidates: List[CandidatePoint], color: list = [255, 0, 0, 255]) -> trimesh.PointCloud:
    """
    Create a point cloud from candidate sample points.
    
    Args:
        candidates: List of candidate points
        color: RGBA color for points
        
    Returns:
        trimesh.PointCloud
    """
    if not candidates:
        return trimesh.PointCloud(vertices=np.zeros((0, 3)))
    
    points = np.array([c.sample_point for c in candidates])
    colors = np.tile(color, (len(points), 1))
    
    return trimesh.PointCloud(vertices=points, colors=colors)


def create_ceiling_markers(
    candidates: List[CandidatePoint],
    marker_size_cm: float = 20.0,
    color: list = [0, 255, 0, 200]
) -> trimesh.Trimesh:
    """
    Create small square markers at ceiling hit points.
    
    Args:
        candidates: List of candidate points
        marker_size_cm: Size of each marker square
        color: RGBA color for markers
        
    Returns:
        Combined mesh of all markers
    """
    if not candidates:
        return trimesh.Trimesh()
    
    markers = []
    half_size = marker_size_cm / 2
    
    for c in candidates:
        # Get ceiling hit point
        center = c.top_hit_point.copy()
        
        # Get plane normal (should be roughly pointing down for ceiling)
        normal = c.ceiling_plane.normal.copy()
        
        # Ensure normal points downward (into the room)
        if normal[2] > 0:
            normal = -normal
        
        # Create two perpendicular vectors on the plane
        if abs(normal[2]) > 0.9:
            # Nearly horizontal plane - use X and Y
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])
        else:
            # Find perpendicular vectors
            up = np.array([0.0, 0.0, 1.0])
            u = np.cross(normal, up)
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)
            v = v / np.linalg.norm(v)
        
        # Create square vertices
        vertices = np.array([
            center - half_size * u - half_size * v,
            center + half_size * u - half_size * v,
            center + half_size * u + half_size * v,
            center - half_size * u + half_size * v,
        ])
        
        # Offset slightly below ceiling to avoid z-fighting
        vertices[:, 2] -= 1.0
        
        # Create two triangles for the square
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        
        marker = trimesh.Trimesh(vertices=vertices, faces=faces)
        markers.append(marker)
    
    if not markers:
        return trimesh.Trimesh()
    
    # Combine all markers
    combined = trimesh.util.concatenate(markers)
    
    # Set color
    combined.visual.face_colors = np.tile(color, (len(combined.faces), 1))
    
    return combined


def visualize_results(
    mesh_data: MeshData,
    candidates: List[CandidatePoint],
    building_opacity: float = 0.3,
    point_color: list = [255, 50, 50, 255],
    marker_color: list = [50, 255, 50, 200],
    marker_size_cm: float = 20.0,
    show_points: bool = True,
    show_markers: bool = True
) -> None:
    """
    Open an interactive 3D viewer showing the building and detection results.
    
    Args:
        mesh_data: Loaded mesh data
        candidates: List of candidate points
        building_opacity: Opacity for the building mesh (0-1)
        point_color: RGBA color for sample points
        marker_color: RGBA color for ceiling markers
        marker_size_cm: Size of ceiling marker squares
        show_points: Whether to show sample points
        show_markers: Whether to show ceiling markers
    """
    # Create scene
    scene = trimesh.Scene()
    
    # Add building mesh with transparency
    building_mesh = mesh_data.mesh.copy()
    
    # Set face colors with transparency
    alpha = int(building_opacity * 255)
    face_colors = np.full((len(building_mesh.faces), 4), [180, 180, 180, alpha], dtype=np.uint8)
    building_mesh.visual.face_colors = face_colors
    
    scene.add_geometry(building_mesh, geom_name="building")
    
    if candidates:
        # Add sample points as point cloud
        if show_points:
            point_cloud = create_point_cloud(candidates, color=point_color)
            scene.add_geometry(point_cloud, geom_name="sample_points")
        
        # Add ceiling markers
        if show_markers:
            markers = create_ceiling_markers(
                candidates,
                marker_size_cm=marker_size_cm,
                color=marker_color
            )
            if len(markers.faces) > 0:
                scene.add_geometry(markers, geom_name="ceiling_markers")
    
    print(f"\nVisualization:")
    print(f"  - Building mesh: {len(building_mesh.faces)} faces (opacity: {building_opacity:.0%})")
    print(f"  - Candidate points: {len(candidates)}")
    if show_markers:
        print(f"  - Ceiling markers: {len(candidates)} squares ({marker_size_cm}cm)")
    print("\nOpening 3D viewer...")
    print("  - Left-click + drag: Rotate")
    print("  - Right-click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'q' or close window to exit")
    
    # Show the scene
    try:
        scene.show()
    except Exception as e:
        error_msg = str(e).lower()
        if "display" in error_msg or "screen" in error_msg or "index" in error_msg:
            print(f"\nError: No display available for visualization.")
            print("This typically happens when running in a headless environment.")
            print("Try running on a machine with a display, or use --no-visualize.")
        else:
            raise


def visualize_with_open3d(
    mesh_data: MeshData,
    candidates: List[CandidatePoint],
    building_opacity: float = 0.3,
    point_size: float = 5.0,
    marker_size_cm: float = 20.0
) -> None:
    """
    Alternative visualization using Open3D (if available).
    
    Open3D provides better interactive controls and rendering.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D not available, falling back to trimesh viewer")
        visualize_results(mesh_data, candidates, building_opacity)
        return
    
    geometries = []
    
    # Convert building mesh to Open3D
    building_mesh = mesh_data.mesh
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(building_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(building_mesh.faces)
    o3d_mesh.compute_vertex_normals()
    
    # Set color with transparency (Open3D uses vertex colors)
    colors = np.full((len(building_mesh.vertices), 3), [0.7, 0.7, 0.7])
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    geometries.append(o3d_mesh)
    
    if candidates:
        # Add sample points
        points = np.array([c.sample_point for c in candidates])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(
            np.full((len(points), 3), [1.0, 0.2, 0.2])
        )
        geometries.append(point_cloud)
        
        # Add ceiling markers
        markers = create_ceiling_markers(candidates, marker_size_cm=marker_size_cm)
        if len(markers.faces) > 0:
            o3d_markers = o3d.geometry.TriangleMesh()
            o3d_markers.vertices = o3d.utility.Vector3dVector(markers.vertices)
            o3d_markers.triangles = o3d.utility.Vector3iVector(markers.faces)
            o3d_markers.compute_vertex_normals()
            marker_colors = np.full((len(markers.vertices), 3), [0.2, 1.0, 0.2])
            o3d_markers.vertex_colors = o3d.utility.Vector3dVector(marker_colors)
            geometries.append(o3d_markers)
    
    print(f"\nOpening Open3D viewer...")
    print(f"  - Building: {len(building_mesh.faces)} faces")
    print(f"  - Points: {len(candidates)}")
    print("\nControls:")
    print("  - Left-click + drag: Rotate")
    print("  - Ctrl + left-click + drag: Pan")
    print("  - Scroll: Zoom")
    print("  - Press 'q' to exit")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Room Detection Results",
        width=1280,
        height=720
    )
