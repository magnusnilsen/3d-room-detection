"""
3D visualization for room detection results using Open3D.
"""

import numpy as np
from typing import List, Optional
from .types import CandidatePoint
from .mesh_loader import MeshData


def create_ceiling_marker_mesh(
    candidates: List[CandidatePoint],
    marker_size_cm: float = 20.0
) -> tuple:
    """
    Create vertices and triangles for ceiling markers.
    
    Returns:
        Tuple of (vertices, triangles) as numpy arrays
    """
    if not candidates:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    
    all_vertices = []
    all_triangles = []
    vertex_offset = 0
    half_size = marker_size_cm / 2
    
    for c in candidates:
        center = c.top_hit_point.copy()
        
        # Create square vertices (horizontal plane)
        vertices = np.array([
            [center[0] - half_size, center[1] - half_size, center[2] - 2],
            [center[0] + half_size, center[1] - half_size, center[2] - 2],
            [center[0] + half_size, center[1] + half_size, center[2] - 2],
            [center[0] - half_size, center[1] + half_size, center[2] - 2],
        ])
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ]) + vertex_offset
        
        all_vertices.append(vertices)
        all_triangles.append(triangles)
        vertex_offset += 4
    
    return np.vstack(all_vertices), np.vstack(all_triangles)


def visualize_results(
    mesh_data: MeshData,
    candidates: List[CandidatePoint],
    building_opacity: float = 0.3,
    point_color: list = [1.0, 0.2, 0.2],
    marker_color: list = [0.2, 1.0, 0.2],
    marker_size_cm: float = 20.0,
    show_points: bool = True,
    show_markers: bool = True,
    point_size: float = 8.0,
    wireframe: bool = True
) -> None:
    """
    Open an interactive 3D viewer showing the building and detection results.
    
    Uses Open3D for reliable cross-platform visualization.
    
    Args:
        mesh_data: Loaded mesh data
        candidates: List of candidate points
        building_opacity: Opacity for the building mesh (0-1) - affects color brightness
        point_color: RGB color for sample points (0-1 range)
        marker_color: RGB color for ceiling markers (0-1 range)
        marker_size_cm: Size of ceiling marker squares
        show_points: Whether to show sample points
        show_markers: Whether to show ceiling markers
        point_size: Size of point visualization
        wireframe: If True, show building as wireframe (recommended to see interior points)
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: Open3D is required for visualization.")
        print("Install it with: pip install open3d")
        return
    
    geometries = []
    
    # Create building mesh
    building_mesh = mesh_data.mesh
    o3d_building = o3d.geometry.TriangleMesh()
    o3d_building.vertices = o3d.utility.Vector3dVector(building_mesh.vertices.astype(np.float64))
    o3d_building.triangles = o3d.utility.Vector3iVector(building_mesh.faces.astype(np.int32))
    o3d_building.compute_vertex_normals()
    
    if wireframe:
        # Convert to wireframe (LineSet) for see-through effect
        wireframe_lines = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_building)
        # Set wireframe color - subtle gray
        line_color = [0.4, 0.4, 0.5]
        wireframe_lines.paint_uniform_color(line_color)
        geometries.append(wireframe_lines)
    else:
        # Solid mesh with color
        gray_value = 0.5 + (1 - building_opacity) * 0.3
        building_colors = np.full((len(building_mesh.vertices), 3), [gray_value, gray_value, gray_value])
        o3d_building.vertex_colors = o3d.utility.Vector3dVector(building_colors)
        geometries.append(o3d_building)
    
    if candidates:
        # Add sample points as point cloud
        if show_points:
            points = np.array([c.sample_point for c in candidates], dtype=np.float64)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(
                np.tile(point_color, (len(points), 1))
            )
            geometries.append(point_cloud)
        
        # Add ceiling markers as mesh
        if show_markers:
            marker_verts, marker_tris = create_ceiling_marker_mesh(
                candidates, marker_size_cm=marker_size_cm
            )
            if len(marker_verts) > 0:
                o3d_markers = o3d.geometry.TriangleMesh()
                o3d_markers.vertices = o3d.utility.Vector3dVector(marker_verts.astype(np.float64))
                o3d_markers.triangles = o3d.utility.Vector3iVector(marker_tris.astype(np.int32))
                o3d_markers.compute_vertex_normals()
                marker_colors = np.tile(marker_color, (len(marker_verts), 1))
                o3d_markers.vertex_colors = o3d.utility.Vector3dVector(marker_colors)
                geometries.append(o3d_markers)
    
    print(f"\nVisualization:")
    mode = "wireframe" if wireframe else f"solid (opacity: {building_opacity:.0%})"
    print(f"  - Building mesh: {len(building_mesh.faces)} faces ({mode})")
    print(f"  - Candidate points: {len(candidates)}")
    if show_markers:
        print(f"  - Ceiling markers: {len(candidates)} squares ({marker_size_cm}cm)")
    print("\nOpening Open3D viewer...")
    print("\nControls:")
    print("  - Left-click + drag: Rotate")
    print("  - Scroll: Zoom")
    print("  - Middle-click + drag: Pan")
    print("  - 'R': Reset view")
    print("  - 'Q' or Esc: Close window")
    
    # Create visualizer with custom settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Room Detection Results",
        width=1400,
        height=900
    )
    
    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.15])  # Dark background
    opt.point_size = point_size
    opt.mesh_show_back_face = True
    
    # Set view to look at the scene from above-ish angle
    ctr = vis.get_view_control()
    
    # Get bounding box to set camera
    bounds = building_mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    size = np.linalg.norm(bounds[1] - bounds[0])
    
    # Set camera to look at center from an angle
    ctr.set_lookat(center)
    ctr.set_zoom(0.5)
    ctr.set_front([0.5, -0.5, 0.7])  # Looking from front-left-above
    ctr.set_up([0, 0, 1])  # Z is up
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()


def create_point_cloud(candidates: List[CandidatePoint], color: list = [255, 0, 0, 255]):
    """Legacy function for trimesh compatibility."""
    import trimesh
    if not candidates:
        return trimesh.PointCloud(vertices=np.zeros((0, 3)))
    
    points = np.array([c.sample_point for c in candidates])
    colors = np.tile(color, (len(points), 1))
    
    return trimesh.PointCloud(vertices=points, colors=colors)


def create_ceiling_markers(
    candidates: List[CandidatePoint],
    marker_size_cm: float = 20.0,
    color: list = [0, 255, 0, 200]
):
    """Legacy function for trimesh compatibility."""
    import trimesh
    
    if not candidates:
        return trimesh.Trimesh()
    
    markers = []
    half_size = marker_size_cm / 2
    
    for c in candidates:
        center = c.top_hit_point.copy()
        
        vertices = np.array([
            [center[0] - half_size, center[1] - half_size, center[2] - 2],
            [center[0] + half_size, center[1] - half_size, center[2] - 2],
            [center[0] + half_size, center[1] + half_size, center[2] - 2],
            [center[0] - half_size, center[1] + half_size, center[2] - 2],
        ])
        
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        marker = trimesh.Trimesh(vertices=vertices, faces=faces)
        markers.append(marker)
    
    if not markers:
        return trimesh.Trimesh()
    
    combined = trimesh.util.concatenate(markers)
    combined.visual.face_colors = np.tile(color, (len(combined.faces), 1))
    
    return combined
