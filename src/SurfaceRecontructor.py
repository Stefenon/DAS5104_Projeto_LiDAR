import open3d as o3d
import numpy as np

from src.Parameters import Parameters

class SurfaceReconstructor():
  def isolate_load_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud, nb_neighbors: int,
                          std_ratio: float, nb_points: int, radius: float, threshold_distance: float
                          ) -> o3d.geometry.PointCloud:
    kd_tree = o3d.geometry.KDTreeFlann(bucket)
    inner_load_points = []

    for point in load.points:
        [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
        closest_point = bucket.points[idx[0]]
        if np.linalg.norm(np.array(point) - np.array(closest_point)) > threshold_distance:
            inner_load_points.append(point)

    removed_points = o3d.geometry.PointCloud()
    removed_points.points = o3d.utility.Vector3dVector(inner_load_points)

    inner_load, _ = removed_points.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                              std_ratio=std_ratio)
    inner_load, _ = inner_load.remove_radius_outlier(nb_points=nb_points, radius=radius)
    
    return inner_load
  
  def point_to_line_distance(points, origin, direction):
    point_vecs = points - origin
    cross_prods = np.cross(direction, point_vecs)
    distances = np.linalg.norm(cross_prods, axis=1) / np.linalg.norm(direction)
    return distances
  
  def find_points_near_ray(self, load: o3d.geometry.PointCloud, bucket: o3d.geometry.PointCloud, ray_origin: list,
                           ray_direction: np.ndarray, detection_threshold: float):
      near_points = []
      for point in bucket.points:
          distance = self.point_to_ray_distance(point, ray_origin, ray_direction)
          if distance <= detection_threshold:
              for load_point in load.points:
                  distance = self.point_to_ray_distance(load_point, ray_origin, ray_direction)
                  if distance <= detection_threshold:
                      near_points.append(point)
                      break
      return near_points

  def generate_rays_with_slope(angular_step: float, slope: float, radius: float) -> list:
      rays = []
      angles = np.arange(0, 360, angular_step)
      for angle in angles:
          rad = np.deg2rad(angle)
          x = np.cos(rad) * radius
          z = np.sin(rad) * radius
          y = -slope
          direction = np.array([x, y, z])
          rays.append(direction)
      return rays

  def merge_load_and_bucket_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud,
                                   detection_threshold: float, distance_threshold: float, angular_step: float,
                                   slope:float) -> o3d.geometry.PointCloud:
    # Define the ray origin and direction
    ray_origins = [np.array([11.5, 1000, -1800]), np.array([11.5, 800, -1300]), np.array([11.5, 800, -2300])]
    rays = []

    rays += self.generate_rays_with_slope(angular_step, slope, radius=50)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=200)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=300)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=500)
    near_points = []

    # Get direction as unit vector of each ray
    directions = [ray / np.linalg.norm(ray) for ray in rays]
    threshold = 20

    bucket_points = np.asarray(bucket.points)
    inner_load_points = np.asarray(load.points)

    # Iterate over each direction to find the lines that meet the threshold criteria
    valid_lines = []
    for origin in ray_origins:
        for direction in directions:
            bucket_distances = self.point_to_line_distance(bucket_points, origin, direction)
            load_distances = self.point_to_line_distance(inner_load_points, origin, direction)
            
            if np.any(bucket_distances < detection_threshold) and np.any(load_distances < detection_threshold):
                valid_lines.append(direction)
                
    near_points = []
    for origin in ray_origins:
        for direction in valid_lines:
            bucket_distances = self.point_to_line_distance(bucket.points, origin, direction)
            close_points = bucket_points[np.where(bucket_distances < detection_threshold)]
            near_points.extend(close_points)

    near_pcd = o3d.geometry.PointCloud()
    near_pcd.points = o3d.utility.Vector3dVector(near_points)
    kd_tree = o3d.geometry.KDTreeFlann(near_pcd)
    inner_bucket_points = []

    for point in bucket.points:
        [_, idx, _] = kd_tree.search_knn_vector_3d(point, 1)
        closest_point = near_pcd.points[idx[0]]
        if np.linalg.norm(np.array(point) - np.array(closest_point)) < distance_threshold:
            inner_bucket_points.append(point)
            
    points = np.concatentate((np.asarray(inner_bucket_points), np.asarray(load.points)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd
    
  def reconstruct_load_mesh(self, load: o3d.geometry.PointCloud, radius: float, max_nn: int, graph_knn: int,
                            n_filter_iterations: int) -> o3d.geometry.TriangleMesh:

    load.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    load.orient_normals_consistent_tangent_plane(graph_knn)
    
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(load)
    bbox = load.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Refine the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)

    # Save and visualize the mesh
    o3d.io.write_triangle_mesh("load.ply", mesh)
    return mesh
