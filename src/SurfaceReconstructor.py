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

  def get_max_coordinate_in_plane(points, section, plane='xy', max_axis='y', tolerance=10):
    fixed_axis_idx = 2 if plane == 'xy' else 1 if plane == 'xz' else 0
    max_axis_idx = 0 if max_axis == 'x' else 1 if max_axis == 'y' else 2
    plane_coords = [p[max_axis_idx] for p in points if (section - tolerance) < p[fixed_axis_idx] < (section + tolerance)]
    
    return max(plane_coords)

  def get_min_coordinates(points):
    return min(points, key=lambda p: p[0])[0], min(points, key=lambda p: p[1])[1], min(points, key=lambda p: p[2])[2]

  def get_max_coordinates(poitns):
    return max(poitns, key=lambda p: p[0])[0], max(poitns, key=lambda p: p[1])[1], max(poitns, key=lambda p: p[2])[2]

  def merge_load_and_bucket_points(self, bucket: o3d.geometry.PointCloud, load: o3d.geometry.PointCloud,
                                   detection_threshold: float, distance_threshold: float, angular_step: float,
                                   slope:float, nb_neighbors: int, std_ratio: float) -> o3d.geometry.PointCloud:
    # Define the ray origin and direction
    min_x, _, min_z = self.get_min_coordinates(inner_load_points)
    max_x, _, max_z = self.get_max_coordinates(inner_load_points)
    center_x = (min_x + max_x) / 2

    delta_z = max_z - min_z
    lower_z = min_z + delta_z*0.15
    center_z = (min_z + max_z) / 2
    upper_z = max_z - delta_z*0.15

    lower_y = self.get_max_coordinate_in_plane(inner_load_points, lower_z, 'xy', 'y', 10)*1.2
    center_y = self.get_max_coordinate_in_plane(inner_load_points, center_z, 'xy', 'y', 10)*1.2
    upper_y = self.get_max_coordinate_in_plane(inner_load_points, upper_z, 'xy', 'y', 10)*1.2

    ray_origins = [
    np.array([center_x, lower_y, lower_z]),
    np.array([center_x, center_y, center_z]),
    np.array([center_x, upper_y, upper_z])]
    rays = []

    rays += self.generate_rays_with_slope(angular_step, slope, radius=50)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=200)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=300)
    rays += self.generate_rays_with_slope(angular_step, slope, radius=500)
    near_points = []

    # Get direction as unit vector of each ray
    directions = [ray / np.linalg.norm(ray) for ray in rays]

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
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    return pcd
    
  def reconstruct_load_mesh(self, load: o3d.geometry.PointCloud, alpha: float,
                            n_filter_iterations: int) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(load, alpha)
    bbox = load.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Refine the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=n_filter_iterations)
    mesh.paint_uniform_color([0.7, 0.7, 0.7])
    mesh.compute_triangle_normals()

    return mesh
