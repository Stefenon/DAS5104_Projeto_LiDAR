import os
import numpy as np

from src.PointCloudReconstructor import PointCloudReconstructor
from src.VolumeCalculator import VolumeCalculator
from src.PointCloudPlotter import PointCloudPlotter
from src.Registration import Registration
from SurfaceReconstructor import SurfaceReconstructor
from src.Constants import Constants
from src.Parameters import Parameters


class DataManager():

    def __init__(self):
        self.point_cloud_plotter = PointCloudPlotter()
        self.volume_calculator = VolumeCalculator()
        self.pcd_reconstructor = PointCloudReconstructor()
        self.registration = Registration()
        self.surface_reconstructor = SurfaceReconstructor()

    def process_data(self, scan_path: str) -> float:
        # if os.path.isfile(f"{scan_path}data.npz"):
        #     xyz = np.load(f"{scan_path}data.npz")["xyz"]
        # else:
        xyz = self.pcd_reconstructor.create_point_cloud(scan_path)
        np.savez_compressed(f"{scan_path}data.npz", xyz=xyz)

        truck_bucket = self.pcd_reconstructor.create_point_cloud(Constants.BUCKET_PATH)

        aligned_pcd = self.registration.align_truck_bucket_and_load(xyz, truck_bucket, Parameters.Registration.VOXEL_SIZE,
                                                                    Parameters.Registration.MAX_ITERATION_RANSAC,
                                                                    Parameters.Registration.CONFIDENCE,
                                                                    Parameters.Registration.MAX_NN_NORMALS,
                                                                    Parameters.Registration.MAX_NN_FPFH,
                                                                    Parameters.Registration.EPSILON,
                                                                    Parameters.Registration.MAX_ITERATION_ICP,
                                                                    Parameters.Registration.RANSAC_LOOP_SIZE)

        load_pcd = self.surface_reconstructor.isolate_load_points(aligned_pcd, truck_bucket,
                                                                  Parameters.BucketRemoval.NB_NEIGHBORS,
                                                                  Parameters.BucketRemoval.STD_RATIO,
                                                                  Parameters.BucketRemoval.NB_POINTS,
                                                                  Parameters.BucketRemoval.RADIUS,
                                                                  Parameters.BucketRemoval.THRESHOLD_DISTANCE)

        full_pcd = self.surface_reconstructor.merge_load_and_bucket_points(load_pcd, truck_bucket,
                                                                           Parameters.MergePoints.DETECTION_THRESHOLD,
                                                                           Parameters.MergePoints.DISTANCE_THRESHOLD,
                                                                           Parameters.MergePoints.ANGULAR_STEP,
                                                                           Parameters.MergePoints.SLOPE,
                                                                           Parameters.MergePoints.NB_NEIGHBORS,
                                                                           Parameters.MergePoints.STD_RATIO)
        
        load_mesh = self.surface_reconstructor.reconstruct_load_mesh(full_pcd, Parameters.MeshReconstruction.ALPHA,
                                                                     Parameters.MeshReconstruction.N_FILTER_ITERATIONS)
