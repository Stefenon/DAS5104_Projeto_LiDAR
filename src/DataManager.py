import os
import numpy as np

from src.PointCloudReconstructor import PointCloudReconstructor
from src.VolumeCalculator import VolumeCalculator
from src.PointCloudPlotter import PointCloudPlotter
from src.Registration import Registration
from src.Constants import Constants
from src.Parameters import Parameters


class DataManager():

    def __init__(self):
        self.point_cloud_plotter = PointCloudPlotter()
        self.volume_calculator = VolumeCalculator()
        self.pcd_reconstructor = PointCloudReconstructor()
        self.registration = Registration()

    def process_data(self, scan_path: str) -> float:
        # if os.path.isfile(f"{scan_path}data.npz"):
        #     xyz = np.load(f"{scan_path}data.npz")["xyz"]
        # else:
        xyz = self.pcd_reconstructor.create_point_cloud(scan_path)
        np.savez_compressed(f"{scan_path}data.npz", xyz=xyz)
