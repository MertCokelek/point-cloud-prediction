import os
import glob
import numpy as np
import shutil
from pcf.utils.utils import range_projection


if __name__ == '__main__':
    # Load and project a point cloud
    FOV_DOWN = -25.0
    FOV_UP = 3.0
    HEIGHT = 64
    MAX_RANGE = 80.0
    WIDTH = 2048

    path_root = "/home/mcokelek21/datasets/KITTI-Raw"
    out_path = "/h"
    for day in os.listdir(path_root):
        print('Processing day:', day)
        for drive in os.listdir(f"{path_root}/{day}"):
            print('\tProcessing drive:', drive)
            velodyne_points = sorted(glob.glob(f"{path_root}/{day}/{drive}/velodyne_points/data/*.bin"))
            for path in velodyne_points:
                current_vertex = np.fromfile(path, dtype=np.float32)
                current_vertex = current_vertex.reshape((-1, 4))
                proj_range, proj_vertex, proj_intensity, proj_idx = range_projection(
                    current_vertex,
                    fov_up=FOV_UP,
                    fov_down=FOV_DOWN,
                    proj_H=HEIGHT,
                    proj_W=WIDTH,
                    max_range=MAX_RANGE,
                )
                dst_folder = f"{path_root}/{day}/{drive}/velo_preprocessed/"
                shutil.rmtree(dst_folder, ignore_errors=True)
                # Save range
                idx = os.path.basename(path).split('.')[0]
                dst_path_range = os.path.join(dst_folder, "range")
                if not os.path.exists(dst_path_range):
                    os.makedirs(dst_path_range)
                file_path = os.path.join(dst_path_range, idx)
                np.save(file_path, proj_range)

                # Save xyz
                dst_path_xyz = os.path.join(dst_folder, "xyz")
                if not os.path.exists(dst_path_xyz):
                    os.makedirs(dst_path_xyz)
                file_path = os.path.join(dst_path_xyz, idx)
                np.save(file_path, proj_vertex)

                # Save intensity
                dst_path_intensity = os.path.join(dst_folder, "intensity")
                if not os.path.exists(dst_path_intensity):
                    os.makedirs(dst_path_intensity)
                file_path = os.path.join(dst_path_intensity, idx)
                np.save(file_path, proj_intensity)