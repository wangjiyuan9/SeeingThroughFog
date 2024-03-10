import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..', '..')
datasetviewer_lib_path = os.path.join(root_dir, 'DatasetViewer', 'lib')
project_lib_path = os.path.join(root_dir, 'ProjectionTools', 'Lidar2RGB', 'lib')
if datasetviewer_lib_path not in sys.path:
    sys.path.append(datasetviewer_lib_path)
if project_lib_path not in sys.path:
    sys.path.append(project_lib_path)

from read import load_velodyne_scan
from read import load_calib_data
from util import filter, find_closest_neighbors, find_missing_points, transform_coordinates
from visi import plot_spherical_scatter_plot, plot_image_projection
import cv2
import numpy as np

import os
import argparse

# changed!! github

notOk = [11, 15, 23, 33, 40, 47, 49, 62, 67, 76, 79, 103, 106, 114, 115, 116, 133, 140, 168, 171, 179, 183, 190, 192, 201, 203, 209, 218, 220, 233, 234, 235, 251, 259, 276, 280, 285, 287, 288, 297, 299, 302, 308, 310, 311, 312, 324, 326, 327, 342, 344, 349, 350, 356, 400, 405, 418, 420, 422, 431, 438, 443, 444, 449, 453, 458, 460, 463, 468, 481, 483, 493, 508, 515, 529, 534, 542, 545, 556, 561, 565, 568, 569, 571]
Ok = [15, 40, 67, 133, 233, 287, 438, 444, 534, 565]


def parsArgs():
    parser = argparse.ArgumentParser(description='Lidar 2d projection tool')
    parser.add_argument('--root', '-r', help='Enter the root folder')
    parser.add_argument('--lidar_type', '-t', help='Enter the root folder', default='lidar_hdl64',
        choices=['lidar_hdl64', 'lidar_vlp32'])
    args = parser.parse_args()

    return args


interesting_samples = []
with open('/opt/data/private/wjy/backup/weather_datasets/dense/SeeingThroughFog/splits/snow_day.txt', 'r') as f:
    interesting_samples += f.read().splitlines()

echos = [
    ['last', 'strongest'],
]


def create_and_project(interesting_sample):
    rgb_file = os.path.join(args.root, 'cam_stereo_left_lut', interesting_sample.replace(',', '_') + '.png')
    rgb_data = cv2.imread(rgb_file)
    velo_file_last = os.path.join(args.root, args.lidar_type + '_' + echos[0][0], interesting_sample.replace(',', '_') + '.bin')
    # velo_file_strongest = os.path.join(args.root, args.lidar_type + '_' + echos[0][1], interesting_sample.replace(',', '_') + '.bin')
    lidar_data_last = load_velodyne_scan(velo_file_last)
    # lidar_data_strongest = load_velodyne_scan(velo_file_strongest)

    print('last shape:', lidar_data_last.shape)
    lidar_data_last = filter(lidar_data_last, 1.5)  # filter out points that are too near
    depth = plot_image_projection(rgb_data, lidar_data_last, vtc, velodyne_to_camera, name=str(idx))
    return depth

def move_image(sample):
    rgb_file = os.path.join(args.root, 'cam_stereo_left_lut', sample.replace(',', '_') + '.png')
    rgb_data = cv2.imread(rgb_file)
    cv2.imwrite('/opt/data/private/wjy/backup/weather_datasets/dense/snowy/{}.png'.format(sample.replace(',', '_')), rgb_data)

if __name__ == '__main__':

    args = parsArgs()

    velodyne_to_camera, camera_to_velodyne, P, R, vtc, radar_to_camera, zero_to_camera = load_calib_data(
        args.root, name_camera_calib='calib_cam_stereo_left.json', tf_tree='calib_tf_tree_full.json',
        velodyne_name='lidar_hdl64_s3_roof' if args.lidar_type == 'lidar_hdl64' else 'lidar_vlp32_roof')
    depth_all = []
    with open('/opt/data/private/wjy/backup/weather_datasets/dense/SeeingThroughFog/snowy.txt', 'w') as f:
        for idx, interesting_sample in enumerate(interesting_samples):
            # x,y, z, intensity, ring?
            if idx % 4 != 0:
                continue
            if (idx / 4 + 1) in notOk:
                if (idx / 4 + 1) in Ok:
                    idx += 1
                    interesting_sample = interesting_samples[idx]
                else:
                    continue
            f.write(interesting_sample + '\n')
            depth = create_and_project(interesting_sample)
            depth_all.append(depth)

    np.save('/opt/data/private/wjy/backup/weather_datasets/dense/SeeingThroughFog/gt_depths.npy', depth_all)
