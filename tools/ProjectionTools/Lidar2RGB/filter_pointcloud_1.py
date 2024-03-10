# coding=utf-8
# Implementation of DROR in python
# https://github.com/nickcharron/lidar_snow_removal/blob/master/src/DROR.cpp
# Slightly modified due to radiusSearch not being implemented in python-pcl

import numpy as np
import pcl
# import pcl.pcl_visualization
import sys, math


# seq = sys.argv[1] #'0006'
# print(seq)

def dror_filter(input_cloud, output_cloud):
    radius_multiplier_ = 6
    azimuth_angle_ = 0.16  # 0.04
    min_neighbors_ = 2
    k_neighbors_ = min_neighbors_ + 1
    min_search_radius_ = 0.04

    filtered_cloud_list = []

    # init. kd search tree
    kd_tree = input_cloud.make_kdtree_flann()

    # Go over all the points and check which doesn't have enough neighbors
    # perform filtering
    for p_id in range(input_cloud.size):
        x_i = input_cloud[p_id][0]
        y_i = input_cloud[p_id][1]
        range_i = math.sqrt(pow(x_i, 2) + pow(y_i, 2))
        search_radius_dynamic = \
            radius_multiplier_ * azimuth_angle_ * 3.14159265359 / 180 * range_i

        if (search_radius_dynamic < min_search_radius_):
            search_radius_dynamic = min_search_radius_

        [ind, sqdist] = kd_tree.nearest_k_search_for_point(input_cloud, p_id, k_neighbors_)

        # Count all neighbours
        neighbors = -1  # Start at -1 since it will always be its own neighbour
        for val in sqdist:
            if math.sqrt(val) < search_radius_dynamic:
                neighbors += 1;

        # This point is not snow, add it to the filtered_cloud
        if (neighbors >= min_neighbors_):
            filtered_cloud_list.append(output_cloud[p_id]);
            # print(filtered_cloud_list)

    return np.array(filtered_cloud_list, dtype=np.float32)


def crop_cloud(input_cloud):
    clipper = input_cloud.make_cropbox()
    clipper.set_Translation(0, 0, 0)  # tx,ty,tz
    clipper.set_Rotation(0, 0, 0)  # rx,ry,rz
    min_vals = [-4, -4, -3, 0]  # x,y,z,s
    max_vals = [4, 4, 10, 0]  # x,y,z,s
    clipper.set_MinMax(min_vals[0], min_vals[1], min_vals[2], min_vals[3], max_vals[0], max_vals[1], max_vals[2], max_vals[3])
    return clipper.filter()


def do_filter(lidar):
    # Convert lidar 2d array to pcl cloud
    lidarIn = lidar[:, 0:3]
    point_cloud = pcl.PointCloud()
    point_cloud.from_array(lidarIn)
    # Crop the pointcloud to around autnomoose

    lidar_filtered = dror_filter(point_cloud, lidar)
    number_snow_points = point_cloud.size - lidar_filtered.shape[0]
    print number_snow_points,
    # lidar_filtered = np.array(lidar_filtered)
    return lidar_filtered


BASE = "/opt/data/private/wjy/backup/weather_datasets/cadcd/2019_02_27/"

LOOP = False

# for frame in range(21, 83):
# print_snow_points(2)
# Low 74
# Medium 81
# High 80
# High example 68
