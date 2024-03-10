import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from util import project_pointcloud
from util import transform_coordinates
import numpy as np
from tools.CreateTFRecords.generic_tf_tools.resize import resize
from filter_pointcloud_1 import do_filter


def plot_spherical_scatter_plot(pointlcoud, pattern='hot', plot_show=True, title=None):
    norm = mpl.colors.Normalize(vmin=0, vmax=80)
    cmap = None
    if pattern == 'hot':
        cmap = cm.hot
    elif pattern == 'cool':
        cmap = cm.cool
    elif pattern == 'jet':
        cmap = cm.jet
    else:
        print('Wrong color map specified')
        exit()
    m = cm.ScalarMappable(norm, cmap)
    transformed_pointcloud = transform_coordinates(pointlcoud)
    depth_map_color = m.to_rgba(transformed_pointcloud[:, 0])

    plt.scatter(transformed_pointcloud[:, 1],
        transformed_pointcloud[:, 2], c=depth_map_color, s=1)

    if title is not None:
        plt.title(title)

    if plot_show:
        plt.show()


def plot_image_projection(rgb, pointcloud, vtc, velodyne_to_camera, frame='default', name=None):
    # Resize image to other crop
    r = resize(frame)
    pointcloud = do_filter(pointcloud)

    lidar_image, depth = project_pointcloud(pointcloud, np.matmul(r.get_image_scaling(), vtc), velodyne_to_camera, list(r.dsize)[::-1] + [3], init=rgb)
    # set size
    plt.figure(figsize=(20, 20))
    plt.imsave('/opt/data/private/wjy/backup/weather_datasets/dense/SeeingThroughFog/images/{}'.format(name + ".png"), lidar_image)
    return depth


def plot_3d_scatter(pointlcoud, plot_show=True):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(pointlcoud[:, 0], pointlcoud[:, 1], pointlcoud[:, 2], c='black', alpha=1, s=0.01)

    if plot_show:
        plt.show()
