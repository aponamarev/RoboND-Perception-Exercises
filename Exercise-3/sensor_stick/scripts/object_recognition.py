#!/usr/bin/env python

import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

import pickle

from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker

from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # Convert ROS msg to PCL data
    pcd = ros_to_pcl(pcl_msg)

    # Voxel Grid Downsampling
    # create filter
    vox = pcd.make_voxel_grid_filter()
    # apply leaf size
    # A voxel grid filter allows you to downsample the data by taking a spatial average of the points in the cloud
    # confined by each voxel.
    # Size is specified for each axis (3D data), with units in meters (1 == 1 meter).
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    # extract downsampled data
    cloud_filtered = vox.filter()

    # PassThrough Filter
    # PassTrhough Filter works as a cut function. If you have a known location of the object, you can cut that object
    # and disregard the rest of the points.
    # create filter
    passthrough = cloud_filtered.make_passthrough_filter()
    # Pass through z axis - specify axis and the range
    filter_axis = 'z'
    axis_min = 0.3
    axis_max = 5.0
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    # filter out points
    cloud_filtered = passthrough.filter()
    # Pass through x axis - specify axis and the range
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    axis_min = 0.34
    axis_max = 1.0
    passthrough.set_filter_field_name(filter_axis)
    passthrough.set_filter_limits(axis_min, axis_max)
    # filter out points
    cloud_filtered = passthrough.filter()

    # RANSAC Plane Segmentation
    # create a segmentation filter
    seg = cloud_filtered.make_segmenter()
    # set the model
    seg.set_model_type(pcl.SACMODEL_PLANE)
    # set segmentation method
    seg.set_method_type(pcl.SAC_RANSAC)
    # set point threshold to be considered an outlier
    max_distance = 0.006
    seg.set_distance_threshold(max_distance)
    # obtain model coefficients and inlier indices
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    # inliers (table)
    table = cloud_filtered.extract(inliers, negative=False)
    # outliers (objects)
    objects = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(objects)
    # Create k-d tree
    # https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/e5bfcfbd-3f7d-43fe-8248-0c65d910345a/lessons/2cc29bbd-5c51-4c3e-b238-1282e4f24f42/concepts/aff79804-e31d-468e-9f12-03536a1b16dc
    # The k-d tree data structure is used in the Euclidian Clustering algorithm to decrease the computational
    # burden of searching for neighboring points.
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold
    ec.set_ClusterTolerance(0.001)
    # as well as minimum and maximum cluster size (in points)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(10000)
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    # Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                             white_cloud[indice][1],
                                             white_cloud[indice][2],
                                             rgb_to_float(cluster_color[j])])

    # Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):

        # Grab the points for the cluster
        pcl_cluster = objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster)

        # Compute the associated feature vector
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction
        prediction = clf.predict(scaler.transform(feature.reshape(1, -1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4

        object_markers_pub.publish(make_label(label, label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('clustering', anonymous=False)

    # Create Subscribers
    pcl_sub = rospy.Subscriber("/sensor_stick/point_cloud", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=10)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=10)

    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
