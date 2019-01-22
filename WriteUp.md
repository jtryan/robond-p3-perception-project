

# Project 3: 3D Perception in a Cluttered Space

### Udacity Robotics Nanodegree

January 2019
The Pick and Place project

## Overview

The `Pick and Place Project` goal is to develop a perception pipeline that allows a RGB - 3D camera to identify an object on a table so it can be picked up with a PR2 robot. This task can be acheived by breaking the problem into a number of steps. The first step is to develop a model that can be accessed in order to identify objects. Semantic Segmantation works by dividing a set of data into clusters that belong to the same object. I spent time taking pictures of an object in different perspectives. The data from the pictures was converted into point cloud data, (pcd). The pcd was converted into histograms that were used to identify the objects. After that step the results were trained using functions from the Sk Learn python library. This resulted in a model that was then saved to be used by the PR2 Robot.

For the RP2 robot to functiom, the process that created the model is then used again to process the camera results in real-time while viewing the presented objects. Once an object has been processsed it is checked against the model and an identification is made by the program to pick the object.

## 1. Perform filtering

The pr2_robot has a camera that converts the images on the target table and creates a point cloud which is sent to the running program that identifies the objects. After receiving the point cloud data it must be filtered to remove noise and make it easier to identify. The point cloud data is very dense with a high level of pixels. Performing calulations on this much data will be slower and may not iimprove results. The first step to deal with this is to perform Voxel Grid Downsampling. This removes extra point cloud data yet leaves enough that the reults are not affected. The following code snippets show how to acheive this.

```python
# Voxel Grid Downsampling
vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.003
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = vox.filter()
```

### Noise Filter
The data is then run through a statistical outlier filter to remove noise for the data. Though there is no noise in the simultor, this is a good practice for real life.

```python
# Statistical Outlier Filtering
outlier_filter = cloud_filtered.make_statistical_outlier_filter()
# Set the number of neighboring points to analyze for any given point
outlier_filter.set_mean_k(50)
# Set threshold scale factor
x = 1.0
# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)
# Finally call the filter function for magic
cloud_filtered = outlier_filter.filter()


### PassThrough Filter z-axis
passThrough = cloud_filtered.make_passthrough_filter()

# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passThrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passThrough.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passThrough.filter()

### Passthrough Filter y-axis
passThrough = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the passthrough filter object.
filter_axis = 'y'
passThrough.set_filter_field_name(filter_axis)
axis_min = -0.5
axis_max = 0.5
passThrough.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passTh
# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
outlier_filter.set_std_dev_mul_thresh(x)
# Finally call the filter function for magic
cloud_filtered = outlier_filter.filter()
```

### Passthrough filters
The data is next run through a passthrough filter on the 'z' and 'y' axes. A passthrough filter removes data that is above or below a given range. Filtering on the `z` axis will remove the table form the data. Filtering on the `y` axis removes the the boxes from the data. This makes the object recognition more effective.

```python
 ### PassThrough Filter z-axis
passThrough = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the passthrough filter object.
filter_axis = 'z'
passThrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passThrough.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passThrough.filter()

### Passthrough Filter y-axis
passThrough = cloud_filtered.make_passthrough_filter()
# Assign axis and range to the passthrough filter object.
filter_axis = 'y'
passThrough.set_filter_field_name(filter_axis)
axis_min = -0.5
axis_max = 0.5
passThrough.set_filter_limits(axis_min, axis_max)
# Finally use the filter function to obtain the resultant point cloud. 
cloud_filtered = passThrough.filter()
```

### RANSAC Filter

Random Sample Consensus filtering or RANSAC is a method that can separate the data into outliers and inliers if you have an idea of the shaoe of your data. In this case the goal is to identify the plane that is the table so it can be filtered out of the clouda. 

```python

# Create the segmentation object
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
max_distance = 0.01
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

### Extract inliers and outliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
```

The next step in the pipeline is to perform clustering on the data so it is identifiable.

## 2. Perform cluster segmantation

In order to perform clustering I began using PCL Euclidean Clustering algorithm. To perform the Euclidena clustering I converted the data to remove the color information. Then I created a kd tree from the data.


```python
# Go from XYZRGB to RGB since to build the k-d tree we only need spatial data
white_cloud = XYZRGB_to_XYZ(extracted_outliers)
# Apply function to convert XYZRGB to XYZ
tree = white_cloud.make_kdtree()
```

I then took the treee and performed cluster extraction with the result containing a list of the cluster indices.

```python
 ### Create a cluster extraction object
ec = white_cloud.make_EuclideanClusterExtraction()
# Set tolerances for distance threshold 
# as well as minimum and maximum cluster size (in points)
ec.set_ClusterTolerance(0.02) # 0.02
ec.set_MinClusterSize(100) # 50
ec.set_MaxClusterSize(50000) # 50000

# Search the k-d tree for clusters
ec.set_SearchMethod(tree)

# Extract indices for each of the discovered clusters
cluster_indices = ec.Extract()
```
I then used the cluster indices and applied a color to each cluster object. This enabled the objects to be indentified and labeled.

```python
# Classify the clusters! (loop through each detected cluster one at a time)
detected_objects_labels = []
detected_objects = []

for index, pts_list in enumerate(cluster_indices):
	# Grab the points for the cluster from the extracted outliers (cloud_objects)
	pcl_cluster = extracted_outliers.extract(pts_list)
	ros_cluster = pcl_to_ros(pcl_cluster)

	# Extract histogram features
	chists = compute_color_histograms(ros_cluster, using_hsv=True)
	normals = get_normals(ros_cluster)
	nhists = compute_normal_histograms(normals)
	feature = np.concatenate((chists, nhists))

	# Make the prediction, retrieve the label for the result
	# and add it to detected_objects_labels list
	prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
	label = encoder.inverse_transform(prediction)[0]
	detected_objects_labels.append(label)

	# Publish a label into RViz
	label_pos = list(white_cloud[pts_list[0]])
	label_pos[2] += .4
	object_markers_pub.publish(make_label(label,label_pos, index))

	# Add the detected object to the list of detected objects.
	do = DetectedObject()
	do.label = label
	do.cloud = ros_cluster
	detected_objects.append(do)
```

## 3. Extract features and perform SVM training 

 I then published the detected objects so they could be used by the Pr2 robot for idntifying them to pick. 
 
 ```python
 detected_objects_pub.publish(detected_objects)
 ```

### Make your model

Before the project can be run the objects need to be modeled. I used the Gazebo simulator to take multiple pictures of objects that could be placed on a table to be chosen by a robot arm. The modeling is done by running a script that presents multiple views of the objects to the camera. As the data is being read histogram features are extracted, then labeled then saved to an output file. Color Histograms based on the RGB channels were created as well as normalized histograms over the x,y,z axis. 

The code for the color histogram is this:

```python
def compute_color_histograms(cloud, using_hsv=False):
	nbins = 64
	nrange = (0, 265)

	# Compute histograms for the clusters
	point_colors_list = []

	# Step through each point in the point cloud
	for point in pc2.read_points(cloud, skip_nans=True):
		rgb_list = float_to_rgb(point[3])
		if using_hsv:
			point_colors_list.append(rgb_to_hsv(rgb_list) * 255)
		else:
			point_colors_list.append(rgb_list)

	# Populate lists with color values
	channel_1_vals = []
	channel_2_vals = []
	channel_3_vals = []

	for color in point_colors_list:
		channel_1_vals.append(color[0])
		channel_2_vals.append(color[1])
		channel_3_vals.append(color[2])
	
	# TODO: Compute histograms
	r_hist = np.histogram(channel_1_vals, bins=nbins, range=nrange)
	g_hist = np.histogram(channel_2_vals, bins=nbins, range=nrange)
	b_hist = np.histogram(channel_3_vals, bins=nbins, range=nrange)

	# TODO: Concatenate and normalize the histograms
	hist_features = np.concatenate((r_hist[0], g_hist[0], b_hist[0])).astype(np.float64)

	normed_features = hist_features / np.sum(hist_features)
	return normed_features 
```

To create the normallized histograms my code is this:

```python
def compute_normal_histograms(normal_cloud):
	nbins = 64
	nrange = (0, 256)

	norm_x_vals = []
	norm_y_vals = []
	norm_z_vals = []

	for norm_component in pc2.read_points(normal_cloud,
						field_names = ('normal_x', 'normal_y', 'normal_z'),
						skip_nans=True):
		norm_x_vals.append(norm_component[0])
		norm_y_vals.append(norm_component[1])
		norm_z_vals.append(norm_component[2])

	# TODO: Compute histograms of normal values (just like with color)
	x_hist = np.histogram(norm_x_vals, bins=nbins, range=nrange)
	y_hist = np.histogram(norm_x_vals, bins=nbins, range=nrange)
	z_hist = np.histogram(norm_x_vals, bins=nbins, range=nrange)

	# TODO: Concatenate and normalize the histograms
	hist_features = np.concatenate((x_hist[0], x_hist[0], z_hist[0])).astype(np.float64)


	# Generate random features for demo mode.  
	# Replace normed_features with your feature vector
	# normed_features = np.random.random(96)
	normed_features = hist_features / np.sum(hist_features)

	return normed_features
```

These histograms help to create the model that is used in this project. The model uses segmantic segmantation to classify objects by the histograms they produce. I then ran a training script using the sklearn python library. This script produced a model file to be used in the project and it produced graphs showing the effectiveness. There are two graphs, this first showing the total number of good selections while the second shows the the data normalized.

Training Model Not Normalized             |  Training Model Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/51212815-702cb180-18e7-11e9-98bd-18340b8e30bc.png)  |  ![](https://user-images.githubusercontent.com/487926/51212900-aa964e80-18e7-11e9-9a83-7f1c08a1dbdc.png)


## Pick and Place

The `detected_objects` list was snet to the `prs_mover` function. Two lists were populated with the dropbox data and the object data. The object data contained informtaion about the objects that were placed on the table. 

```python
# Get/Read parameters
object_list_param = rospy.get_param('/object_list')
box_param = rospy.get_param('/dropbox')
```

The box name, group and postion were saved so that the arm knew where to drop the object. 

```python
box_name = []
box_group = []
box_position = []
# We'll loop through the two boxes
for i in range(0, len(box_param)):
	box_name.append(box_param[i]['name'])
	box_group.append(box_param[i]['group'])
	box_position.append(box_param[i]['position'])
```

The object data was placed into dictionaries for the labels and centroids of the objects.

```python
for object in object_list:
	labels.append(object.label)
	points_arr = ros_to_pcl(object.cloud).to_array()
	centroids.append(np.mean(points_arr, axis=0)[:3])
```

By reading the object_group for an object I could select the arm that needed to be activated for the pick operation.

```python
# arm_name - Right for Green Box, Left for Red Box
if object_group == 'red':
	arm_name.data = 'left'
	place_pose.position.x = box_position[0][0]
	place_pose.position.y = box_position[0][1]
	place_pose.position.z = box_position[0][2]
else:
	arm_name.data = 'right'
	place_pose.position.x = box_position[1][0]
	place_pose.position.y = box_position[1][1]
	place_pose.position.z = box_position[1][2]   
```

Then the informaton was used to call the needed services and perfomr the pick and place operation.

```python
# pick_pose
pick_pose = Pose()
desired_object = object_list_param[i]['name']
print "\n\nPicking up ", desired_object, "\n\n"
print "\n\nPutting in ", object_group, "\n\n"

# Match desired object with the centroid list/labels
try:
	labelPosition = labels.index(desired_object)
	pick_pose.position.x = np.asscalar(centroids[labelPosition][0])
	pick_pose.position.y = np.asscalar(centroids[labelPosition][1])
	pick_pose.position.z = np.asscalar(centroids[labelPosition][2])
except ValueError:
	continue

# Populate various ROS messages
yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
dict_list.append(yaml_dict)

# Wait for 'pick_place_routine' service to come up
rospy.wait_for_service('pick_place_routine')
```

This code shows perfomring the operation with error checking on failure. And the ouptu to the yaml files.

```python
try:
	pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

	# Insert your message variables to be sent as a service request
	resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

	print ("Response: ",resp.success)

except rospy.ServiceException, e:
	print "Service call failed: %s"%e

# Output your request parameters into output yaml file
yaml_filename = 'output_{}.yaml'.format(test_scene_num.data)
send_to_yaml(yaml_filename, dict_list)
```

The `yaml` files are found at `RBD-P3_PERCEPTION-PROJECT/output`

[output_1.yaml](./output/output_1.yaml)

[output_2.yaml](./output/output_2.yaml)

[output_3.yaml](./output/output_3.yaml)


## Perception Results 
Using the model I created and running the simulator I received the following results for the obgject recognition pipeline.

### Perception World #1
![](https://user-images.githubusercontent.com/487926/51214328-97857d80-18eb-11e9-817b-7517a6b42c76.png)


### Perception World #2
![](https://user-images.githubusercontent.com/487926/51214341-9fddb880-18eb-11e9-8824-a6de5b5d7027.png)


### Perception World #3
![](https://user-images.githubusercontent.com/487926/51214354-a9672080-18eb-11e9-8f28-0bdf884a9e5b.png)


## Project Discussion

This was a fun project. Semantic Segmantation is a low cost machine learning technique that works well to identify a known object. The code for sections #1, #2, and #3 had been written previously in earlier labs so implementing it was straight forward. I added an outlier filter for noise as suggested. I did not feel that it added anythng to the performance. I ran the project with and wiht out that seciton of code My results dod not vary. It may be that there is not much noise while using the simulator and I would see results in the real world. 

I found it very interesting how small value changes in cxelould affect the results. For example in Euclidean Clustering, changing values for `ec.set_ClusterTolerance(0.02)`, `ec.set_MinClusterSize(100)`, and `ec.set_MaxClusterSize(50000)` could greatly change the accuracy of object identification, usually for the worse.

I had some difficulty in the last code section. I didn't understand what data was in the object lists initially. I used print statement to debug this information. I think the ROS environment needs better debugging or I need more understanding of ROS. 

I was pleased with the results and I beleive my pipeline worked very well. I did not like the operation of the simulator at all. I think improving the code would be a good challenge for me in the future.