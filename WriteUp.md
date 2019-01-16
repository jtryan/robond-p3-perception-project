
[rotx]: ./misc_images/rot-x.png
[roty]: ./misc_images/rot-y.png
![Arm running][run]


# Project 3: 3D Perception in a Cluttered Space

#### Udacity Robotics Nanodegree

January 2019
The Pick and Place project

Overview

The Pick and Place Project has a goal to develop a perception pipeline that allows a RGB - 3D camera to identify object on a table that can bve picked up with a pr2 robot.  This task can be acheived by breaking the problem into a number of steps. I used the Gazebo simulator to take multiple pictures of objects that could be placed on a table to be chosen by a robot arm. I then ran a training script using the sklearnpython library. This script produced a model file to be used in the project and it produced graphs showing the effectiveness. There are two graphs, this first showing the total number of good selections while the second shows the the data normalized.

Training Model Not Normalized             |  Training Model Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/51212815-702cb180-18e7-11e9-98bd-18340b8e30bc.png)  |  ![](https://user-images.githubusercontent.com/487926/51212900-aa964e80-18e7-11e9-9a83-7f1c08a1dbdc.png)


### Make your model
Before the project can be run the objects need to be modeled. The modeling is done by running a script that presents multiple views of the objects to the camera. As the data is being read histogram features are extracted, then labeled then saved to an output file. Color Histograms based on the RGB channels were created as well as normalized histograms over the x,y,z axis. 

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

	# Generate random features for demo mode.  
	# Replace normed_features with your feature vector
	# normed_features = np.random.random(96) 
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

These histograms help to create the model that is used in this project. The model uses segmantic segmantation to classify objects by the histograms they produce. 


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
```

### Passthrough filters
The data is next run through a passthrough filter on the 'z' and 'y' axes. A passthrough filter removes data that is above or below a given range. it is used to remove the table and the boxes from the data to make it easier for object recognition.

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

In order to perform clustering I began using PCL Euclidean Clustering algorithm.
#### Ex2 - segmantyic segmantion overview
### some pictures for prev labs

3. Extract features and perform SVM training 



### some pictures for prev labs

### Ex 3 Overview


World #1 Not Normalized             |  World #1 Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/51212815-702cb180-18e7-11e9-98bd-18340b8e30bc.png)  |  ![](https://user-images.githubusercontent.com/487926/51212900-aa964e80-18e7-11e9-9a83-7f1c08a1dbdc.png)


Using the model I created and running the simkulator I recevied the following results for the obgject recognition pipeline.

## Perception Results 
Using the model I created and running the simkulator I recevied the following results for the obgject recognition pipeline.

### Perception World #1
![](https://user-images.githubusercontent.com/487926/51214328-97857d80-18eb-11e9-817b-7517a6b42c76.png)


### Perception World #2
![](https://user-images.githubusercontent.com/487926/51214341-9fddb880-18eb-11e9-8824-a6de5b5d7027.png)


### Perception World #3
![](https://user-images.githubusercontent.com/487926/51214354-a9672080-18eb-11e9-8f28-0bdf884a9e5b.png)

yaml file links


### Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  