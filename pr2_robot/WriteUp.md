# Project 3: 3D Perception in a Cluttered Space

#### Udacity Robotics Nanodegree

January 2019

Overview
The Pick and Place project 

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.






##  Confusion Matrices

World #1 Not Normalized             |  World #1 Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/50670459-c02b8000-0f99-11e9-8ad4-f10e920bc7f1.png)  |  ![](https://user-images.githubusercontent.com/487926/50670474-dcc7b800-0f99-11e9-9d86-201cfb9c50f7.png)

text
World #2 Not Normalized             |  World #2 Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/50670493-f7019600-0f99-11e9-8348-331aa6a2ac97.png)  |  ![](https://user-images.githubusercontent.com/487926/50670481-e5b88980-0f99-11e9-9e67-7e1e24d91b1a.png)

text

World #3 Not Normalized             |  World #3 Normalized
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/487926/50670496-fec13a80-0f99-11e9-9c28-526a42a327aa.png)  |  ![](https://user-images.githubusercontent.com/487926/50670486-ec470100-0f99-11e9-866a-62c55b636563.png)



### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.


Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  

Perception Results

## Perception World #1
![](https://user-images.githubusercontent.com/487926/50670518-2adcbb80-0f9a-11e9-8f2a-332ba68c5651.png)


## Perception World #2
![](https://user-images.githubusercontent.com/487926/50670511-18fb1880-0f9a-11e9-82c5-d6d0e49efc68.png)



## Perception World #3
![](https://user-images.githubusercontent.com/487926/50670506-0f71b080-0f9a-11e9-8c6e-f45d120f6d55.png)