# Self_Driving_Car Capstone Project 

<p align="center">
  <img src="CarND-Capstone/gif/Simulator.gif">
</p>

## Team Members
The members of team:

| Name                          | GitHub account                                    | Contributions |
|:------------------------------|:--------------------------------------------------| :---------------|
| Marcelo                 |[CheloGE](https://github.com/CheloGE)              |  Perception and Sensor Subsystem               |
| Andres                  |[AndresGarciaEscalante](https://github.com/AndresGarciaEscalante)| Planning and Control Subsystem      |

## Installation:
1. Clone the Repository:
```
git clone https://github.com/AndresGarciaEscalante/Self_Driving_Car
```
2. Install the requirements for the project with the following command lines:
```
cd CarND-Capstone
pip install -r requirements.txt
```
The [CarND-Capstone-Project](https://github.com/AndresGarciaEscalante/Self_Driving_Car/tree/master/CarND-Capstone) README.md in the repository provides a detailed explanation of the installation steps to ***download and execute the simulator***.


## Run Code:

Once the simulator is installed follow the next steps to execute the simulation:

1. In the  ***CarND-Capstone/ros*** folder execute the following commands:

```
cd CarND-Capstone/ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```

2. Run the simulator, enable the camera and disable the manual options when the simulation starts.

## Project Description
Implement a safe autonomous navigation in a simulated 3D environment. In the simulation there will be traffic lights, that must be recognized by the self-driving car using a camera. Based on the state of the traffic light the car keeps moving, reduce the velocity, or stops. The self-driving car needs to be able to not exceed the maximum velocity, not exceed the maximum acceleration, not exceed the maximum jerk, and create smooth trajectories.

The **System Architecture** of the project consists of three main subsystems (Perception, Planning, and Control). The nodes `waypoint_updater.py`, `dbw_node.py`, and `traffic_light_detection_node.py` were implemented to provide a safe autonomous navigation in the environment. 

<p align="center">
  <img width= 500 src="CarND-Capstone/imgs/system_architecture.png">
</p>

The aforementioned nodes are described in more detail in the following subsections:

### Waypoint Updater 
This package contains the waypoint updater node: `waypoint_updater.py`. The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the `/base_waypoints`, `/current_pose`, `/obstacle_waypoint`, and `/traffic_waypoint` topics, and publish a list of waypoints ahead of the car with target velocities to the `/final_waypoints` topic.

<p align="center">
  <img width= 500 src="CarND-Capstone/imgs/waypoint-updater-ros-graph.png">
</p>

### Drive-By-Wire (DBW)
The project is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node `dbw_node.py` and the file `twist_controller.py`, along with a pid and lowpass filter. The dbw_node subscribes to the `/current_velocity` topic along with the `/twist_cmd` topic to receive target linear and angular velocities. Additionally, this node will subscribe to `/vehicle/dbw_enabled`, which indicates if the car is under dbw or driver control. This node will **publish throttle, brake, and steering commands** to the `/vehicle/throttle_cmd`, `/vehicle/brake_cmd`, and `/vehicle/steering_cmd` topics.

<p align="center">
  <img width= 500 src="CarND-Capstone/imgs/dbw-node-ros-graph.png">
</p>

### Traffic Light Detection
This package contains the traffic light detection node: `tl_detector.py`. This node takes in data from the `/image_color`, `/current_pose`, and `/base_waypoints` topics and publishes the locations to stop for red traffic lights to the `/traffic_waypoint` topic.

The `/current_pose` topic provides the vehicle's current position, and `/base_waypoints` provides a complete list of waypoints the car will be following. The **Traffic light detection** is implemented in the `tl_detector.py`, whereas traffic light classification in `../tl_detector/light_classification_model/tl_classfier.py`.

<p align="center">
  <img width= 500 src="CarND-Capstone/imgS/tl-detector-ros-graph.png">
</p>

**IMPORTANT:** Please refer to the following repository for more detailed information of the **Detection and Classification of Traffic Lights** [CarND-Traffic-Light-Classifier](https://github.com/CheloGE/CarND-Traffic-Light-Classifier).

## Project Outcome
The car was able to complete drive autonomously without breaking any of the restrictions.  

**IMPORTANT** Check the full video of the self-driving car navigating in the simulated environment [Self_Driving_Car Capstone Project]().
