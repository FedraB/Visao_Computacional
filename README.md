# 3D Pedestrian Detection From Multiple Cameras

This project is part of the Computer Vision course in the Computer Science Bachelor's program at UFRPE. It focuses on detecting pedestrians in 3D using data from multiple cameras. The project uses the [WILDTRACK](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) dataset and the Reconstructing Groups of People with Hypergraph Relational Reasoning [(GroupRec)](https://github.com/boycehbz/GroupRec) algorithm to extract 3D points from the Seven-Camera HD Dataset images. 

The 3D poses of each pedestrian in the scene captured by the 7 cameras are merged based on the algorithm described in the paper Multi-View Multi-Person 3D Pose Estimation with Plane Sweep Stereo [(Plane Sweep Pose)](https://github.com/jiahaoLjh/PlaneSweepPose/tree/main).

## Get Started

To get started with this project, follow these steps:

### Prerequisites

Make sure you have the following installed:

- Python 3.11.9
- NumPy 1.26.4
- OpenCV 4.10.0.84
- Scipy 1.10.1
- Matplotlib 3.9.1

You can install the dependencies using `pip`:

```bash
pip install numpy==1.26.4 opencv-python-headless==4.10.0.84 scipy==1.10.1 matplotlib==3.9.1
