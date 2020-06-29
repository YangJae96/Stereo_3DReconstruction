# Stereo_3DReconstruction
3D Reconstruction from Stereo Images. Our code Computes the disparity using PSMNet Architecture 
Model and then triangulate the 3D Points. Finally, 3D Points are visualized using Open3D.

PSMNet=> " https://github.com/JiaRenChang/PSMNet "

# CapStone Project
This project is for 2020-1 semester Capstone Project. 

## 1. Contributor
- Jae Won Yang, Jae Won U

## 2. Version
- python3.6
- Open3d 0.10
- torch1.5 and torchvision 0.6
- Cuda >=10.1 
- Ubuntu 18.04
- Multiple GPU: Nvidia Titan X (Use Dataparallel GPU Number 2 and 3)

## 3. Build
- git clone "https://github.com/YangJae96/Stereo_3DReconstruction.git"
- Run  "pip3 install -r requirments.txt"

## 4. Run Demo
- Command: "python3 reconstruction.py --folder_name chair"

## 5. Add custom dataset
- Stereo Datasets => "https://vision.middlebury.edu/stereo/data/scenes2014/"
- Download the dataset zip file 
- Create dir inside the dataset dir -> (The dir name will be an argument when you type in the command)
- Put the calib.txt, im0.png, im1.png inside the (Your dataset) dir. 
- Then Run "python3 reconstruction.py --folder_name (Your data dir name inside dataset dir)"
