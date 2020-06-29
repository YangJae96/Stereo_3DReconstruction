# Stereo_3DReconstruction

# CapStone Project
2020학년도 1학기 캡스톤디자인 프로젝트로 Stereo 이미지에 대한 3D Reconstruction을 만들기를 위한 레포지토리입니다.

## 1. Contributor
- 양재원

## 2. Version
- python3.6
- torch1.5 and torchvision 0.6
- Cuda >=10.1 
- Ubuntu 18.04
- Multiple GPU: Nvidia Titan X

## 3. Build
- git clone "https://github.com/YangJae96/Stereo_3DReconstruction.git"
- Run  "pip3 install -r requirments.txt"

## 4. Run Demo
- Command: python3 reconstruction.py --folder_name chair

## 5. Add custom dataset
- Stereo Datasets => "https://vision.middlebury.edu/stereo/data/scenes2014/"
- Download the dataset zip file 
- Create dir -> (The dir name will be an argument when you type in the command)
- Put the calib.txt, im0.png, im1.png inside the (Your dataset) dir. 
- Then Run "python3 reconstruction.py --folder_name (Your data dir name)"
