import time
import skimage.io
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import argparse
import os
import plyfile
import open3d 
import io
from PSMNet import compute_disparity

parser = argparse.ArgumentParser(description='Stereo 3D Reconstruction')
parser.add_argument('--folder_name', default='chair.png',
                    help='dataset folder name')
# parser.add_argument('--left_image', default='left.png',
#                     help='original left image')
# parser.add_argument('--right_image', default='right.png',
#                     help='original right image')

args = parser.parse_args()
path = 'dataset/'+args.folder_name
ply_name=args.folder_name
image_name=args.folder_name

dataset_left_image =  path + '/im0.png'
dataset_right_image = path + '/im1.png'


read_file = open(path+'/calib.txt',mode='r',encoding='utf-8')

while True:

    line = read_file.readline()

    if not line:
        break

    l = line.split('=')

    if l[0] == 'cam0' :
        focal = float(l[1][1:].split(' ')[0])
    elif l[0] == 'doffs' :
        doff = float(l[1][:-1])
    elif l[0] == 'baseline':
        baseline = float(l[1][:-1])

read_file.close()

focal = focal /5 
baseline = baseline /5



def convert_to_ply(depth, model_3d, name, image, cmp_range, percent=15, downsample_n=0):

    file_path = str(name) + '.ply'
    rows = image.shape[0]
    cols = image.shape[1]

    vertices = [] 
    for p, c, d in zip(model_3d.T, image.reshape(-1, 3), depth.reshape(-1, 1)):
        if d != 0: # ignore points with zero depth
            s = "{} {} {} {} {} {}".format(p[0], p[1], p[2], c[2], c[1], c[0])
            vertices.append(s)
   
    return points_to_ply_string(vertices)       


def ply_header(count_vertices, with_normals=False):
    if with_normals:
        header = [
            "ply",
            "format ascii 1.0",
            "element vertex {}".format(count_vertices),
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    else:
        header = [
            "ply",
            "format ascii 1.0",
            "element vertex {}".format(count_vertices),
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    return header


def points_to_ply_string(vertices):
    header = ply_header(len(vertices))
    return '\n'.join(header + vertices + [''])

def open_wt(path):
    """Open a file in text mode for writing utf-8."""
    return io.open(path, 'w', encoding='utf-8')  

def create_depthMap(disparity):
    
    disparity=disparity
    depth =(baseline * focal /(disparity+doff))
    
    return depth

def create_3D(image,depth):
    height, width, _ = image.shape
    rows = image.shape[0]
    cols = image.shape[1]

    temp_points = np.ndarray((height,width,3))    
    for i in range(0, rows):
    	for j in range(0, cols):
    		z = depth[i,j] 
    		temp_points[i,j,0] = i * z / focal
    		temp_points[i,j,1] = j * z / focal
    		temp_points[i,j,2] = z
    
    points = np.vstack((temp_points[:,:,0].flatten(), temp_points[:,:,1].flatten(), temp_points[:,:,2].flatten()))
    return points

if __name__ == '__main__':
    start=time.time()
    
    left_image = cv.imread(dataset_left_image)
    right_image = cv.imread(dataset_right_image)# big image

    smallLeft_image = cv.resize(left_image,dsize=(0,0),fx=0.2,fy=0.2,interpolation=cv.INTER_AREA)
    smallRight_image = cv.resize(right_image,dsize=(0,0),fx=0.2,fy=0.2,interpolation=cv.INTER_AREA)

    disparity = compute_disparity.get_disparity(smallLeft_image, smallRight_image)
    disparity =disparity /50

    depth = create_depthMap(disparity)
    model3D_matrix = create_3D(smallLeft_image,depth)

    print("Conversion to PLY\n")
    ply = convert_to_ply(depth=depth, model_3d= model3D_matrix, name=ply_name,
     image=smallLeft_image, cmp_range=70)
    print("Conversion to PLY Completed\n")
    recon_time=time.time()-start
    print("Reconstruction Time == {:.0f}m {:.0f}s\n".format(recon_time//60, recon_time%60))
    print("Starting Model Visualization")

    with open_wt(path+'/'+ ply_name+'.ply') as fout:
        fout.write(ply)

    file_path=path +'/'+ply_name+'.ply'
    pcd = open3d.io.read_point_cloud(file_path)
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    open3d.visualization.draw_geometries([pcd])
    