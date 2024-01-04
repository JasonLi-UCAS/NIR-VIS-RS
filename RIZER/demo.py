import os
import time
os.chdir("..")
import warnings
import math
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release...")
os.chdir("..")
from copy import deepcopy

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg

##You should fill in the following paths to run RIZER:1.***/RIZER/weights/outdoor_ds.ckpt
##                                                    2.***/City/VIS_1.png
##                                                    3.***/City/NIR_TF_1.png
##                                                    4.***/RIZER/demo.pdf

start_time = time.time()
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("***/RIZER/outdoor_ds.ckpt")['state_dict'])  ##
matcher = matcher.eval().cuda()
default_cfg['coarse']
# Load example images
img0_pth = "***/City/VIS_1.png"       # input size shuold be divisible by 8
img1_pth = "***/City/NIR_TF_1.png"    # input size shuold be divisible by 8
img0_color = cv2.imread(img0_pth)
img1_color = cv2.imread(img1_pth)
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}

# Inference with LoFTR and get prediction
with torch.no_grad():
    matcher(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    xuhao = np.asarray(batch['xuhao'])

H, status = cv2.findHomography(mkpts0[:xuhao[0],:], mkpts1[:xuhao[0],:])
T = 2
green = (0, 1, 0)  # RGB code for green
red = (1, 0, 0)
def correct_jisuan(mkpts0,mkpts1,xuhao0,xuhao1,H):
    inlier_pts0 = []
    inlier_pts1 = []
    color_cf=[]
    for i in range(xuhao0,xuhao1):
        x1, y1 = mkpts0[i]
        x2, y2 = mkpts1[i]
        x2p, y2p, w = np.dot(H, [x1, y1, 1])
        x2p /= w
        y2p /= w
        dist = np.sqrt((x2p - x2) ** 2 + (y2p - y2) ** 2)
        if dist < T:
            inlier_pts0.append(mkpts0[i])
            inlier_pts1.append(mkpts1[i])
            color_cf.append(green)
        else:
            color_cf.append(red)
    return(np.array(inlier_pts0),np.array(inlier_pts1),np.array(color_cf))
inlier_pts0_1, inlier_pts1_1, color_cf_1 = correct_jisuan(mkpts0,mkpts1,0,xuhao[0],H)
inlier_pts0_2, inlier_pts1_2, color_cf_2 = correct_jisuan(mkpts0, mkpts1,xuhao[0], xuhao[2],H)
inlier_pts0=np.concatenate((inlier_pts0_1, inlier_pts0_2), axis=0)
inlier_pts1 = np.concatenate((inlier_pts1_1,inlier_pts1_2),axis=0)
color_cf=np.concatenate((color_cf_1,color_cf_2),axis=0)
print("Number of Correct Matches：{} ".format(len(inlier_pts0)))

end_time = time.time()
run_time = end_time - start_time
print("Running Time:{}".format(run_time))

# Draw
text = [
    'RIZER',
    'Matches: {}'.format(len(inlier_pts0)),
]
fig = make_matching_figure(img0_color , img1_color, mkpts0,mkpts1, color_cf, text=text,
                           path="***/RIZER/demo.pdf")
