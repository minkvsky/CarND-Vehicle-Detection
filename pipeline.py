import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import time
import pickle
import glob
from PIL import Image

from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from utils import *

# load model
dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )

svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
color_space = dist_pickle['color_space']
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

# data for analysis
labels_num_list = []


# function pipeline
def pipeline(img):

    ystart = 400
    ystop = 656
    scale = 1.5

    track_records = {}
    labels_boxes = []
    labels_num = None

    out_img, labels = find_cars(img, ystart, ystop, scale, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    # save unusual for tuning
    if labels[1] > 2:
        # img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])
        img_name = str(time.time())
        im = Image.fromarray(img)
        im.save("unusual_images/unusual_{}.jpg".format(img_name))
    labels_num_list.append(labels[1])

    boxes = labels2boxes(labels)
    # print(boxes)

    labels_num = labels[1]
    track_records['labels_boxes'] = labels_boxes
    track_records['labels_num'] = labels_num

    return(out_img)

if not os.path.exists('unusual_images'):
	os.mkdir('unusual_images')

# video preprocessing
input_video = 'test_video.mp4'
# input_video = 'project_video.mp4'
# clip = VideoFileClip(input_video).subclip(39,43)
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(pipeline)
output_clip.write_videofile('output_' + input_video, audio=False)

data_analysis = {}
data_analysis['labels_num_list'] = labels_num_list
# data_analysis['labels_boxes'] = labels_boxes
with open('data_analysis.p', 'wb') as f:
    pickle.dump(data_analysis, f)
