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
import shutil

from moviepy.editor import VideoFileClip
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from utils import *

# lane line
from camera import *
from line import *

labels_num_list = []

# function pipeline
def pipeline(img, isvideo=True):

    # lane line detect
    try:
        l = Line(img, auto=True)
    except Exception as e:
        print (str(e))
        img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])
        im = Image.fromarray(img)
        im.save("unusual_images/error_{}.jpg".format(img_name))
        raise
    # return l.result
    # vehicle detect

    ystart = 400
    ystop = 656
    scale = 1.5

    # track_records = {}
    # labels_boxes = []
    # labels_num = None

    bboxes = []
    bboxes += find_cars(l.result, 0, 400, 400, 500,
                        1.25, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bboxes += find_cars(l.result, 0, 400, 400, 656,
                        1.5, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    bboxes += find_cars(l.result, 800, 1280, 400, 500,
                        1.05, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    bboxes += find_cars(l.result, 800, 1280, 400, 550,
                        1.25, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    bboxes += find_cars(l.result, 800, 1280, 400, 656,
                        1.5, color_space,
                        svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)



    out_img, labels = display_vehicle(l.result, bboxes)

    if isvideo:
        # save unusual for tuning
        if labels[1] > 2:
            # img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])
            img_name = str(time.time())
            im = Image.fromarray(img)
            im.save("unusual_images/unusual_{}.jpg".format(img_name))
        labels_num_list.append(labels[1])

        boxes = labels2boxes(labels)
        print(boxes)
        # print(boxes)
        if os.path.exists('data_analysis.p'):
            data_analysis = pickle.load(open('data_analysis.p', 'rb'))
        else:
            # data for sanity_check
            data_analysis = {}
            data_analysis['boxes_list'] = [] # boxes detected after heatmap fliter not drawed
            data_analysis['frame_detected'] = []
            data_analysis['boxes_draw'] = []
            # boxes_list = data_analysis['boxes_list']
        data_analysis['boxes_list'].append(boxes)
        data_analysis['frame_detected'].append(len(boxes) > 0)
        frame_detected = data_analysis['frame_detected']
        if frame_detected[-3:].count(True) == 3:
            # todo
            boxes = sanity_track_heat(l.result, data_analysis['boxes_list'][-3:])
            out_img = draw_boxes(l.result, boxes)
            data_analysis['boxes_draw'].append(boxes)
        else:
            out_img = l.result
        data_analysis['boxes_draw'].append(boxes)
        print('after sanity check:{}'.format(boxes))
        if len(boxes) > 2:
            # img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])
            img_name = str(time.time())
            im = Image.fromarray(img)
            im.save("unusual_images/unusual_{}.jpg".format(img_name))


        with open('data_analysis.p', 'wb') as f:
            pickle.dump(data_analysis, f)

    # return(out_img, l, boxes)
    return out_img

if __name__ == '__main__':

    if os.path.exists('unusual_images'):
        shutil.rmtree('unusual_images')
    os.mkdir('unusual_images')

    if os.path.exists('line_fit.p'):
    	os.remove('line_fit.p')

    if os.path.exists('data_analysis.p'):
    	os.remove('data_analysis.p')

    if not os.path.exists('unusual_images'):
    	os.mkdir('unusual_images')

    if os.path.exists('track_records.csv'):
    	os.remove('track_records.csv')

    with open('track_records.csv', 'w') as f:
    		f.write('{},{},{},{},{},{},{}\n'.format('left_curverad', 'right_curverad', 'dist_from_center_in_meters', 'lane_line_width', 'img_name', 'leftx_base', 'rightx_base'))


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


    # video preprocessing
    # input_video = 'test_video.mp4'
    input_video = 'project_video.mp4'
    clip = VideoFileClip(input_video)
    # clip = VideoFileClip(input_video).subclip(22,25)
    output_clip = clip.fl_image(pipeline)
    output_clip.write_videofile('output_' + input_video, audio=False)

    # data_analysis = {}
    # data_analysis['labels_num_list'] = labels_num_list
    # data_analysis['boxes_list'] = boxes_list
    # # data_analysis['labels_boxes'] = labels_boxes
    # with open('data_analysis.p', 'wb') as f:
    #     pickle.dump(data_analysis, f)
