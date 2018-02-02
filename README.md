
# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Data

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  


[//]: # (Image References)
[image1]: ./output_images/HOG.jpg
[image2]: ./output_images/test1.jpg
[image3]: ./output_images/test2.jpg
[image4]: ./output_images/test3.jpg
[image5]: ./output_images/test4.jpg
[image6]: ./output_images/test5.jpg
[image7]: ./output_images/test6.jpg
[video1]: ./project_video.mp4

---

## main file

- `pipeline.py`
    - to make output video
- `train_model.py`
    - to train svm model for car
- `utils.py`
    - functions for training and vehicle detection
- `camera.py`
    - contain two class: camera and img_camera
- `image_process.py`
    - contain some function for image processing
- `line.py`
    - contain one class: Line

## how to get output video

steps to use:
- `python train_model.py`
- `python pipeline.py`

## Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

- `get_hog_features`: the code for extracted HOG features from images
- steps to extracted HOG features
    - convert image from RGB to HSV
    - Compute individual channel HOG features for the entire image
    - combine 3 channel HOG features
- parameters used:
```
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
```

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters.
- In order to select appropriate color space, I tried HSV, LUV, HLS, YUV, YCrCb then selected HSV by effect of detection on test images. The code is in `P5_tuning_param.ipynb`.
- after trial, changes of cell_per_block and orient will not effect result of detection apparently.
- select pix_per_cell 8 because the train image has (64,64) shape

here is an example:

![alt text][image1]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

the code to train the classifier is `train_svc` in `tain_model.py`.

steps to train the classifier:
- data preparing
    - extract features using `bin_spatial`, `color_hist`, `get_hog_features` then combine them
    - use StrandardScaler to normalize the features
- random split the data into train and validation data.
- using SVC to train with default parameters

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

the code to implement a sliding window search is `find_cars`. The parameters is as folowing:
- pix_per_cell and cell_per_block is same as the parameters for HOG features
- cells_per_step = 2, increasing of this parameter will lead to many cars missed

the code to apply multiple-scale window search is `pipeline`:
- use more small scale window to detect more remote car

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of classifer, I used SVC modoel instead of linearSVC model.
Ultimately I searched on multiple scales using HSV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
I created a heatmap and then thresholded that map to identify vehicle positions.
I then assumed each blob corresponded to a vehicle.
I constructed bounding boxes to cover the area of each blob detected.
The code is in `sanity_check_heat`.

In fact I apply heatmap twice:
- filter for false positives for only one frame
- filter for false positives for continuous many frames



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?



- The main problem I faced in my implementation of this project is false positives of model predition.
- So I have to adjust threshold for heatmap and design sanity check to filter false positve.
- my pipeline seem too slow and maybe decrease time-consuming by decrease search windows applying lane-line detected.
