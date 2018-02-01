import numpy as np
import cv2
import glob
# import cPickle as pickle
import pickle
import os
import time
import math

# camera class
from image_process import *
class camera():
	# only init is used
	def __init__(self):
		self.img_path = glob.glob('camera_cal/calibration*.jpg') # given a set of chessborad images
		self.img_pattern=(9, 6)
		self.img_size = None
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

		self.objp = objp
		self.objpoints = [] # 3D
		self.imgpoints = [] # 2D
		self.mtx = None # calibration matrix
		self.dist = None # distortion coefficients
		self.check = False
		# necessary to output?
		if os.path.exists("calibration.p"):
		    calibration_data = pickle.load(open( "calibration.p", "rb" ))
		    self.mtx, self.dist = calibration_data['mtx'], calibration_data['dist']
		    self.check = True
		else:
		    self.update_mtx_and_dist()

	def update_points(self):
		images = [cv2.imread(image) for image in self.img_path]
		for img in images:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# update img_size
			self.img_size = gray.shape[::-1]
			ret, corners = cv2.findChessboardCorners(gray, self.img_pattern, None)
			if ret:
				self.objpoints.append(self.objp)
				self.imgpoints.append(corners)

		return self.objpoints, self.imgpoints


	def update_mtx_and_dist(self):
		if self.check:
			return None
		objpoints, imgpoints = self.update_points()
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size, None, None)
		self.mtx = mtx
		self.dist = dist
		calibration_data = {'mtx':mtx,'dist':dist}
		pickle.dump(calibration_data, open( "calibration.p", "wb" ) )
		self.check = True
		return mtx, dist

class img_camera(camera):
	# update_M_and_Minv and combined_thresh are main
	def __init__(self, img):
		camera.__init__(self)
		self.img = img
		self.img_name = '-'.join([str(x) for x in time.localtime(time.time())[:5]])

		# only relate to img.shape so consistent for a video
		self.src = None
		self.dst = None
		self.M = None
		self.Minv = None

		self.undist = None
		self.combined_threshold_img = None
		self.binary_top_down_image = None
	# need to modify
	def update_src_and_dst(self):
		# only relate to img.shape,is this ok?
		# or make it hard
		# will be tuned
		# here we can use hough line detection
		# todo
		# offset = [50, 0]
		# corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
		# src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
		# dst_points = np.float32([corners[0] + offset, corners[1] + offset, corners[2] - offset, corners[3] - offset])
		h, w = self.img.shape[0], self.img.shape[1]

		src_points = np.float32([
								[int(w/2.15), int(h/1.6)],[w - int(w/2.15), int(h/1.6)],
								[w - w//7, h], [w//7, h]
								])
		dst_points = np.float32([
								[int(w/4), 0], [w - int(w/4), 0],
								[w - int(w/4), h], [int(w/4), h]
								])

		self.src = src_points
		self.dst = dst_points

		return src_points, dst_points

	def update_M_and_Minv(self):
		src_points, dst_points = self.update_src_and_dst()
		self.M = cv2.getPerspectiveTransform(src_points, dst_points)
		self.Minv = cv2.getPerspectiveTransform(dst_points, src_points)
		return self.M, self.Minv

	def transform_perspective(self):
		img_size = self.img.shape[1], self.img.shape[0]
		if self.combined_threshold_img is None:
			self.combined_thresh()
		warped = cv2.warpPerspective(self.combined_threshold_img, self.M, img_size, flags=cv2.INTER_LINEAR)
		self.binary_top_down_image = warped
		return warped

	def undistort(self):
		undist = cv2.undistort(self.img, self.mtx, self.dist, None, self.mtx)
		self.undist = undist
		return undist


	def combined_thresh(self):
		self.undistort()
		img = self.undist
		# land = lambda *x: np.logical_and.reduce(x)
		# lor = lambda * x: np.logical_or.reduce(x)
		# Choose a Sobel kernel size
		ksize = 3
		# Apply each of the thresholding functions
		gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(10, 100))
		grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10, 100))
		mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(30, 100))
		dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(1.0, 1.3))
		color_binary = hls_select(img, thresh=(100, 255))
		equalize_color_binary = equalize_histogram_color_select(img, thresh=(250, 255))
		luv_color = luv_select(img, thresh=(225, 255))
		lab_color = lab_select(img, thresh=(150, 200))

		combined = np.zeros_like(img[:,:,0])
		# combined[((gradx == 1) & (grady == 1) & (dir_binary == 1)) | (luv_color==1)] = 1
		# combined[(luv_color==1)] = 1
		combined[((gradx == 1) & (grady == 1)) & ((mag_binary == 1)) | ((color_binary == 1 ) & (equalize_color_binary == 1) & (lab_color==1))| (luv_color==1)] = 1
		# combined = region_of_interest(combined)
		self.combined_threshold_img = combined
		return combined
