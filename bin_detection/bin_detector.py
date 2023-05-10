'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''

		theta1 = np.array([0.22029858])
		theta2 = np.array([0.31237979])
		theta3 = np.array([0.21245801])
		theta4 = np.array([0.25486362])
		
		mu1 = np.array([112.32440294, 199.31498777, 156.83753417])
		mu2 = np.array([96.33296314, 52.13694194, 156.39632988])
		mu3 = np.array([39.61283229, 121.91786571, 82.84486255])
		mu4 = np.array([22.64031739, 36.43760991, 140.14086696])
		
		sigma1 = np.array([5.31381898, 43.61521714, 50.27691982])
		sigma2 = np.array([33.94199953, 44.02605476, 67.08426259])
		sigma3 = np.array([23.32921684, 52.68961033, 50.55900782])
		sigma4 = np.array([12.95039812, 23.21085857, 67.05881508])
		
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

		y1 = np.zeros((img.shape[0],img.shape[1]))
		y1 += (np.square(img[:,:,0] - mu1[0])/sigma1[0]) + (np.square(img[:,:,1] - mu1[1])/sigma1[1]) + (np.square(img[:,:,2] - mu1[2])/sigma1[2]) + \
			   np.sum(np.log(np.square(sigma1))) + np.log(np.square(1/theta1))
			   
		y2 = np.zeros((img.shape[0],img.shape[1]))
		y2 += (np.square(img[:,:,0] - mu2[0])/sigma2[0]) + (np.square(img[:,:,1] - mu2[1])/sigma2[1]) + (np.square(img[:,:,2] - mu2[2])/sigma2[2]) + \
			   np.sum(np.log(np.square(sigma2)))  + np.log(np.square(1/theta2))

		y3 = np.zeros((img.shape[0],img.shape[1]))
		y3 += (np.square(img[:,:,0] - mu3[0])/sigma3[0]) + (np.square(img[:,:,1] - mu3[1])/sigma3[1]) + (np.square(img[:,:,2] - mu3[2])/sigma3[2]) + \
			   np.sum(np.log(np.square(sigma3))) + np.log(np.square(1/theta3))

		y4 = np.zeros((img.shape[0],img.shape[1]))
		y4 += (np.square(img[:,:,0] - mu4[0])/sigma4[0]) + (np.square(img[:,:,1] - mu4[1])/sigma4[1]) + (np.square(img[:,:,2] - mu4[2])/sigma4[2]) + \
			   np.sum(np.log(np.square(sigma4))) + np.log(np.square(1/theta4))
			   
		# fmin = np.min(y1, y2, y3)

		mask_img = np.zeros((img.shape[0], img.shape[1]))
		y_mask = np.zeros((img.shape[0], img.shape[1]))
		for i in range(0, img.shape[0]):
			for j in range(0, img.shape[1]):
				y_mask[i,j] = min(y1[i,j], y2[i,j], y3[i,j], y4[i,j])

				if y_mask[i,j] == y1[i,j]:
					mask_img[i,j] = 255

				else:
					mask_img[i,j] = 0
		
		# Replace this with your own approach 
		
		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE
		
		# Replace this with your own approach 
		mask_img = img
		mask_labels = label(np.asarray(mask_img))
		props = regionprops(mask_labels)

		img_copy = np.asarray(mask_img)
		boxes =[]
		for prop in props:
			if ((prop.bbox[2]-prop.bbox[0])/(prop.bbox[3]-prop.bbox[1]) > 1) and ((prop.bbox[2]-prop.bbox[0])/(prop.bbox[3]-prop.bbox[1]) < 2) and (prop.bbox_area > 6000):
					if ((np.count_nonzero(mask_img[prop.bbox[0]:prop.bbox[2],prop.bbox[1]:prop.bbox[3]]))/((prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1])) > 0.45):
						cv2.rectangle(img_copy, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (128, 0, 0), 3)
						boxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
				
		fig, (ax2, ax3) = plt.subplots(1, 2, figsize = (10, 5))
		ax3.set_title('Image with derived bounding box')
		ax2.imshow(mask_img)#, cmap='gray')
		ax3.imshow(img_copy)
		# plt.show()

		# cv2.imshow('image', mask_img)
		# cv2.waitKey(1500)

		
		# YOUR CODE BEFORE THIS LINE
		################################################################
		
		return boxes


