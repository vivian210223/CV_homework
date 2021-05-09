import  cv2
import numpy as np
import glob
import pdb
import matplotlib.pyplot as plt
import random
import math

def readImage(images):
	left, right = [],[]
	for idx, name in enumerate(images):
		img = cv2.imread(name)
		if idx%2==0:	
			left.append(img)
		else:
			right.append(img)	
	return left, right

def faeture_match(left, right, left_kp, right_kp, threshold):
	kpsize_1, dim1 = left.shape 
	kpsize_2, dim2 = right.shape
	graph1 = []
	graph2 = []
	for i in range(kpsize_1):
		dist = {}
		for j in range(kpsize_2):
			dist[j] = np.sqrt(np.sum(np.square(left[i]-right[j])))
		sort = dict(sorted(dist.items(), key=lambda item: item[1]))
		first,second = list(sort.items())[:2]
		# ratio distance
		if first[1]/second[1]<=threshold:
			graph1.append(left_kp[i].pt)
			graph2.append(right_kp[first[0]].pt)
	return graph1, graph2

def plot_matches(left, right, kp_left, kp_right, name):
	(hA ,wA),(hB, wB) = left.shape[:2], right.shape[:2]
	canvas = np.zeros((max(hA, hB),wA+wB, 3), dtype = 'uint8')
	canvas[0:hA, 0:wA] = left
	canvas[0:hB, wA:]= right

	for a, b in zip(kp_left, kp_right):
		color = np.random.randint(0, high=255, size=(3,))
		color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
		ptA = (int(a[0]), int(a[1]))
		ptB = (int(b[0] + wA), int(b[1]))
		cv2.line(canvas, ptA, ptB, tuple(color), 1)
	cv2.imshow('My Image', canvas)
	cv2.imwrite(name+".jpg", canvas)
	cv2.waitKey(0)

def homography(left, right):
	row = []
	for p1, p2 in zip(left, right):
		row1 = [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1], -p2[0]*1]
		row2 = [ 0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1], -p2[1]*1]
		row.append(row1)
		row.append(row2)
	row = np.array(row)
	_, _, V = np.linalg.svd(row)
	H = V[-1].reshape(3, 3)
	if H[2,2]<0:
		H = H*(-1)
	# standardize	
	H = H/H[2, 2] 
	return H

def Ransac(kp_left, kp_right, N, threshold):
	num_of_kp = len(kp_left)
	inliers = 0
	H = np.zeros((3, 3))
	for k in range(N):
		# randmom sample 4 pairs
		rand = random.sample(range(num_of_kp), 4)
		candidate_H = homography(kp_left, kp_right)
		count = 0
		# check the number of inliers/outliers
		for i in range(num_of_kp):
			l = np.array([kp_left[i][0],kp_left[i][1],1])
			pixel_coord = np.dot(candidate_H,l)
			if np.linalg.norm(kp_right[i]-(pixel_coord/pixel_coord[2])[0:2]) <= threshold:
				count +=1
				

		if count >= inliers:
			inliers = count
			H = candidate_H

	print("inliers/total: {}/{}".format(inliers,num_of_kp))
	print(H)
	return H

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des

def decide_out_size(img1, img2, homography):
	# img1.shape[0]:=height, img1.shape[]:=width 
    four_corner = np.zeros((4, 3))
    four_corner[0, :] = [0, 0, 1]
    four_corner[1, :] = [img1.shape[1], 0, 1]
    four_corner[2, :] = [0, img1.shape[0], 1]
    four_corner[3, :] = [img1.shape[1], img1.shape[0], 1]
    min_x = 0
    min_y = 0
    max_y, max_x, _ = img2.shape
    for corner in four_corner:
        trans_corner = homography@corner.T
        trans_corner /= trans_corner[2] # standardize
        x, y, _ = trans_corner
        min_x = min(min_x, math.floor(x))
        min_y = min(min_y, math.floor(y))
        max_x = max(max_x, math.ceil(x))
        max_y = max(max_y, math.ceil(y))
    return min_x, min_y, max_x, max_y


def bilinear_interpolation(x, y, img1, img2):
    h, w, _ = img1.shape
    x1 = math.floor(x)
    if x1 < 0:
        x1 = 0
    y1 = math.floor(y)
    if y1 < 0:
        y1 = 0
    x2 = math.ceil(x)
    if x2 >= w:
        x2 = w-1
    y2 = math.ceil(y)
    if y2 >= h:
        y2 = h-1

    q11 = img1[y1, x1, :]
    q21 = img1[y1, x2, :]
    q12 = img1[y2, x1, :]
    q22 = img1[y1, x1, :]

    if x1 == x2 and y1 == y2:
        rgb = q11
    elif x1 == x2 and y1 != y2:
        rgb = (q11 * (y2 - y) + q12 * (y - y1))/(y2-y1+0.0)
    elif y1 == y1 and x1 != x2:
        rgb = (q21 * (x - x1) + q22 * (x2-x))/(x2-x1+0.0)
    else:
        rgb = (q11 * (x2 - x) * (y2 - y) +
               q21 * (x - x1) * (y2 - y) +
               q12 * (x2 - x) * (y - y1) +
               q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1)+0.0)
    return rgb

def img_morphing(offset_x, img1, x):
    left_border = offset_x
    right_border = img1.shape[1]
    ratio1 = (right_border-x)/(right_border-left_border+0.0)
    ratio2 = (x-left_border)/(right_border-left_border+0.0)
    return ratio1, ratio2

def img_warping(img1, img2, homography, img_name):
    min_x, min_y, max_x, max_y = decide_out_size(img1, img2, homography)
    if min_x < 0:
        offset_x = -min_x
    else:
        offset_x = 0
    if min_y < 0:
        offset_y = -min_y
    else:
        offset_y = 0
    w = max_x+offset_x
    h = max_y+offset_y
    out_image = np.full((h, w, 3), 0)
    # move image 2 to output image
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            y = offset_y+i
            x = offset_x+j
            out_image[y, x, :] = img2[i, j, :]
            
    tmp_image = np.float32(out_image/255.0)
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)
    
    # move image 1 to output image
    h_inv = np.linalg.inv(homography)
    for i in range(h):
        for j in range(w):
            p2 = np.array([j-offset_x, i-offset_y, 1])
            p1 = h_inv@p2.T
            p1 /= p1[-1]
            x, y, _ = p1

            if x < 0 or x >= img1.shape[1] or y < 0 or y >= img1.shape[0]:
                continue
            elif j < offset_x or i < offset_y or i >= offset_y+img2.shape[0]:
                out_image[i, j, :] = bilinear_interpolation(x, y, img1, img1)
            else:
                img1_rgb = bilinear_interpolation(x, y, img1, img1)
                out_image[i, j, :] = img1_rgb*0.99+out_image[i, j, :]*0.01
    out_image = np.float32(out_image/255.0)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(20, 10))
    plt.imshow(out_image)
    plt.savefig(img_name+"_panorama.jpg")
    plt.show()
    return out_image

def crop_image(image, ratio):
    height, width, channel = image.shape
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    cropped_image = image[crop_height: height -
                          crop_height, crop_width: width - crop_width]
    return cropped_image

if __name__ == '__main__':
	# Make a list of images
    images = glob.glob('data/*')
    left, right = readImage(images)
    thres = [0.45,0.4,0.25]
    thres2 = [0.5,1,0.5]
    names = ["TV","Hill","Roof"]
    for i in range(len(left)):
    	# hills.jpg with white egdes
    	if i ==1: 
    		left[i] = crop_image(left[i], 0.03)
    		right[i] = crop_image(right[i], 0.03)
    	# find keypoints
    	left_kp_img, left_kp, left_des = sift_kp(left[i])
    	right_kp_img, right_kp, right_des = sift_kp(right[i])
    	# keypoints matching
    	kp_left, kp_right = faeture_match(left_des,right_des,left_kp,right_kp,thres[i])
    	# plot match in graph
    	plot_matches(left[i],right[i],kp_left, kp_right,names[i])
    	# Ransac
    	H = Ransac(kp_right, kp_left, 1000, thres2[i])
    	# warping
    	name = "./{}".format(names[i])
    	out_image = img_warping(right[i],left[i], H, name)

    	