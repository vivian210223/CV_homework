from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import os


DATA_PATH="./hw2_data/task1,2_hybrid_pyramid/"


def fourier_transform(feature):
	F=np.fft.fft2(feature)
	F=np.fft.fftshift(F)
	return F


def gauss_low_pass(height,width,D0):
	H=np.zeros((height,width))
	center_point=(height/2,width/2)
	for v in range(height):
		for u in range(width):
			D=np.sqrt((v-center_point[0])**2+(u-center_point[1])**2)
			H[v][u]=np.exp(-D**2/(2*D0**2))
	return H


def inv_fourier_transform(F):
	f=np.fft.ifftshift(F)
	f=np.fft.ifft2(f)
	return f


if __name__=='__main__':
	if not os.path.exists("./results"):
		os.mkdir("./results")
	num_list=["0","1","2","3","4","5","6"]
	for num in num_list:
		image_pair=[]
		plt.figure(figsize=(10,4))
		for image_path in glob.glob(DATA_PATH+num+"*"):
			image_pair.append(image_path)

		image1=Image.open(image_pair[0])
		image1=np.array(image1)
		height1,width1,channel=image1.shape
		
		output_image_1=np.zeros((height1,width1,channel))

		for c in range(channel):
			feature=image1[:,:,c]

			# fourier transform
			F=fourier_transform(feature)

			# H(u,v) -> low pass filter
			H=gauss_low_pass(height1,width1,30)

			# F(u,v)*H(u,v)
			low_pass_image=F*H

			# Compute the inverse Fourier transformation
			f=inv_fourier_transform(low_pass_image)

			# Obtain the real part 
			real_image=np.abs(f)

			output_image_1[:,:,c]=real_image

		output_image_1=Image.fromarray(np.uint8(output_image_1)).convert("RGB")
		plt.subplot(1,3,1)
		plt.imshow(output_image_1)
		
		image2=Image.open(image_pair[1])
		image2=np.array(image2)
		height2,width2,channel=image2.shape

		output_image_2=np.zeros((height2,width2,channel))

		for c in range(channel):
			feature=image2[:,:,c]

			# fourier transform
			F=fourier_transform(feature)

			# H(u,v) -> high pass filter
			H=gauss_low_pass(height2,width2,10)
			H=1-H

			high_pass_image=F*H

			# compute the inverse fourier transform
			f=inv_fourier_transform(high_pass_image)

			# obtain the real part
			real_image=np.abs(f)

			output_image_2[:,:,c]=real_image

		output_image_2=Image.fromarray(np.uint8(output_image_2)).convert("RGB")
		plt.subplot(1,3,2)
		plt.imshow(output_image_2)

		# merge two image
		height=height1 if height1>=height2 else height2
		width=width1 if width1>=width2 else width2

		output_image_1=output_image_1.resize((width,height))
		output_image_2=output_image_2.resize((width,height))
		blend_image=Image.blend(output_image_1,output_image_2,0.5)
		plt.subplot(1,3,3)
		plt.imshow(blend_image)

		plt.savefig("./results/"+num+".jpg")

		print(f'{num} completed')





