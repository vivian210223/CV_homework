import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import scipy.ndimage
import pdb

# read img from a path
def readImage(path):
    images = glob.glob(path+'/*')
    img=[]
    for idx, img in enumerate(images):
        pdb.set_trace()
        img[int(idx)] = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    return img

# show 1 img
def showImage(image):
    # display
    cv2.imshow('My Image', image)
    # pause any key to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# turn RGB to greyscale
# https://www.tutorialspoint.com/dip/grayscale_to_rgb_conversion.htm
def rgb2gray(image):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def gaussian_filter(image):
    # Gaussian kernel(filter)
    gaussian_kernel = np.array([[  1,  4,  6,  4,  1],
                                [  4, 16, 24, 16,  4],
                                [  6, 24, 36, 24,  6],
                                [  4, 16, 24, 16,  4],
                                [  1,  4,  6,  4,  1]])/256

    height = image.shape[0]
    width = image.shape[1]

    # define padding
    pad_size = gaussian_kernel.shape[0]//2
    padding_img = np.pad(image,((pad_size, pad_size),(pad_size, pad_size)),'constant',constant_values = 0)
    blurring = np.zeros([height, width], dtype=int)

    # apply Gaussian filter
    for i in range(height):
        for j in range(width):
            blurring[i, j] = int(np.sum(padding_img[i: i + (2 * pad_size) + 1, j: j + (2 * pad_size) + 1] * gaussian_kernel))

    return blurring

def downsample(image):
    height = image.shape[0]
    width = image.shape[1]
    result = np.zeros(((height + 1) // 2, (width + 1) // 2), dtype=int)

    # each 2x2 pixel downsample to 1 pixel
    for i in range((height + 1) // 2):
        for j in range((width + 1) // 2):
            result[i, j] = int(np.sum(image[2 * i:2 * i + 2, 2 * j:2 * j + 2]) / 4)

    # convert to [0,255]
    result = result.astype(np.uint8)
    return result

def upsample(image):
    # Resampled by a factor of 2 with nearest interpolation
    return scipy.ndimage.zoom(image, 2, order=0)

    #return np.insert(np.insert(image, np.arange(1, image.shape[0] + 1), 0, axis=0),
    #                 np.arange(1,image.shape[1] + 1), 0,axis=1)

def Laplacian(old,new):
    new = gaussian_filter(upsample(new))#.astype(np.uint8)
    return np.subtract(old, new[0:old.shape[0], 0:old.shape[1]])

def magnitude_spectrum(image):
    # Fourier transform
    f = np.fft.fft2(image)
    # shift the zero frequency to the center
    fshift = np.fft.fftshift(f)
    # take log to compress the value useful for visualization
    spectrum = 20 * np.log(np.abs(fshift))
    return spectrum

def Plot(gaussian, g_spectrum, laplacian, l_spectrum, level, path):
    for i in range(level):
        plt.subplot(5, 4, 4 * i + 1)
        if i == 0:
            plt.title('Gaussian')
        plt.axis('off')
        plt.imshow(gaussian[i+1], cmap='gray')

        plt.subplot(5, 4, 4 * i + 2)
        if i == 0:
            plt.title('Spectrum')
        plt.axis('off')
        plt.imshow(g_spectrum[i], cmap='gray')

        plt.subplot(5, 4, 4 * i + 3)
        if i == 0:
            plt.title('Laplacian')
        plt.axis('off')
        plt.imshow(laplacian[i], cmap='gray')

        plt.subplot(5, 4, 4 * i + 4)
        if i == 0:
            plt.title('Spectrum')
        plt.axis('off'), plt.imshow(l_spectrum[i], cmap='gray')

    plt.tight_layout()
    plt.savefig(path+'.png')
    plt.show()

if __name__ == '__main__':
    # Make a list of calibration images
    # images = glob.glob('hw2_data/task1,2_hybrid_pyramid/*')
    images = glob.glob('my_data/pikachu.jpg')
    savepath = 'result/'
    level = 5
    for idx, name in enumerate(images):
        gaussian_img, gaussian_spectrum = [], []
        laplacian_img, laplacian_spectrum = [], []
        # original image
        gaussian_img.append(cv2.imread(name, cv2.IMREAD_GRAYSCALE))
        # Image Pyramid
        for i in range(level):
            gaussian_img.append(downsample(gaussian_filter(gaussian_img[i])))
            gaussian_spectrum.append(magnitude_spectrum(gaussian_img[i+1]))
            laplacian_img.append(Laplacian(gaussian_img[i], gaussian_img[i+1]))
            laplacian_spectrum.append((magnitude_spectrum(laplacian_img[i])))
        # plot all image together
        Plot(gaussian_img, gaussian_spectrum, laplacian_img, laplacian_spectrum, level, savepath+str(idx))
