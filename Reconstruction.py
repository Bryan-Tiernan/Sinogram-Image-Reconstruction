#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:25:59 2020

@author: bryantiernan 
Name: Bryan Tiernan
ID: 16169093 
"""

##############################################################################
###############################   Imports  ###################################
##############################################################################

from skimage.transform import rotate 
import scipy.fftpack as fft          
import imageio 
import numpy as np 
import imutils
                 
##############################################################################
##############################  Functions  ###################################
##############################################################################

"Using FFT to translate sinogram to freq domain"
def fft_translate(x):
    return fft.rfft(x, axis=1)


"Ramp Filter"
def ramp_ft(a):
    ramp = np.floor(np.arange(0.5, a.shape[1]//2 + 0.1, 0.5))
    return a * ramp


"Using inverse FFT to translate sinogram back to spatial domain"
def inverse_fft_translate(x):
    return fft.irfft(x, axis=1)


"Reconstruct image using back projection"
def back_projection(x):
    laminogram = np.zeros((x.shape[1],x.shape[1]))
    z = 180.0 / x.shape[0]
    for i in range(x.shape[0]):
        tp = np.tile(x[i],(x.shape[1],1))
        tp = rotate(tp, z*i)
        laminogram += tp
    return laminogram


def rescale(image):
    high, low = image.max(), image.min()
    norm = 255 * (image - low) / (high - low)
    img8bit = np.floor(norm).astype('uint8')
    return img8bit


##############################################################################
##############################  Statements  ##################################
##############################################################################

"Import and show sinogram, split into single channels"
sinogram = imutils.imread('sinogram.png',False) #Setting greyscale to False
sinogramr = sinogram[:, :, 0] #Red
sinogramg = sinogram[:, :, 1] #Green
sinogramb = sinogram[:, :, 2] #Blue
print("Sinogram")
imutils.imshow(sinogram)
imutils.imshow(sinogramr)
imageio.imwrite('./images/originalSinogramImage.png', sinogram)



"No Filter reconstruction"
print("Reconstruction with no filtering")
nofilter_r = back_projection(sinogramr)
nofilter_g = back_projection(sinogramg)
nofilter_b = back_projection(sinogramb)
SR = rescale(nofilter_r)
SG = rescale(nofilter_g)
SB = rescale(nofilter_b)
unfiltered = np.dstack((SR,SG,SB))
imutils.imshow(unfiltered)
imageio.imwrite('./images/unfiltered.png', unfiltered)



"FFT to translate channels to freq domain"
print("Freq Domain of sinogram")
freqr = fft_translate(sinogramr)
imutils.imshow(freqr)
imageio.imwrite('./images/freqr.png', 
                  freqr)
freqg = fft_translate(sinogramg)
imutils.imshow(freqg)
imageio.imwrite('./images/freqg.png', 
                  freqg)
freqb = fft_translate(sinogramb)
imutils.imshow(freqb)
imageio.imwrite('./images/freqb.png', 
                  freqb)



"Filter the frequency domain projections by multiplying each one by the frequency domain ramp filter"
print("Frequency domain projections multipled with a ramp filter")
filtered_freqr = ramp_ft(freqr)
imutils.imshow(filtered_freqr)
imageio.imwrite('./images/freqFilterr.png', 
                  filtered_freqr)
filtered_freqg = ramp_ft(freqg)
imutils.imshow(filtered_freqg)
imageio.imwrite('./images/freqFilterg.png', 
                  filtered_freqg)
filtered_freqb = ramp_ft(freqb)
imutils.imshow(filtered_freqb)
imageio.imwrite('./images/freqFilterb.png', 
                  filtered_freqb)



"Use the inverse FFT to return to the spatial domain"
print("Spatial ramp filtered sinogram")
spatial_r = inverse_fft_translate(filtered_freqr)
imutils.imshow(spatial_r)
imageio.imwrite('./images/spatialr.png', 
                  spatial_r)
spatial_g = inverse_fft_translate(filtered_freqg)
imutils.imshow(spatial_g)
imageio.imwrite('./images/spatialg.png', 
                  spatial_g)
spatial_b = inverse_fft_translate(filtered_freqb)
imutils.imshow(spatial_b)
imageio.imwrite('./images/spatialb.png', 
                  spatial_b)



"Back Project all channels"
print("Back Projected channels")
reconstructed_imager = back_projection(spatial_r)
imutils.imshow(reconstructed_imager)
imageio.imwrite('./images/ReconstructedImageR.png', 
                  reconstructed_imager)


reconstructed_imageg = back_projection(spatial_g)
imutils.imshow(reconstructed_imageg)
imageio.imwrite('./images/ReconstructedImageG.png', 
                  reconstructed_imageg)


reconstructed_imageb = back_projection(spatial_b)
imutils.imshow(reconstructed_imageb)
imageio.imwrite('./images/ReconstructedImageB.png', 
                  reconstructed_imageb)



"Rescale and reconstruct image"
R = rescale(reconstructed_imager)
G = rescale(reconstructed_imageg)
B = rescale(reconstructed_imageb)

finalimage = np.dstack((R,G,B))
print("Reconstructed Image")
imutils.imshow(finalimage)
imageio.imwrite('./images/finalimage.png', 
                  reconstructed_imageb)


"Incorrect implementation of Hamming window"
#"Hamming-Windowed Ramp Filter"
#print("Hamming-Windowed reconstructed image")
#window = np.hamming(566)
#
#HR = filtered_freqr * window
#HG = filtered_freqg * window
#HB = filtered_freqb * window
#spatial_r = inverse_fft_translate(HR)
#spatial_g = inverse_fft_translate(HG)
#spatial_g = inverse_fft_translate(HB)
#reconstructed_imager = back_projection(spatial_r)
#reconstructed_imageg = back_projection(spatial_g)
#reconstructed_imageb = back_projection(spatial_b)
#R = rescale(reconstructed_imager)
#G = rescale(reconstructed_imageg)
#B = rescale(reconstructed_imageb)
#
#hamming = np.dstack((R,G,B))
#imutils.imshow(hamming)
#imageio.imwrite('./images/hammingImage.png', 
#                  hamming)