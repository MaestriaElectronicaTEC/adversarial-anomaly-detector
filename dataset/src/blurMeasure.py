#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:18:31 2018

@author: luisalonsomurillorojas
"""

import numpy as np
import sys, getopt
from PIL import Image
from scipy import ndimage
from scipy.signal import argrelextrema
from skimage import filters

def xrange(x):
    return iter(range(x))

def non_maximum_supression(det):
  gmax = np.zeros(det.shape)
  for i in xrange(gmax.shape[0]):
    for j in xrange(gmax.shape[1]):

      if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
        # 0 degrees
          if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
            gmax[i][j] = det[i][j]
  return gmax

def findLocalExtremaAux(row):
    # for local maxima
    localMaxima = argrelextrema(row, np.greater_equal)

    # for local minima
    localMinima = argrelextrema(row, np.less_equal)

    # concatenate maxima and minima arrays
    localExtrema = np.concatenate((localMinima[0], localMaxima[0]), axis=None)
    return np.sort(localExtrema, axis=None)
    

def findLocalExtrema(index, row):
    # Init values
    minLocal = -1
    maxLocal = -1
    extremas = findLocalExtremaAux(row)
    
    for n in xrange(extremas.shape[0]):
        y = extremas[n]
        if (index > y):
            
            # set the local minima
            minLocal = y
            
            # set the local maxima
            maxLocal = extremas[n + 1]
            
    return (minLocal, maxLocal)

def getBlur(img):

    # Filter the image
    im = ndimage.gaussian_filter(img, 0.5)

    # Apply Sobel filter
    dx = ndimage.sobel(im.astype(float), 1)  # horizontal derivative (vertical edges)

    # Apply non-maximun supression
    nms = non_maximum_supression(abs(dx))

    # Apply hysteresis tresholding
    low = 0.1
    high = 0.35

    low_threshold = np.amax(nms) * low
    high_threshold = np.amax(nms) * high

    hyst = filters.apply_hysteresis_threshold(nms, low_threshold, high_threshold)

    # Init the blur measure variables
    NbEdges = 0
    TotBM = 0

    for i in xrange(hyst.shape[0]):
        for j in xrange(hyst.shape[1]):
        
            if (hyst[i][j]):
            
                # Find the local extremas around the pixel.
                extrema = findLocalExtrema(j, im[i])
            
                if (extrema[0] != -1 and extrema[1] != -1):
                    # Update the variables
                    NbEdges = NbEdges + 1
                    edgeWidth = extrema[1] - extrema[0]
                    TotBM = TotBM + edgeWidth

    blurMeasure = TotBM / NbEdges
    
    return blurMeasure
