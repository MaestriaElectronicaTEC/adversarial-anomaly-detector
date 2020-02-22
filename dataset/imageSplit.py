#!/usr/bin/env python

from pathlib import Path
import numpy as np
import argparse
import glob
import cv2
import os

def splitImage(imgPath, subsectionSize, distPath):
    # Load the image
    img = cv2.imread(imgPath)

    # Get image dimensions
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    # Get the subsetion dimensions
    height =  subsectionSize[0]
    width = subsectionSize[1]

    # Calculate the number of subsections
    halfHeight = int(height / 2)
    halfWidth = int(width / 2)
    numRows = int(imgHeight / height) + int((imgHeight - halfHeight) / height)
    numCols = int(imgWidth / width) + int((imgWidth - halfWidth) / width)
    totSubImg = numRows * numCols

    # Get the base name for each of the subsection images
    baseName = imgPath.split('/')[-1]
    baseName = baseName[0:-4]
    pathBase = Path(distPath)
    print('Total of sub-images for {}: {}'.format(baseName, totSubImg))

    # Create the subimages
    for row in range(numRows):
        for col in range(numCols):
            subimg = img[row*halfHeight : row*halfHeight + height, col*halfWidth : col*halfWidth + width]
            imgName = baseName + '_' + str(row*halfHeight) + 'x' + str(col*halfWidth) + '.png'
            imgPath = pathBase / imgName
            cv2.imwrite(str(imgPath), subimg)
            
def splitImageWithoutOverlap(imgPath, subsectionSize, distPath):
    # Load the image
    img = cv2.imread(imgPath)

    # Get image dimensions
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]

    # Get the subsetion dimensions
    height =  subsectionSize[0]
    width = subsectionSize[1]

    # Calculate the number of subsections
    numRows = int(imgHeight / height)
    numCols = int(imgWidth / width)
    totSubImg = numRows * numCols

    # Get the base name for each of the subsection images
    baseName = imgPath.split('/')[-1]
    baseName = baseName[0:-4]
    pathBase = Path(distPath)
    print('Total of sub-images for {}: {}'.format(baseName, totSubImg))

    # Create the subimages
    for row in range(numRows):
        for col in range(numCols):
            subimg = img[row*height : row*height + height, col*width : col*width + width]
            imgName = baseName + '_' + str(row*height) + 'x' + str(col*width) + '.png'
            imgPath = pathBase / imgName
            cv2.imwrite(str(imgPath), subimg)

def process_image(imagePath, subsectionSize, distPath, overlap):
    # verify the subsection size variable
    if subsectionSize != None:
        subsectionSize = tuple(subsectionSize)
    else:
        subsectionSize = (50, 50)

    # apply or not the overlap durint the image generation
    if overlap:
        splitImage(imagePath, subsectionSize, str(distPath))
    else:
        splitImageWithoutOverlap(imagePath, subsectionSize, str(distPath))

def process_images(imagesPath, subsectionSize, distPath, overlap):
    imgList = glob.glob(imagesPath + '*.png')
    pathBase = Path(distPath)

    # verify the subsection size variable
    if subsectionSize != None:
        subsectionSize = tuple(subsectionSize)
    else:
        subsectionSize = (50, 50)

    # Process all the images in the specified path
    for img in imgList:
        # Create the where the subsections of img will be stored
        baseName = img.split('/')[-1]
        baseName = baseName[0:-4]
        imgDistPath = pathBase / baseName
        os.mkdir(str(imgDistPath))

        if overlap:
            splitImage(img, subsectionSize, str(imgDistPath))
        else:
            splitImageWithoutOverlap(img, subsectionSize, str(imgDistPath))
