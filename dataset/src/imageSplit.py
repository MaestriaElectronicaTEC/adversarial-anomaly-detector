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

def processImages(imagesPath, subsectionSize, distPath, overlap):
    imgList = glob.glob(imagesPath + '*.png')
    pathBase = Path(distPath)

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

def main():
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Script that splits an image into subsections')
    parser.add_argument('--image', help='Image path', default='')
    parser.add_argument('--images', help='Directory with a group of images', default='')
    parser.add_argument('--overlap', help='Activate the overlapping of the generated images', default=False, action='store_true')
    parser.add_argument('--distDir', help='Directory where the subsections will be stored', default = '')
    parser.add_argument('--subsectionSize', help='Size of each subsection H x W (Default (50, 50))', nargs='+', type=int)
    args = parser.parse_args()

    # Init the variables
    imagePath = str(args.image)
    imagesPath = str(args.images)
    distPath = str(args.distDir)
    subsectionSize = (50, 50)
    overlap = args.overlap
    if args.subsectionSize != None:
        subsectionSize = tuple(args.subsectionSize)

    # Verify if the initial conditions are met
    if (imagePath == '' or imagesPath == '') and distPath == '':
        raise(RuntimeError("The minimun amount of arguments were not provided."))

    # Process the variables
    if imagePath != '':
        if overlap:
            splitImage(imagePath, subsectionSize, str(distPath))
        else:
            splitImageWithoutOverlap(imagePath, subsectionSize, str(distPath))
    if imagesPath != '':
        processImages(imagesPath, subsectionSize, distPath, overlap)

if __name__ == "__main__":
    main()
