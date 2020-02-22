#!/usr/bin/env python

from datetime import datetime
import numpy as np
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import glob

from dataset.blurMeasure import getBlur

def plotBlurStat (blur, x_label, y_label, plot_label, show, plotDir):
    # Graph information
    plt.plot(blur, label=plot_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    
    if show:
        plt.show()
    else:
        plt.savefig(plotDir + '.png')
        plt.clf()

def getFrames(video, interval, imgDir, treshold, showGraph):
    cap = cv2.VideoCapture(video)
    start = time.time()
    end = time.time()
    blurAcc = []
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            break
        
        intervalTime = end - start
        if intervalTime >= interval:
            dt_object = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = getBlur(gray)
            blurAcc.append(blur)
            start = time.time()
            
            print('Blur: ' + str(blur))
            
            if treshold >= blur:
                cv2.imwrite(imgDir + dt_object + '.png', frame)
            else:
                print('Blur over the treshold of ' + str(treshold))

        # Display the resulting frame
        cv2.imshow('frame',frame)
        end = time.time()
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot the blur stats
    plotBlurStat(blurAcc, 'Images', 'Blur', 'Blure measure', showGraph, video[0:-4])
        
    print('Average blur: ' + str(sum(blurAcc)/len(blurAcc)))

def process_video(videoPath, interval, imgDir, blurTreshold, videoDir, videoExtension):
    if videoDir != '':
        print('Processing the videos of the directory: ' + videoDir)
        plots = glob.glob(videoDir + '*.png')
        allVideos = glob.glob(videoDir + '*' + videoExtension)
        for videoFile in allVideos:
            tmpVideoFile = videoFile[:-4] + '.png'
            if tmpVideoFile not in plots:
                print('Processing the video ' + videoFile)
                getFrames(videoFile, interval, imgDir, blurTreshold, False)
    else:
        getFrames(videoPath, interval, imgDir, blurTreshold, True)

