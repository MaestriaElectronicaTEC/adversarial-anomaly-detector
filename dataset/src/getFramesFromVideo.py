#!/usr/bin/env python

from datetime import datetime
import numpy as np
import argparse
import time
import cv2
import blurMeasure
import matplotlib.pyplot as plt
import glob

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
            blur = blurMeasure.getBlur(gray)
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
    
def main():
    # Setup the command line arguments
    parser = argparse.ArgumentParser(description='Script that get frames out of a video')
    parser.add_argument('--video', help='Video path', default='')
    parser.add_argument('--imgDir', help='Directory where the frame will be stored', default = './')
    parser.add_argument('--interval', help='Interval time between frames', default=5,type=float)
    parser.add_argument('--treshold', help='Treshold value for the blur in the frame', default=6, type=float)
    parser.add_argument('--videoDir', help='Directory with a series of videos', default='')
    parser.add_argument('--videoExtension', help='Extension of the videos to be processed', default='.MOV')
    args = parser.parse_args()
    
    videoPath = str(args.video)
    interval = float(args.interval)
    imgDir = str(args.imgDir)
    blurTreshold = float(args.treshold)
    videoDir = str(args.videoDir)
    videoExtension = str(args.videoExtension)
    
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
    
if __name__ == "__main__":
    main()
