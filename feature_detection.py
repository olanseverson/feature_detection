# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:31 2019
@author: olanseverson

https://realpython.com/python-logging/
https://pysource.com/2018/03/23/feature-matching-brute-force-opencv-3-4-with-python-3-tutorial-26/
https://www.life2coding.com/resize-opencv-window-according-screen-resolution/
"""

import cv2
import numpy as np
import time 
import logging
#%% Initiate logger for DEBUGGING
logging.basicConfig(level = logging.WARNING,format='[%(levelname)s] => (%(name)s||%(threadName)-s):  %(message)s')
# logging.root.setLevel(logging.WARNING)
FORMATTER = logging.Formatter("[%(levelname)s] => (%(name)s||%(threadName)-s):  %(message)s")
c_handler = logging.StreamHandler() # handler
c_handler.setFormatter(FORMATTER)
logger = logging.getLogger(__name__)
# logger.addHandler(c_handler) # ADD HANDLER TO LOGGER
logger.setLevel(logging.DEBUG) # change DEBUG to another value to remove the debug logger


MAX_CTRL_FRAME = 14 # how many control frame used
SKIPPED_NUM = 3 #skip reading the frame every Skipped_num frame
NUM_MATCH_TH = 150 # two image is similar, if larger than this threshold
MATCH_DISTANCE_TH = 60 # ignore distance if larger than this threshold

## GET THE IMAGE 
list_img = []
for i in range (1,MAX_CTRL_FRAME + 1):
    temp = {}
    temp['img'] = cv2.imread("./control_frame/" + str(i) + ".jpg",cv2.IMREAD_GRAYSCALE) # get control image
    temp['isFound'] = False
    temp['foundIdx'] = 0
    temp['matchVal'] = 0
    temp['foundImg'] = None
    list_img.append(temp)
    
cv2.waitKey(0)
#ctrl_image = cv2.imread("1.jpg",cv2.IMREAD_GRAYSCALE) # get control image
matches_list = []
max_match = 0
#found_frame = ctrl_image

## GET THE VIDEO
cap = cv2.VideoCapture('ori.mp4') # read Video that we want to check
logger.info('frame count: %d', cap.get(cv2.CAP_PROP_FRAME_COUNT))
logger.info('fps : %d', cap.get(cv2.CAP_PROP_FPS))

idx = 0
while True:
#    global SKIPPED_NUM
#    global NUM_MATCH_TH 
#    global MATCH_DISTANCE_TH 
#    global idx
    
    _, frame = cap.read()
    if frame is None:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale
    ## SHOW FRAME 
    cv2.namedWindow('app', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('app', 400,600)
    cv2.imshow('app', frame)
    
    ## ORB Detector
    orb = cv2.ORB_create()
    ctrl_image = list_img[idx]['img']
    kp1, des1 = orb.detectAndCompute(ctrl_image, None)
    kp2, des2 = orb.detectAndCompute(frame, None)    
    
    ## Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
#    matches = sorted(matches, key = lambda x:x.distance)
    matches = list(filter(lambda x: x.distance<MATCH_DISTANCE_TH, 
                          matches)) # ignore value if smaller than match_distance
                              
    logger.debug(len(matches))
#    for m in matches:
#        print(m.distance)
        
#    matches_list.append(len(matches))
    
    ## Find most similar picture
    if(len(matches)>NUM_MATCH_TH):        
        list_img[idx]['isFound'] = True
        list_img[idx]['foundIdx'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
        list_img[idx]['matchVal'] = len(matches)
        list_img[idx]['foundImg'] = frame
        logger.info("frame %d is found at idx %d", idx, temp['foundIdx'])
        idx = idx + 1
        
        
#        max_match = len(matches)
#        found_frame = frame
    
    #matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, flags=2) # draw matching comparison
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (idx>=MAX_CTRL_FRAME):
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+SKIPPED_NUM) # skip every SKIPPED_NUM frames
    #end while
#logger.debug(matches_list)
#logger.debug(max_match)
cap.release
cv2.destroyWindow('app')

# SHOW FRAME
for img in list_img:
    cv2.namedWindow('CONTROL IMG', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CONTROL IMG', 500,700)
    cv2.imshow('CONTROL IMG', img['img'])
    
    if (img['isFound'] == True):
        cv2.namedWindow('FOUND', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FOUND', 500,700)
        cv2.imshow('FOUND', img['foundImg'])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
