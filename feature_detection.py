# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:50:31 2019
@author: olanseverson

https://realpython.com/python-logging/
https://pysource.com/2018/03/23/feature-matching-brute-force-opencv-3-4-with-python-3-tutorial-26/
https://www.life2coding.com/resize-opencv-window-according-screen-resolution/
https://computer-vision-talks.com/2011-07-13-comparison-of-the-opencv-feature-detection-algorithms/
https://arxiv.org/pdf/1710.02726.pdf
"""

import cv2
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


MAX_CTRL_FRAME = 13 # number of control frame
SKIPPED_NUM = 3 #skip reading the frame every Skipped_num frame

## GET THE IMAGE 
list_img = []
th_dict = {'n_match': [100,280,170,130,240,120,250,150,180,100,200,210,170], #[180,280,170,130,240,120,250,150,180,100,200,210,170],
                  'distance': [35,35,30,35,20,35,35,35,40,20,35,35,30]}
print(th_dict['distance'][0])
for i in range (1,MAX_CTRL_FRAME + 1):
    temp = {}
    temp['img'] = cv2.imread("./control_frame/" + str(i) + ".jpg") # get control image
    temp['isFound'] = False
    temp['foundIdx'] = 0
    temp['matchVal'] = 0
    temp['foundImg'] = None
    list_img.append(temp)
    
cv2.waitKey(0)
matches_list = []
max_match = 0

## GET THE VIDEO
cap = cv2.VideoCapture('ori.mp4') # read Video that we want to check
logger.info('frame count: %d', cap.get(cv2.CAP_PROP_FRAME_COUNT))
logger.info('fps : %d', cap.get(cv2.CAP_PROP_FPS))

idx = 0
while True:
    _, frame = cap.read()
    logger.debug("frame nth: %d", cap.get(cv2.CAP_PROP_POS_FRAMES))
    if frame is None:
        break
    
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
#    logger.debug('%d %d', th_dict['distance'][idx], th_dict['n_match'][idx])
    matches = list(filter(lambda x: x.distance<th_dict['distance'][idx], 
                          matches)) # ignore value if smaller than match_distance
    logger.debug('feature found: %d', len(matches))
    ## Find most similar picture
    if(len(matches)>th_dict['n_match'][idx]):        
        list_img[idx]['isFound'] = True
        list_img[idx]['foundIdx'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
        list_img[idx]['matchVal'] = len(matches)
        list_img[idx]['foundImg'] = frame
        logger.info("frame %d is found at idx %d", idx, list_img[idx]['foundIdx'])
        idx = idx + 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (idx>=MAX_CTRL_FRAME):
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES)+SKIPPED_NUM) # skip every SKIPPED_NUM frames
    #end while
    
cv2.destroyWindow('app')

## SHOW FRAME
for img in list_img:
    
    cv2.namedWindow('CONTROL IMG', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CONTROL IMG', 500,700)
    cv2.moveWindow('CONTROL IMG', 0,0) 
    cv2.imshow('CONTROL IMG', img['img'])
    
    if (img['isFound'] == True):
        
        cv2.namedWindow('FOUND', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('FOUND', 500,700)
        cv2.moveWindow('FOUND', 600,0) 
        cv2.imshow('FOUND', img['foundImg'])
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def drawing_feature(img1, img2):
    # ORB Detector
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)
    matches = list(filter(lambda x: x.distance<35, 
                          matches)) # ignore value if smaller than match_distance
    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    
    logger.debug("total value is [%d]", len(matches))

    ## Draw
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 800,800)
    cv2.moveWindow('result', 300,0) 
    cv2.imshow("result", img2)
    cv2.imshow("result", matching_result)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#cap.set(cv2.CAP_PROP_POS_FRAMES, 13)
#_, img2 = cap.read()
#drawing_feature(list_img[0]['img'], img2)

cap.release

