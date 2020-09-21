import dlib
import numpy as np
from skimage import io
import cv2
import sys

# face landmark를 구하는 함수
def use_dlib(img,filename):

       predictor_path = "page/module/shape_predictor_68_face_landmarks.dat"

       detector = dlib.get_frontal_face_detector()
       predictor = dlib.shape_predictor(predictor_path)

       dets = detector(img)

       for k, d in enumerate(dets):
              shape = predictor(img, d)
       
       vec = np.empty([76, 2], dtype = int)
       # vec에 shape의 각 좌표들을 기억한다
       for b in range(68):
              vec[b][0] = shape.part(b).x
              vec[b][1] = shape.part(b).y
              #cv2.circle(img, (vec[b][0], vec[b][1]), 2, (0, 255, 0), -1)
       # cv2.imshow('result', img)
       # key = cv2.waitKey(0)

       r = len(img)
       c = len(img[0])

       vec[68][0], vec[68][1] = 0,0
       vec[69][0], vec[69][1] = 0,r//2
       vec[70][0], vec[70][1] = 0,r-1
       vec[71][0], vec[71][1] = c//2,r-1
       vec[72][0], vec[72][1] = c-1,r-1
       vec[73][0], vec[73][1] = c-1,r//2
       vec[74][0], vec[74][1] = c-1,0
       vec[75][0], vec[75][1] = c//2,0

       return vec
