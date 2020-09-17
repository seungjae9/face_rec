import dlib
import numpy as np
from skimage import io
import cv2
import sys

# face landmark를 수하는 함수
def use_dlib(img,filename):

       predictor_path = "page/module/shape_predictor_68_face_landmarks.dat"

       detector = dlib.get_frontal_face_detector()
       predictor = dlib.shape_predictor(predictor_path)

       dets = detector(img)

       for k, d in enumerate(dets):
              shape = predictor(img, d)
       
       vec = np.empty([68, 2], dtype = int)
       # vec에 shape의 각 좌표들을 기억한다
       for b in range(68):
              vec[b][0] = shape.part(b).x
              vec[b][1] = shape.part(b).y

       r = len(img)
       c = len(img[0])

       fw = open(filename + '1.txt','w')

       # landmark를 txt파일도 저장한다.
       # txt 파일을 사용해서 접근한다
       for i in vec:
              fw.write(str(i[0]) + " " + str(i[1])+'\n')
       # 배경도 합성을 하기 때문에 이미지의 시작값 끝값 중간값도 추가를 해준다 
       fw.write(str(0) + " " + str(0)+'\n')
       fw.write(str(0) + " " + str(r//2)+'\n')
       fw.write(str(0) + " " + str(r-1)+'\n')
       fw.write(str(c//2) + " " + str(r-1)+'\n')
       fw.write(str(c-1) + " " + str(r-1)+'\n')
       fw.write(str(c-1) + " " + str(r//2)+'\n')
       fw.write(str(c-1) + " " + str(0)+'\n')
       fw.write(str(c//2) + " " + str(0))
       fw.close()

# def use_dlib_swap(img,filename):
    
#        predictor_path = "shape_predictor_68_face_landmarks.dat"

#        detector = dlib.get_frontal_face_detector()
#        predictor = dlib.shape_predictor(predictor_path)

#        dets = detector(img)

#        for k, d in enumerate(dets):
#               shape = predictor(img, d)
       
#        vec = np.empty([68, 2], dtype = int)
#        # vec에 shape의 각 좌표들을 기억한다
#        for b in range(68):
#               vec[b][0] = shape.part(b).x
#               vec[b][1] = shape.part(b).y

#        r = len(img)
#        c = len(img[0])

#        fw = open(filename + '_swap.txt','w')

#        # landmark를 txt파일도 저장한다.
#        # txt 파일을 사용해서 접근한다
#        for i in vec:
#               fw.write(str(i[0]) + " " + str(i[1])+'\n')