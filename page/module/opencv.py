import numpy as np
import cv2
import sys
from page.module.img_cropped import img_crop, use_crop
from page.module.dilb_use import use_dlib

def readPoints(path) :

    points = [];

    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

def applyAffineTransform(src, srcTri, dstTri, size) :
    
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# 삼각형을 만들면서 합성을 해준다
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :
    
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def use_opencv(me, name, alpha):
    filename1 = name
    filename2 = me
    
    name = name.split('.')[0].split('/')[-1]
    
    # 받은 이미지를 잘라내고 크기를 동일하게 한다
    img1 = img_crop(filename1)
    img2 = use_crop(filename2)
    
    # 이미지의 랜드마크를 찾아서 txt로 저장한다
    points1 = use_dlib(img1,name)
    points2 = use_dlib(img2,'me')

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 저장한 txt를 사용해서 landmark를 불러온다
    # points1 = readPoints(name + '1.txt')
    # points2 = readPoints('me' + '1.txt')
    points = [];

    # 모든 랜드마크를 돌면서 point1과 point2의 land마크를 묶어서 list에 저장한다
    #alpha = 0.3
    #while alpha<=0.7:
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))

    max_r = max(img1.shape[0],img2.shape[0])
    max_c = max(img1.shape[1],img2.shape[1])
    imgMorph = np.zeros((max_r,max_c,3), dtype = img1.dtype)

    # 미리 만든 삼각형의 세꼭지점을 확인하면서 이미지를 생성해준다
    with open("tri copy 2.txt") as file :
        for line in file :
            x,y,z = line.split()
            
            x = int(x)
            y = int(y)
            z = int(z)
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    return np.uint8(imgMorph)