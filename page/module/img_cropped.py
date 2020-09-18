import face_recognition
import cv2

# face를 찾아서 이미지를 일정한 크기로 자르고 크기를 모두 동일하게 만들어주는 함수
def img_crop(title):
    img = face_recognition.load_image_file(title)
    return use_crop(img)

def use_crop(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_copy = img_rgb.copy()

    scale_factor = 1.1
    sz1 = img_rgb.shape[1] * 2
    sz2 = img_rgb.shape[0] * 2

    face_locations = face_recognition.face_locations(img_rgb)

    for top, right, bottom, left in face_locations:

        crop_img = img_rgb[top:bottom, left:right]

        width = right - left
        height = bottom - top
        cX = left + width // 2
        cY = top + height // 2
        M = (abs(width) + abs(height)) / 2

        newLeft = max(0, int(cX - scale_factor * M))
        newTop = max(0, int(cY - scale_factor * M))
        newRight = min(img_rgb.shape[1], int(cX + scale_factor * M))
        newBottom = min(img_rgb.shape[0], int(cY + scale_factor * M))

        #cv2.rectangle(img_rgb_copy, (newLeft, newTop), (newRight, newBottom), (255, 0, 0), 2)
        
        # face가 있는 부분만 잘라낸다 
        dst = img_rgb_copy[newTop:newBottom, newLeft:newRight]
        
        # face가 있는 부분의 크기를 항상 500 * 500으로 설정한다
        dst = cv2.resize(dst, dsize=(200,200),interpolation=cv2.INTER_AREA)

        # cv2.imshow("Morphed Face", dst)
        # cv2.waitKey(0)
        #print("!!!!!!!!!!!!!!!!",dst)
        return dst

