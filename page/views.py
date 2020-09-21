from django.shortcuts import render
import cv2, os,json
import numpy as np
import face_recognition as fr
import matplotlib.pyplot as plt
from page.module.img_cropped import img_crop, use_crop
from page.module.opencv import use_opencv
from page.module.base64 import use_base64

def index(request):
    return render(request, 'index.html')

def who(request):

    if request.method == 'POST':
        sex = request.POST.get('sex')
        me = fr.load_image_file(request.FILES['profile_pt'].file)

        me = use_crop(me)

        b,g,r = cv2.split(me)
        me = cv2.merge([r,g,b])
        #me = use_base64(me)

        # session에 저장하기 위해 list로 변경
        list_me = me.tolist()
        # me로 저장
        request.session['me'] = list_me

        # 얼굴 찾기
        me_face_locations = fr.face_locations(me, model='cnn')
        for (top, right, bottom, left) in me_face_locations:
            me_face = me[top:bottom, left:right]
        enc_me = fr.face_encodings(me_face)

        me = use_base64(me)

        who_top = []
        # 전체 배우 데이터
        file_path = "actor.json"
        with open(file_path, "r") as json_file:
            json_data = json.load(json_file)
            if sex == 'man':
                for i in range(len(json_data['man'])):
                    image = json_data['man'][i]['image']

                    enc = json_data['man'][i]['enc_actor']
                    
                    img_src = use_base64(image)
                    
                    who_top.append(
                        {'name': json_data['man'][i]['name'],
                        'how': (fr.face_distance(np.array(enc), enc_me)).tolist(),
                        'image': img_src,
                        })

            else:
                for i in range(len(json_data['woman'])):
                    image = json_data['woman'][i]['image']
                    enc = json_data['woman'][i]['enc_actor']
                    
                    # 배우 이미지 가져오기
                    img_src = use_base64(image)

                    who_top.append(
                        {'name': json_data['woman'][i]['name'],
                        'how': (fr.face_distance(np.array(enc), enc_me)).tolist(),
                        'image': img_src,
                        })

            top3 = sorted(who_top, key=(lambda x: x['how']))

            return render(request, 'who.html', 
                            {
                            # 'me':me,
                            'img_src': [top3[0], top3[1], top3[2]]
                            }
                            )
    else:
        return render(request, 'index.html')


def merge(request):
    return render(request, 'merge.html')

def mixed(request, name):
    arr = request.session['me']
    me = np.array(arr, dtype = np.uint8)

    me_re = use_crop(me)
    img_3 = use_opencv(me, 'page/image/'+name+'.jpg',0.3)
    img_5 = use_opencv(me, 'page/image/'+name+'.jpg',0.5)
    img_7 = use_opencv(me, 'page/image/'+name+'.jpg',0.7)
    me_no = img_crop('page/image/'+name+'.jpg')

    # 색상을 반전시켜준다
    b,g,r = cv2.split(me_re)
    me_re = cv2.merge([r,g,b])
    me_re = use_base64(me_re)

    b,g,r = cv2.split(img_3)
    img_3 = cv2.merge([r,g,b])
    img_3 = use_base64(img_3)

    b,g,r = cv2.split(img_5)
    img_5 = cv2.merge([r,g,b])
    img_5 = use_base64(img_5)

    b,g,r = cv2.split(img_7)
    img_7 = cv2.merge([r,g,b])
    img_7 = use_base64(img_7)

    b,g,r = cv2.split(me_no)
    me_no = cv2.merge([r,g,b])
    me_no = use_base64(me_no)

    return render(request, 'mix.html',{'me_re':me_re,'me_3':img_3,'me_5':img_5,'me_7':img_7,'me_no':me_no})

    #cv2.imshow("Morphed Face", me)
    #cv2.waitKey(0)

def mixed_all(request, name1, name2, name3):
    arr = request.session['me']
    me = np.array(arr, dtype = np.uint8)
    
    me_re = use_crop(me)
    img_1 = use_opencv(me, 'page/image/'+name1+'.jpg',0.5)
    img_2 = use_opencv(me, 'page/image/'+name2+'.jpg',0.5)
    img_3 = use_opencv(me, 'page/image/'+name3+'.jpg',0.5)
    me_no_1 = img_crop('page/image/'+name1+'.jpg')
    me_no_2 = img_crop('page/image/'+name2+'.jpg')
    me_no_3 = img_crop('page/image/'+name3+'.jpg')

    b,g,r = cv2.split(me_re)
    me_re = cv2.merge([r,g,b])
    me_re = use_base64(me_re)

    b,g,r = cv2.split(img_1)
    img_1 = cv2.merge([r,g,b])
    img_1 = use_base64(img_1)

    b,g,r = cv2.split(img_2)
    img_2 = cv2.merge([r,g,b])
    img_2 = use_base64(img_2)

    b,g,r = cv2.split(img_3)
    img_3 = cv2.merge([r,g,b])
    img_3 = use_base64(img_3)

    b,g,r = cv2.split(me_no_1)
    me_no_1 = cv2.merge([r,g,b])
    me_no_1 = use_base64(me_no_1)

    b,g,r = cv2.split(me_no_2)
    me_no_2 = cv2.merge([r,g,b])
    me_no_2 = use_base64(me_no_2)

    b,g,r = cv2.split(me_no_3)
    me_no_3 = cv2.merge([r,g,b])
    me_no_3 = use_base64(me_no_3)

    return render(request, 'mix_all.html',{'me':me_re,'me_1':img_1,'me_2':img_2,'me_3':img_3,'me_no_1':me_no_1,'me_no_2':me_no_2,'me_no_3':me_no_3})