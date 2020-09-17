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
        file_path = "sample.json"
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
                        {'name': json_data['woman'][i]['woman'],
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
    print(name)

    img = use_opencv(me, 'page/image/'+name+'.jpg')

    # 색상을 반전시켜준다
    b,g,r = cv2.split(img)
    img = cv2.merge([r,g,b])

    img = use_base64(img)

    return render(request, 'mix.html',{'me':img})


    #cv2.imshow("Morphed Face", me)
    #cv2.waitKey(0)