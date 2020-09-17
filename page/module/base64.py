import base64
from io import BytesIO
import numpy as np
from PIL import Image

def use_base64(image):

    actor_image = np.asarray(image, dtype=np.uint8)
    actor = Image.fromarray(actor_image, 'RGB')
    trans_src = BytesIO()
    actor.save(trans_src, "JPEG") # pick your format
    data64 = base64.b64encode(trans_src.getvalue())
    img_src = u'data:img/jpeg;base64,'+data64.decode('utf-8') 
    
    return img_src