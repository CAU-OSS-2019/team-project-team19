from keras.models import load_model
import numpy as np 
import keras 
import os
from PIL import Image
from collections import Counter
import random

def his_equ(img,level=256,mode='RGB'):
    if mode == 'RGB' or mode == 'rgb':
        r, g, b = [], [], []
        w, h = img.size[0], img.size[1]
        sum_pix = w * h
        pix = img.load()
        for x in range(w):
            for y in range(h):
                r.append(pix[x, y][0])
                g.append(pix[x, y][1])
                b.append(pix[x, y][2])
        r_c = dict(Counter(r))
        g_c = dict(Counter(g))
        b_c = dict(Counter(b))
        r_p,g_p,b_p = [],[],[]

        for i in range(level):
            if r_c.__contains__(i):
                r_p.append(float(r_c[i]) / sum_pix)
            else:
                r_p.append(0)
            if g_c.__contains__(i):
                g_p.append(float(g_c[i])/sum_pix)
            else:
                g_p.append(0)
            if b_c.__contains__(i):
                b_p.append(float(b_c[i])/sum_pix)
            else:
                b_p.append(0)
        temp_r,temp_g,temp_b = 0,0,0
        for i in range(level):
            temp_r += r_p[i]
            r_p[i] = int(temp_r * (level-1))
            temp_b += b_p[i]
            b_p[i] = int(temp_b *(level-1))
            temp_g += g_p[i]
            g_p[i] = int(temp_g*(level -1))
        new_photo = Image.new('RGB',(w,h))
        for x in range(w):
            for y in range(h):
                new_photo.putpixel((x,y),(r_p[pix[x,y][0]],g_p[pix[x,y][1]],b_p[pix[x,y][2]]))
        return new_photo

fruit_list = ['Pear', 'Strawberry', 'Lemon', 'Cherry Rainier', 'Apple Braeburn', 'Mango', 'Banana', 'Tomato Cherry Red', 'Orange', 'Peach']
model_cnn = load_model('my_model.h5')


while(True):
    x_test = []
    i = input("please input a number from 0-9 ,stand for one of the ten kinds fruit\n")
    fruit = fruit_list[int(i)]
    print("the fruit is %s "%fruit)
    a = os.listdir('../test/' + fruit)
    imageFile = random.sample(a, 1)[0]
    img = np.array(his_equ(Image.open('../test/' + fruit + '/' + imageFile)))
    x_test.append(img)
    x_test = np.array(x_test)
    fruit_p = model_cnn.predict(x_test)
    fruit_p = fruit_list[np.argmax(fruit_p)]
    print("the fruit is %s predicted\n"%fruit_p)


