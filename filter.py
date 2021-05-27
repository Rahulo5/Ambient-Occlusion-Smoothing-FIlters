import numpy as np
import sys
import cv2
from sklearn.metrics import mean_squared_error

def filter(img,s1,s2,g1,g2,w):
    ret = img.copy()
    im = np.zeros((img.shape[0] + 2*w, img.shape[1] + 2*w, 3), dtype = 'float32')
    im[w:im.shape[0]-w,w:im.shape[1]-w] = img
    go = np.zeros((img.shape[0] + 2*w, img.shape[1] + 2*w, 3), dtype = 'float32')
    go[w:im.shape[0]-w,w:im.shape[1]-w] = g1
    gt = np.zeros((img.shape[0] + 2*w, img.shape[1] + 2*w, 3), dtype = 'float32')
    gt[w:im.shape[0]-w,w:im.shape[1]-w] = g2
    img = im
    g1 = go
    g2 = gt
    for i in range(w,len(img)-w):
        if(i%1 == 0):
            print(i)
        for j in range(w,len(img[0])-w):
            t1 = (img[i-w:i+w+1,j-w:j+w+1])[:,:,0]
            r = (g1[i-w:i+w+1,j-w:j+w+1])[:,:,0]
            s = (g2[i-w:i+w+1,j-w:j+w+1]) - g2[i][j]
            t2 = np.exp(-(((g1[i-w:i+w+1,j-w:j+w+1]-g1[i][j])[:,:,0])**2 + ((g1[i-w:i+w+1,j-w:j+w+1]-g1[i][j])[:,:,1])**2 + ((g1[i-w:i+w+1,j-w:j+w+1]-g1[i][j])[:,:,2])**2)/(s1**2))
            t3 = np.exp(-(((g2[i-w:i+w+1,j-w:j+w+1]-g2[i][j])[:,:,0])**2 + ((g2[i-w:i+w+1,j-w:j+w+1]-g2[i][j])[:,:,1])**2 + ((g2[i-w:i+w+1,j-w:j+w+1]-g2[i][j])[:,:,2])**2)/(s2**2))
            z = np.sum(t2*t3)
            if z == 0:
                req = 0
            else:
                req = np.sum((t1*t2*t3))/z
            ret[i-w][j-w][0] = req
            ret[i-w][j-w][1] = req
            ret[i-w][j-w][2] = req
    return ret



img = cv2.imread("cat_ao_1.png").astype(np.float32)
exp = cv2.imread("cat_ao_128.png").astype(np.float32)
g1 = cv2.imread("cat_normal.png").astype(np.float32)
g2 = cv2.imread("cat_position.png").astype(np.float32)
s1 = (np.array([35,30,40])).tolist()
s2 = (np.array([14])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*1.59381758))
        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*1.59381758))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("cat" + str(x) + "_" + str(y) +"get.png",p)

'''img = cv2.imread("sponza_ao_1.png").astype(np.float32)
exp = cv2.imread("sponza_ao_128.png").astype(np.float32)
g1 = cv2.imread("sponza_normal.png").astype(np.float32)
g2 = cv2.imread("sponza_position.png").astype(np.float32)
s1 = (np.array([30,35,40])).tolist()
s2 = (np.array([14])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*1.44879845))
        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*1.44879845))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("sponza" + str(x) + "_" + str(y) +"get.png",p)

img = cv2.imread("dragon_ao_1.png").astype(np.float32)
exp = cv2.imread("dragon_ao_128.png").astype(np.float32)
g1 = cv2.imread("dragon_normal.png").astype(np.float32)
g2 = cv2.imread("dragon_position.png").astype(np.float32)
s1 = (np.array([45,40])).tolist()
s2 = (np.array([14])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*1.54827215))

        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*1.54827215))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("dragon" + str(x) + "_" + str(y) +"get.png",p)

img = cv2.imread("house_ao_1.png").astype(np.float32)
exp = cv2.imread("house_ao_128.png").astype(np.float32)
g1 = cv2.imread("house_normal.png").astype(np.float32)
g2 = cv2.imread("house_position.png").astype(np.float32)
s1 = (np.array([30])).tolist()
s2 = (np.array([10,2,3])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*1.54827215))

        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*1.54827215))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("house" + str(x) + "_" + str(y) +"get.png",p)

img = cv2.imread("fireplace_ao_1.png").astype(np.float32)
exp = cv2.imread("fireplace_ao_128.png").astype(np.float32)
g1 = cv2.imread("fireplace_normal.png").astype(np.float32)
g2 = cv2.imread("fireplace_position.png").astype(np.float32)
s1 = (np.array([35,40,45])).tolist()
s2 = (np.array([14])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*5.3853215))
        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*5.3853215))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("fireplace" + str(x) + "_" + str(y) +"get.png",p)


img = cv2.imread("fireplace_ao_1.png").astype(np.float32)
exp = cv2.imread("fireplace_ao_128.png").astype(np.float32)
g1 = cv2.imread("fireplace_normal.png").astype(np.float32)
g2 = cv2.imread("fireplace_position.png").astype(np.float32)
s1 = (np.array([35,40,45])).tolist()
s2 = (np.array([14])).tolist()
for x in s1:
    for y in s2:
        p = filter(img,x,y,g1,g2,int(3*y*1.3853215))
        print(np.sum(np.absolute(exp[:,:,0] - p[:,:,0]))/len(img)/len(img[0]),x,y,int(3*y*1.3853215))
        print(np.sum(np.absolute(exp[:,:,0] - img[:,:,0]))/len(img)/len(img[0]))
        cv2.imwrite("fireplace" + str(x) + "_" + str(y) +"get.png",p)'''