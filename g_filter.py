import numpy as np
import cv2

def filter(I,p,q,e,w):
    ret = I.copy()
    coff1 = np.zeros((I.shape[0] + 2*w,I.shape[1] + 2*w,1,6),dtype = 'float32')
    coff2 = np.zeros((I.shape[0] + 2*w,I.shape[1] + 2*w,1,1),dtype = 'float32')
    im = np.zeros((I.shape[0] + 2*w, I.shape[1] + 2*w, 3), dtype = 'float32')
    im[w:im.shape[0]-w,w:im.shape[1]-w] = I
    go = np.zeros((p.shape[0] + 2*w, p.shape[1] + 2*w, 3), dtype = 'float32')
    go[w:im.shape[0]-w,w:im.shape[1]-w] = p
    gp = np.zeros((p.shape[0] + 2*w, p.shape[1] + 2*w, 3), dtype = 'float32')
    gp[w:im.shape[0]-w,w:im.shape[1]-w] = q
    p = go
    q = gp
    I = im[:,:,0]
    for i in range(w,len(I)-w):
        if(i%100 == 0):
            print(i)
        for j in range(w,len(I[0])-w):
            t1 = (I[i-w:i+w+1,j-w:j+w+1])
            c1 = p[i-w:i+w+1,j-w:j+w+1][:,:,0]
            c2 = p[i-w:i+w+1,j-w:j+w+1][:,:,1]
            c3 = p[i-w:i+w+1,j-w:j+w+1][:,:,2]
            c4 = q[i-w:i+w+1,j-w:j+w+1][:,:,0]
            c5 = q[i-w:i+w+1,j-w:j+w+1][:,:,1]
            c6 = q[i-w:i+w+1,j-w:j+w+1][:,:,2]
            '''print("I",t1)
            print("p1",c1)
            print("p2",c2)
            print("p3",c3)
            print("p4",c4)
            print("p5",c5)
            print("p6",c6)'''
            sigma = np.zeros((6,6), dtype = 'float32')
            c1m = np.mean(c1)
            c2m = np.mean(c2)
            c3m = np.mean(c3)
            c4m = np.mean(c4)
            c5m = np.mean(c5)
            c6m = np.mean(c6)
            #print("pmean",c1m,c2m,c3m,c4m,c5m,c6m)
            sigma[0][0] = np.mean((c1 - c1m)*(c1 - c1m))
            sigma[0][1] = np.mean((c1 - c1m)*(c2 - c2m))
            sigma[0][2] = np.mean((c1 - c1m)*(c3 - c3m))
            sigma[0][3] = np.mean((c1 - c1m)*(c4 - c4m))
            sigma[0][4] = np.mean((c1 - c1m)*(c5 - c5m))
            sigma[0][5] = np.mean((c1 - c1m)*(c6 - c6m))
            sigma[1][0] = sigma[0][1]
            sigma[1][1] = np.mean((c2 - c2m)*(c2 - c2m))
            sigma[1][2] = np.mean((c2 - c2m)*(c3 - c3m))
            sigma[1][3] = np.mean((c2 - c2m)*(c4 - c4m))
            sigma[1][4] = np.mean((c2 - c2m)*(c5 - c5m))
            sigma[1][5] = np.mean((c2 - c2m)*(c6 - c6m))
            sigma[2][0] = sigma[0][2]
            sigma[2][1] = sigma[1][2]
            sigma[2][2] = np.mean((c3 - c3m)*(c3 - c3m))
            sigma[2][3] = np.mean((c3 - c3m)*(c4 - c4m))
            sigma[2][4] = np.mean((c3 - c3m)*(c5 - c5m))
            sigma[2][5] = np.mean((c3 - c3m)*(c6 - c6m))
            sigma[3][0] = sigma[0][3]
            sigma[3][1] = sigma[1][3]
            sigma[3][2] = sigma[2][3]
            sigma[3][3] = np.mean((c4 - c4m)*(c4 - c4m))
            sigma[3][4] = np.mean((c4 - c4m)*(c5 - c5m))
            sigma[3][5] = np.mean((c4 - c4m)*(c6 - c6m))
            sigma[4][0] = sigma[0][4]
            sigma[4][1] = sigma[1][4]
            sigma[4][2] = sigma[2][4]
            sigma[4][3] = sigma[3][4]
            sigma[4][4] = np.mean((c5 - c5m)*(c5 - c5m))
            sigma[4][5] = np.mean((c5 - c5m)*(c6 - c6m))
            sigma[5][0] = sigma[0][5]
            sigma[5][1] = sigma[1][5]
            sigma[5][2] = sigma[2][5]
            sigma[5][3] = sigma[3][5]
            sigma[5][4] = sigma[4][5]
            sigma[5][5] = np.mean((c6 - c6m)*(c6 - c6m))
            U = e*np.identity(6,dtype = 'float32')
            #print(sigma)
            first_term = np.linalg.inv(sigma + U)
            t1m = np.mean(t1)
            #print("t1m",t1m)
            
            f1 = np.mean(c1*t1 - c1m*t1m)
            f2 = np.mean(c2*t1 - c2m*t1m)
            f3 = np.mean(c3*t1 - c3m*t1m)
            f4 = np.mean(c4*t1 - c4m*t1m)
            f5 = np.mean(c5*t1 - c5m*t1m)
            f6 = np.mean(c6*t1 - c6m*t1m)
            second_term = np.array([[f1],[f2],[f3],[f4],[f5],[f6]])
            '''print("first_term",first_term)
            print("second_term",second_term)'''
            muik = np.array([[c1m],[c2m],[c3m],[c4m],[c5m],[c6m]])
            #print("muik",muik)
            ak = np.dot(first_term,second_term)
            bk = t1m - np.dot(ak.transpose(),muik)
            '''print("both_ak_bk",ak,bk,muik,t1m)
            print(bk)
            print(w)'''
            coff1[i,j] = ak.transpose()
            coff2[i,j] = bk
            #print("coff",i,j,coff1[0][2])
            #print(coff2[0][2])

            '''if(i>5):
                print(i,j)
                print("coff")
                for k in range(5,10):
                    for l in range(5,10):
                        print(coff1[k][l])
                        print(coff2[k][l])
        if(i%50 == 0):
            print(i,len(I))'''
    
    for i in range(w,ret.shape[0]+w):
        for j in range(w,ret.shape[1]+w):
            z1 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,0])*p[i,j,0]
            z2 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,1])*p[i,j,1]
            z3 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,2])*p[i,j,2]
            z4 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,3])*q[i,j,0]
            z5 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,4])*q[i,j,1]
            z6 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,5])*q[i,j,2]
            z7 = np.mean(coff2[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,0])
            x = z1 + z2 + z3 + z4 + z5 + z6 + z7
            ret[i-w,j-w,0] = x
            ret[i-w,j-w,1] = x
            ret[i-w,j-w,2] = x
    print("max_min",np.max(ret),np.min(ret))
    return ret*255

img = cv2.imread("dragon_ao_1.png").astype(np.float32)/255
exp = cv2.imread("dragon_ao_128.png").astype(np.float32)
g1 = cv2.imread("dragon_normal.png").astype(np.float32)/255
g2 = cv2.imread("dragon_position.png").astype(np.float32)/255

eps = [0.000001]
w = [15,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("dragon_guided_" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("cat_ao_1.png").astype(np.float32)/255
exp = cv2.imread("cat_ao_128.png").astype(np.float32)
g1 = cv2.imread("cat_normal.png").astype(np.float32)/255
g2 = cv2.imread("cat_position.png").astype(np.float32)/255

eps = [0.000001]
w = [15,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("cat_guided_" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("fireplace_ao_1.png").astype(np.float32)/255
exp = cv2.imread("fireplace_ao_128.png").astype(np.float32)
g1 = cv2.imread("fireplace_normal.png").astype(np.float32)/255
g2 = cv2.imread("fireplace_position.png").astype(np.float32)/255

eps = [0.000001]
w = [15,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("fireplace_guided_" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("house_ao_1.png").astype(np.float32)/255
exp = cv2.imread("house_ao_128.png").astype(np.float32)
g1 = cv2.imread("house_normal.png").astype(np.float32)/255
g2 = cv2.imread("house_position.png").astype(np.float32)/255

eps = [0.000001]
w = [15,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("house_guided_" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("sponza_ao_1.png").astype(np.float32)/255
exp = cv2.imread("sponza_ao_128.png").astype(np.float32)
g1 = cv2.imread("sponza_normal.png").astype(np.float32)/255
g2 = cv2.imread("sponza_position.png").astype(np.float32)/255

eps = [0.000001]
w = [15,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("sponza_guided_" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))