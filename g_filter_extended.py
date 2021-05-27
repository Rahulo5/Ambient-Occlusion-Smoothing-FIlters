import numpy as np
import cv2

def filter(I,p,q,e,w):
    ret = I.copy()
    coff1 = np.zeros((I.shape[0] + 2*w,I.shape[1] + 2*w,1,27),dtype = 'float32')
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
        if(i%10 == 0):
            print(i)
        for j in range(w,len(I[0])-w):
            t1 = (I[i-w:i+w+1,j-w:j+w+1])
            c1 = p[i-w:i+w+1,j-w:j+w+1][:,:,0]
            c2 = p[i-w:i+w+1,j-w:j+w+1][:,:,1]
            c3 = p[i-w:i+w+1,j-w:j+w+1][:,:,2]
            c4 = q[i-w:i+w+1,j-w:j+w+1][:,:,0]
            c5 = q[i-w:i+w+1,j-w:j+w+1][:,:,1]
            c6 = q[i-w:i+w+1,j-w:j+w+1][:,:,2]
            c7 = c1*c1
            c8 = c1*c2
            c9 = c1*c3
            c10 = c1*c4
            c11 = c1*c5
            c12 = c1*c6
            c13 = c2*c2
            c14 = c2*c3
            c15 = c2*c4
            c16 = c2*c5
            c17 = c2*c6
            c18 = c3*c3
            c19 = c3*c4
            c20 = c3*c5
            c21 = c3*c6
            c22 = c4*c4
            c23 = c4*c5
            c24 = c4*c6
            c25 = c5*c5
            c26 = c5*c6
            c27 = c6*c6
            cs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27]
            '''print("I",t1)
            print("p1",c1)
            print("p2",c2)
            print("p3",c3)
            print("p4",c4)
            print("p5",c5)
            print("p6",c6)'''
            sigma = np.zeros((27,27), dtype = 'float32')
            c1m = np.mean(c1)
            c2m = np.mean(c2)
            c3m = np.mean(c3)
            c4m = np.mean(c4)
            c5m = np.mean(c5)
            c6m = np.mean(c6)
            c7m = np.mean(c7)
            c8m = np.mean(c8)
            c9m = np.mean(c9)
            c10m = np.mean(c10)
            c11m = np.mean(c11)
            c12m = np.mean(c12)
            c13m = np.mean(c13)
            c14m = np.mean(c14)
            c15m = np.mean(c15)
            c16m = np.mean(c16)
            c17m = np.mean(c17)
            c18m = np.mean(c18)
            c19m = np.mean(c19)
            c20m = np.mean(c20)
            c21m = np.mean(c21)
            c22m = np.mean(c22)
            c23m = np.mean(c23)
            c24m = np.mean(c24)
            c25m = np.mean(c25)
            c26m = np.mean(c26)
            c27m = np.mean(c27)
            cims = [c1m,c2m,c3m,c4m,c5m,c6m,c7m,c8m,c9m,c10m,c11m,c12m,c13m,c14m,c15m,c16m,c17m,c18m,c19m,c20m,c21m,c22m,c23m,c24m,c25m,c26m,c27m]
            for i1 in range(0,27):
                for j1 in range(0,27):
                    if(i1<=j1):
                        sigma[i1][j1] = np.mean((cs[i1]-cims[i1])*(cs[j1]-cims[j1]))
                    else:
                        sigma[i1][j1] = sigma[j1][i1]
            U = e*np.identity(27,dtype = 'float32')
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
            f7 = np.mean(c7*t1 - c7m*t1m)
            f8 = np.mean(c8*t1 - c8m*t1m)
            f9 = np.mean(c9*t1 - c9m*t1m)
            f10 = np.mean(c10*t1 - c10m*t1m)
            f11 = np.mean(c11*t1 - c11m*t1m)
            f12 = np.mean(c12*t1 - c12m*t1m)
            f13 = np.mean(c13*t1 - c13m*t1m)
            f14 = np.mean(c14*t1 - c14m*t1m)
            f15 = np.mean(c15*t1 - c15m*t1m)
            f16 = np.mean(c16*t1 - c16m*t1m)
            f17 = np.mean(c17*t1 - c17m*t1m)
            f18 = np.mean(c18*t1 - c18m*t1m)
            f19 = np.mean(c19*t1 - c19m*t1m)
            f20 = np.mean(c20*t1 - c20m*t1m)
            f21 = np.mean(c21*t1 - c21m*t1m)
            f22 = np.mean(c22*t1 - c22m*t1m)
            f23 = np.mean(c23*t1 - c23m*t1m)
            f24 = np.mean(c24*t1 - c24m*t1m)
            f25 = np.mean(c25*t1 - c25m*t1m)
            f26 = np.mean(c26*t1 - c26m*t1m)
            f27 = np.mean(c27*t1 - c27m*t1m)
            second_term = np.array([[f1],[f2],[f3],[f4],[f5],[f6],[f7],[f8],[f9],[f10],[f11],[f12],[f13],[f14],[f15],[f16],[f17],[f18],[f19],[f20],[f21],[f22],[f23],[f24],[f25],[f26],[f27]])
            #print("first_term",first_term)
            #print("second_term",second_term)
            muik = np.array([[c1m],[c2m],[c3m],[c4m],[c5m],[c6m],[c7m],[c8m],[c9m],[c10m],[c11m],[c12m],[c13m],[c14m],[c15m],[c16m],[c17m],[c18m],[c19m],[c20m],[c21m],[c22m],[c23m],[c24m],[c25m],[c26m],[c27m]])
            #print("muik",muik)
            ak = np.dot(first_term,second_term)
            bk = t1m - np.dot(ak.transpose(),muik)
            '''print("both_ak_bk",ak,bk,muik,t1m)
            print(bk)
            print(w)'''
            coff1[i,j] = ak.transpose()
            coff2[i,j] = bk
    
    for i in range(w,ret.shape[0]+w):
        for j in range(w,ret.shape[1]+w):
            #print(coff1[i,j])
            z1 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,0])*p[i,j,0]
            z2 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,1])*p[i,j,1]
            z3 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,2])*p[i,j,2]
            z4 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,3])*q[i,j,0]
            z5 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,4])*q[i,j,1]
            z6 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,5])*q[i,j,2]
            z7 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,6])*p[i,j,0]*p[i,j,0]
            z8 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,7])*p[i,j,1]*p[i,j,0]
            z9 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,8])*p[i,j,2]*p[i,j,0]
            z10 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,9])*q[i,j,0]*p[i,j,0]
            z11 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,10])*q[i,j,1]*p[i,j,0]
            z12 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,11])*q[i,j,2]*p[i,j,0]
            z13 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,12])*p[i,j,1]*p[i,j,1]
            z14 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,13])*p[i,j,2]*p[i,j,1]
            z15 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,14])*q[i,j,0]*p[i,j,1]
            z16 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,15])*q[i,j,1]*p[i,j,1]
            z17 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,16])*q[i,j,2]*p[i,j,1]
            z18 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,17])*p[i,j,2]*p[i,j,2]
            z19 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,18])*q[i,j,0]*p[i,j,2]
            z20 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,19])*q[i,j,1]*p[i,j,2]
            z21 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,20])*q[i,j,2]*p[i,j,2]
            z22 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,21])*q[i,j,0]*q[i,j,0]
            z23 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,22])*q[i,j,1]*q[i,j,0]
            z24 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,23])*q[i,j,2]*q[i,j,0]
            z25 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,24])*q[i,j,1]*q[i,j,1]
            z26 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,25])*q[i,j,2]*q[i,j,1]
            z27 = np.mean(coff1[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,26])*q[i,j,2]*q[i,j,2]
            z28 = np.mean(coff2[max(i-w,w):min(i+w+1,ret.shape[0]+w),max(j-w,w):min(j+w+1,ret.shape[1]+w),:,0])
            x = z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8 + z9 + z10 + z11 + z12 + z13 + z14 + z15 + z16 + z17 + z18 + z19 + z20 + z21 + z22 + z23 + z24 + z25 + z26 + z27 + z28
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
w = [35,40]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("dragon_guided_extended" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

'''img = cv2.imread("cat_ao_1.png").astype(np.float32)/255
exp = cv2.imread("cat_ao_128.png").astype(np.float32)
g1 = cv2.imread("cat_normal.png").astype(np.float32)/255
g2 = cv2.imread("cat_position.png").astype(np.float32)/255

eps = [0.000001]
w = [10,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("cat_guided_extended" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("fireplace_ao_1.png").astype(np.float32)/255
exp = cv2.imread("fireplace_ao_128.png").astype(np.float32)
g1 = cv2.imread("fireplace_normal.png").astype(np.float32)/255
g2 = cv2.imread("fireplace_position.png").astype(np.float32)/255

eps = [0.000001]
w = [20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("fireplace_guided_extended" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("house_ao_1.png").astype(np.float32)/255
exp = cv2.imread("house_ao_128.png").astype(np.float32)
g1 = cv2.imread("house_normal.png").astype(np.float32)/255
g2 = cv2.imread("house_position.png").astype(np.float32)/255

eps = [0.000001]
w = []
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("house_guided_extended" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))

img = cv2.imread("sponza_ao_1.png").astype(np.float32)/255
exp = cv2.imread("sponza_ao_128.png").astype(np.float32)
g1 = cv2.imread("sponza_normal.png").astype(np.float32)/255
g2 = cv2.imread("sponza_position.png").astype(np.float32)/255

eps = [0.000001]
w = [10,20,25]
for e in eps:
    for W in w:
        p = filter(img,g1,g2,e,W)
        cv2.imwrite("sponza_guided_extended" + str(e) + "_" + str(W) + ".png",p)
        print(np.max(p),np.min(p))
        print("Error:",e,W,np.mean(np.absolute(p-exp)))'''