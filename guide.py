import numpy as np
import sys
sys.path.append('/Users/rahul/Library/Python/3.7/lib/python/site-packages')
import cv2
import math

def get_guassian(x,sigma):
    return (1/(sigma*(6.28)))*math.exp(-((1/2)*((x)/sigma)**2))

def bilateral(input1,window_size,bound_to_a_k,guidance):
    a_k = np.zeros([len(input1),len(input1[0])])
    b_k = np.zeros([len(input1),len(input1[0])])
    for i in range(len(input1)):
        print(i)
        for j in range(len(input1[0])):
            mean_guidance = 0
            variance_guidance = 0
            mean_input = 0
            count = 0
            for k in range(window_size):
                for l in range(window_size):
                    if((i + k) < len(input1) and (j + l) < len(input1[0])):
                        mean_guidance = mean_guidance + guidance[(i + k)][(j + l)]
                        mean_input = mean_input + input1[(i + k)][(j + l)]
                        count = count + 1
            mean_guidance = mean_guidance/count
            mean_input = mean_input/count
            for k in range(window_size):
                for l in range(window_size):
                    if((i + k) < len(input1) and (j + l) < len(input1[0])):
                        variance_guidance = variance_guidance + (guidance[(i + k)][(j + l)]-mean_guidance)**2
            variance_guidance = variance_guidance/count
            temp = 0
            for k in range(window_size):
                for l in range(window_size):
                    if((i + k) < len(input1) and (j + l) < len(input1[0])):
                        temp += guidance[(i + k)][(j + l)]*input1[(i + k)][(j + l)] - mean_guidance*mean_input
            temp = (temp/(variance_guidance+bound_to_a_k))/count
            a_k[i][j] = temp
            b_k[i][j] = mean_input - temp*mean_guidance
    a_k_updated = np.zeros([len(input1),len(input1[0])])
    b_k_updated = np.zeros([len(input1),len(input1[0])])
    for i in range(len(a_k)):
        for j in range(len(a_k[0])):
            count2 = 0
            for k in range(window_size):
                for l in range(window_size):
                    if((i - k) >= 0  and (j - l) >= 0 ):
                        count2 = count2 + 1
                        a_k_updated[i][j] = a_k_updated[i][j] + a_k[i-k][j-l]
                        b_k_updated[i][j] = b_k_updated[i][j] + b_k[i-k][j-l]
            a_k_updated[i][j] = a_k_updated[i][j]/count2
            b_k_updated[i][j] = b_k_updated[i][j]/count2
    ret = ((a_k_updated*guidance + b_k_updated)*255)
    return ret


img = cv2.imread("ao_1_dragon.png").astype(np.float32)
exp = cv2.imread("ao_128_dragon.png").astype(np.float32)
g1 = cv2.imread("normal_dragon.png").astype(np.float32)
g2 = cv2.imread("position_image_dragon.png").astype(np.float32)

p = bilateral(img,3,0.02,g2)
cv2.imwrite("0001112.png",p)