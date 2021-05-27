import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def getRMSE(p1,p2):
    a = cv2.imread(p1).astype(np.float32)
    b = cv2.imread(p2).astype(np.float32)
    print(np.sqrt(np.mean(np.absolute(a-b))))

def getSSIM(p1,p2):
    a = cv2.imread(p1,0).astype(np.float32)
    b = cv2.imread(p2,0).astype(np.float32)
    (score, diff) = ssim(a, b, full=True)
    print("SSIM: {}".format(score))

getRMSE("sponza35_14get.png","sponza_ao_128.png")
getSSIM("sponza35_14get.png","sponza_ao_128.png")


getRMSE("dragon_guided_extended1e-06_20.png","dragon_ao_128.png")
getSSIM("dragon_guided_extended1e-06_20.png","dragon_ao_128.png")

getRMSE("dragon_guided_1e-06_20.png","dragon_ao_128.png")
getSSIM("dragon_guided_1e-06_20.png","dragon_ao_128.png")

getRMSE("cat_guided_extended1e-06_15.png","cat_ao_128.png")
getSSIM("cat_guided_extended1e-06_15.png","cat_ao_128.png")

getRMSE("cat_guided_1e-06_15.png","cat_ao_128.png")
getSSIM("cat_guided_1e-06_15.png","cat_ao_128.png")

getRMSE("fireplace_guided_extended1e-06_15.png","fireplace_ao_128.png")
getSSIM("fireplace_guided_extended1e-06_15.png","fireplace_ao_128.png")

getRMSE("fireplace_guided_1e-06_15.png","fireplace_ao_128.png")
getSSIM("fireplace_guided_1e-06_15.png","fireplace_ao_128.png")

getRMSE("sponza_guided_extended1e-06_15.png","sponza_ao_128.png")
getSSIM("sponza_guided_extended1e-06_15.png","sponza_ao_128.png")

getRMSE("sponza_guided_1e-06_15.png","sponza_ao_128.png")
getSSIM("sponza_guided_1e-06_15.png","sponza_ao_128.png")