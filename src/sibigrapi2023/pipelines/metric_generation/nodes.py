"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

import gc

import logging

log = logging.getLogger(__name__)

import numpy as np
from matplotlib import pyplot as plt
import cv2

import torch
import torchvision

from ISR.models import RDN

import pytesseract

from skimage.restoration import estimate_sigma

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def _clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


# pipeline de melhoramento
def _binarizeAdaptive(image, T, C, show=False):
    binarized = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, T, C)
    return binarized

def _sharp(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def _dilate(image):
        image = cv2.dilate(image, np.zeros((5, 5), 'uint8'), iterations=4)

def _medianBlur(image, filter_size):
    return cv2.medianBlur(image, filter_size)

def _ISR(image):
    lr_img = np.array(image)
    rdn = RDN(weights='psnr-small')
    sr_img = rdn.predict(lr_img)
    return sr_img

# filtragem
def _filterImages(imagesList):

  imagesGood = []
  imagesMedium = []
  imagesBad = []

  for img in imagesList:

    imgBlurLevel = sharpnessTenengrad(img)
    noiseLevel = estimateNoise(img)
    imgAngle = getAngle(img)

    #boas - aceitaveis
    if(imgBlurLevel >= 12111.474131839535 and noiseLevel<= 0.4516075407697839): #verifica ruido e nitidez
        if(imgAngle != None): # verifica se conseguiu calcular um angulo
            if(imgAngle <= 1.249346097310384):# se encaixa em todos os parametros classifica como bom
                imagesGood = np.append(imagesGood,i)
        else:
            imagesGood = np.append(imagesGood,i)


  for img in imagesList:
    if(img in imagesGood): continue

    imgBlurLevel = sharpnessTenengrad(img)
    noiseLevel = estimateNoise(img)
    imgAngle = getAngle(img)

    #descartadas 
    if(imgAngle != None):
        if(noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798 or imgAngle >= 8.065917314340671):
            imagesBad = np.append(imagesBad,i)
        else:
            imagesMedium = np.append(imagesMedium,i)
                
    else:
        if((noiseLevel >= 5.688634446534304 or imgBlurLevel <= 5153.859697022798)):
            imagesBad = np.append(imagesBad,i)
        else:
            imagesMedium = np.append(imagesMedium,i)

  return (imagesGood, imagesMedium, imagesBad)

def _unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, M

def _findDestinationPoints(src, shape):
    w_list = [
        abs(src[0][0] - src[1][0]),
        abs(src[2][0] - src[3][0])
    ]
    h_list = [
        abs(src[0][1] - src[2][1]),
        abs(src[1][1] - src[3][1])
    ]
    max_w = max(w_list) # maior lado horizontal do recibo
    max_h = max(h_list) # maior lado vertical do recibo

    h, w = shape[0], shape[1]
    inc_w = (w - max_w) / 2
    inc_h = (h - max_h) / 2

    return np.float32([(inc_w, inc_h),
                    (inc_w + max_w, inc_h),
                    (inc_w, inc_h + max_h),
                    (inc_w + max_w, inc_h + max_h)])

def _homography(image, points):
    src = np.array(points, dtype="float32")
    dst = _findDestinationPoints(src, image.shape)   
    warped_img, M = unwarp(image, src, dst)
    return warped_img, dst

def generate_method_metrics(images):
    pass

def generate_ablation_metrics(images):
    pass

def generate_methods_report(metrics):
    pass

def generate_ablation_report(metrics):
    pass
