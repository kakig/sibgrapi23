"""
This is a boilerplate pipeline 'metric_generation'
generated using Kedro 0.18.7
"""

import gc
import time

import logging

log = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import cv2

import torch
import torchvision

from ISR.models import RDN

import pytesseract
import imutils

from skimage.restoration import estimate_sigma

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def _clean_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

#sam_checkpoint = "sam_vit_h_4b8949.pth"
#model_type = "vit_h"
#device = "cuda"
#
#sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device=device)
#
#predictor = SamPredictor(sam)

def _biggest_contour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 58000:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.015 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest

def _sharpnessTenengrad(img):
    # Calculate the gradient of the image using the Sobel operator
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = np.sqrt(sobelx**2 + sobely**2)

    # Calculate the variance of the gradient image
    return grad.var()


def _estimateNoise(img):
    return estimate_sigma(img, average_sigmas=True)

def _order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = np.sum(pts,axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[3] = pts[np.argmin(diff)]
        rect[1] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect

def _getAngle(img):
    image = imutils.resize(img, height = 500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 50, 150)
    
  
    cnts = None
    cnts, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

    
    screenCnt = None
    screenCnt = _biggest_contour(cnts)
    
    if(len(screenCnt)==0 ):
        edged = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,13,0.5)
        cnts = None
        cnts, hierarchy  = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        
        screenCnt = None
        screenCnt = _biggest_contour(cnts)

    if(len(screenCnt) >0 ):
        cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
        screenCnt = _order_points(screenCnt[:,0,:])
        angle = cv2.minAreaRect(screenCnt)[-1]
        angle = 90 - angle if (angle>45) else angle        
        
        return angle
    else:
        return None

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

    imgBlurLevel = _sharpnessTenengrad(img)
    noiseLevel = _estimateNoise(img)
    imgAngle = _getAngle(img)

    #boas - aceitaveis
    if(imgBlurLevel >= 12111.474131839535 and noiseLevel<= 0.4516075407697839): #verifica ruido e nitidez
        if(imgAngle != None): # verifica se conseguiu calcular um angulo
            if(imgAngle <= 1.249346097310384):# se encaixa em todos os parametros classifica como bom
                imagesGood = np.append(imagesGood,i)
        else:
            imagesGood = np.append(imagesGood,i)


  for img in imagesList:
    if(img in imagesGood): continue

    imgBlurLevel = _sharpnessTenengrad(img)
    noiseLevel = _estimateNoise(img)
    imgAngle = _getAngle(img)

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

def _fileterImagePipeline(img):
    imgBlurLevel = _sharpnessTenengrad(img)
    noiseLevel = _estimateNoise(img)
    imgAngle = _getAngle(img)
    if (imgBlurLevel >= 12111.474131839535 and noiseLevel<= 0.4516075407697839): #verifica ruido e nitidez
        if(imgAngle != None): # verifica se conseguiu calcular um angulo
            if(imgAngle <= 1.249346097310384):# se encaixa em todos os parametros classifica como bom
                return pytesseract.image_to_string(img)
        else:
            return pytesseract.image_to_string(img)
    else:
        return ""

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

def generate_methods_ocr_text(images):
    all_results = dict()
    pipeline_functions = {
        "naive": lambda x: pytesseract.image_to_string(x),
        "only_filtering": _fileterImagePipeline,
    }
    for pipeline_name, pipeline_func in pipeline_functions.items():
        results = list()
        for file_name, loader in images.items():
            image = np.array(loader())
            start = time.process_time()
            ocr_text = pipeline_func(image)
            pipeline_time = time.process_time() - start
            results.append({
                "file_name": file_name,
                "pipeline_time": pipeline_time,
                "ocr_text": ocr_text,
            })
        all_results[pipeline_name] = results
    return all_results

def generate_ablation_metrics(images):
    pass

def generate_methods_report(metrics):
    pass

def generate_ablation_report(metrics):
    pass
