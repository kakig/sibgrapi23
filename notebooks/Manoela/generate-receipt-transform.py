import random
import cv2
import PIL

from PIL import Image
from skimage.util import random_noise
from skimage.transform import rotate

def showImage(img, title="", name="x-receipt"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, 900, 600)
    while(1):
        pressedKey = cv2.waitKey(0) & 0xFF
        if (pressedKey == ord('a')):
            cv2.destroyWindow(title)
            break
        if (pressedKey == ord('s')):
            cv2.imwrite('Dataset/processing/' + name + '.jpg', img)
            cv2.destroyWindow(title)
            break

def transform(img):
    img = cv2.pyrDown(img)
    # img = random_noise(img, mode='s&p')
    return rotate(img, random.randint(-30, 30), resize=True)

for i in range(1020, 1200):
    if i == 1158:
        continue
    name = str(i) + '-receipt'
    img = cv2.imread('Dataset/receipt/' + name + '.jpg')
    img = transform(img)
    #showImage(img, title='image')
    cv2.imwrite('Dataset/receipt-transform/' + name + '.jpg', img*255)
