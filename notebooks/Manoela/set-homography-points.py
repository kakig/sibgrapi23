import numpy as np
import cv2
import pytesseract
import PIL
import pandas as pd

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # cv2.circle(image,(x,y),3,(255,0,0),-1)
            self.points.append((x,y))

    def clear_points(self):
        self.points = []

def setHomographyPoints(image, name):
    coordinateStore = CoordinateStore()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    # cv2.namedWindow('image')
    cv2.setMouseCallback(name, coordinateStore.select_point)
    cv2.imshow(name, image)
    cv2.resizeWindow(name, 900, 600)
    
    while True:
        cv2.imshow(name, image)
        k = cv2.waitKey(0) & 0xFF # sleeps for 0 seconds
        if k == ord('a'):
            if len(coordinateStore.points) == 4:
                break
            else:
                print('Selected coordinates: ')
                for points in coordinateStore.points:
                    print(points)
                coordinateStore.clear_points()
     
    cv2.destroyAllWindows()
    return coordinateStore

def showImage(img, title="", name="x-receipt"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.resizeWindow(title, 900, 600)
    while(1):
        pressedKey = cv2.waitKey(0) & 0xFF
        if(pressedKey == ord('a')):
            cv2.destroyWindow(title)
            break
        if(pressedKey == ord('s')):
            cv2.imwrite('Dataset/processing/' + name + '.jpg', img)
            cv2.destroyWindow(title)
            break

def readCSV():
    print("Iniciando leitura do CSV")
    df = pd.read_csv("Dataset/homography-transform.csv", sep=";") # .rename(columns = {"theta":"col0"})
    return df

points = [[],[],[],[]]

df = readCSV()
image = df['Image'].tolist()
for i, column in enumerate(df.columns):
    if column == 'Image':
        continue
    for point in df[column]:
        t_point = (int(point.split(',')[0]), int(point.split(',')[1]))
        points[i-1].append(str(t_point[0]) + ',' + str(t_point[1]))

for i in range(1150, 1200):
    if i == 1158:
        continue
    name = str(i) + '-receipt'
    orig_image = cv2.imread('Dataset/receipt-transform/' + name + '.jpg')
    coordinateStore = setHomographyPoints(orig_image, name)
    image.append(str(i))
    for idx, point in enumerate(coordinateStore.points):
        points[idx].append(str(point[0]) + ',' + str(point[1]))

df_final = pd.DataFrame({'Image': image, 'P1': points[0], 'P2': points[1], 'P3': points[2], 'P4': points[3]})
df_final.to_csv('Dataset/homography-transform.csv', index=False, sep=';')
    
