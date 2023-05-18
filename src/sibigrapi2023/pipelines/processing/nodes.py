"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.7
"""

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

# TODO: reference
# https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def _order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl])

def homography(images):
    # process all images
    homography_results = dict()
    for file_name, loader in images.items():
        image = np.array(loader())
        # edge detection
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # bounding convex polygon
        cnt = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        hull = np.intp(cv2.convexHull(np.concatenate([cnt[0], cnt[1], cnt[2], cnt[3]], axis=0)))
        # bounding rect with perspective
        epsilon = 0.1 * cv2.arcLength(hull, True)
        points = cv2.approxPolyDP(hull, epsilon, True)
        if len(points) != 4:
            continue
        points = _order_points(points.reshape(-1).reshape((-1, 2)))
        homography_results[file_name] = {
            'P1X': points[0][0], 'P1Y': points[0][1],
            'P2X': points[1][0], 'P2Y': points[1][1],
            'P3X': points[2][0], 'P3Y': points[2][1],
            'P4X': points[3][0], 'P4Y': points[3][1]
        }
    return pd.DataFrame(homography_results).T.reset_index()

