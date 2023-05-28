"""
This is a boilerplate pipeline 'processing'
generated using Kedro 0.18.7
"""

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from sklearn.cluster import KMeans
from itertools import combinations
from math import atan
import PIL

# TODO: reference
# https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
# https://medium.com/intelligentmachines/document-detection-in-python-2f9ffd26bf65
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

def _get_angle_between_lines(line_1, line_2):
    rho1, theta1 = line_1
    rho2, theta2 = line_2
    # x * cos(theta) + y * sin(theta) = rho
    # y * sin(theta) = x * (- cos(theta)) + rho
    # y = x * (-cos(theta) / sin(theta)) + rho
    m1 = -(np.cos(theta1) / np.sin(theta1))
    m2 = -(np.cos(theta2) / np.sin(theta2))
    return abs(atan(abs(m2-m1) / (1 + m2 * m1))) * (180 / np.pi)

def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2

    A = np.array([
      [np.cos(theta1), np.sin(theta1)],
      [np.cos(theta2), np.sin(theta2)]
    ])

    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]

def homography(images):
    # process all images
    homography_results = dict()
    image_results = dict()
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

        image_results[file_name] = PIL.Image.fromarray(cv2.drawContours(image, [points], 0, (0, 0, 255), 2).astype('uint8'))

    return [pd.DataFrame(homography_results).T.reset_index(), image_results]

def homography_segmentation_batched(mask_archives, images):
    homography_results = dict()
    image_results = dict()

    for file_name, loader in mask_archives.items():
        archive = loader()
        original_image = np.array(images[file_name.replace("batched_", "").replace(".npz", "") + "-receipt.jpg"]())
        border_size = 10
        area_to_image_ratios = list()
        homography_points = list()
        homography_points_hough = list()
        hulls = list()
        ious = list()
        for mask in archive['masks'][0]:
            image_area = mask.shape[0] * mask.shape[1]
            gray = mask.astype(np.uint8) * 255
            edged = cv2.Canny(gray, 75, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnt = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
            if len(cnt) < 1:
                area_to_image_ratios.append(0)
                homography_points.append(np.array([]))
                continue
            chosen_contours = cnt[0]
            # select the 4 biggest contours if available
            for i in range(1, min(len(cnt), 4)):
                chosen_contours = np.concatenate([chosen_contours, cnt[i]], axis=0)
            hull = np.intp(cv2.convexHull(chosen_contours))
            contour_area = cv2.contourArea(hull)
            area_ratio = contour_area / image_area
            area_to_image_ratios.append(area_ratio)

            # Hough transform
            img_canny = cv2.Canny(cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, 0), 75, 200).astype(np.uint8)
            lines = cv2.HoughLines(img_canny, 2, np.pi / 180, 150)
            img_hough = cv2.copyMakeBorder(original_image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, 0)
            if lines is not None:
                # limit to 40 lines
                lines = lines[:40]
                for rho, theta in lines[:, 0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(img_hough, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
            intersections = []
            if lines is None:
                lines = []
            group_lines = combinations(range(len(lines)), 2)
            x_in_range = lambda x: 0 <= x <= img_hough.shape[1]
            y_in_range = lambda y: 0 <= y <= img_hough.shape[0]
            for i, j in group_lines:
              line_i, line_j = lines[i][0], lines[j][0]
              if 80.0 < _get_angle_between_lines(line_i, line_j) < 100.0:
                  int_point = _intersection(line_i, line_j)
                  if x_in_range(int_point[0][0]) and y_in_range(int_point[0][1]): 
                      intersections.append(int_point)
            if len(intersections) > 0:
                X = np.array(intersections)[:, 0]
            else:
                X = np.array([])
            for point in X:
                cv2.circle(img_hough, point, 3, (255, 0, 0), 3).astype('uint8')
            model = KMeans(n_clusters=4, n_init='auto', random_state=42)
            points_hough = np.array([])
            if X.shape[0] >= 4:
                model.fit(X)
                # convert coordinates back to original image space
                points_hough = np.rint(model.cluster_centers_).astype(np.int32)
                for point in points_hough:
                    cv2.circle(img_hough, point, 5, (255, 0, 255), 3).astype('uint8')
                points_hough = np.clip(points_hough - border_size, 0, max(*original_image.shape))
            else:
                points_hough = X
            homography_points_hough.append(points_hough)

            epsilon = 0.1 * cv2.arcLength(hull, True)
            points = cv2.approxPolyDP(hull, epsilon, True)
            homography_points.append(points)

        iou_predictions = archive['iou_predictions'][0]

        candidate_indexes = set([0, 1, 2])
        excluded_indexes = set()
        best_iou = 0
        final_index = -1
        for index in candidate_indexes:
            if area_to_image_ratios[index] < 0.35:
                excluded_indexes.add(index)
                continue
            if len(homography_points[index]) != 4:
                excluded_indexes.add(index)
                continue
            if iou_predictions[index] > best_iou:
                final_index = index
                best_iou = iou_predictions[index]
            elif final_index == -1:
                final_index = index

        if len(excluded_indexes) == 3:
            continue

        print(file_name, final_index, homography_points, iou_predictions)
        points = _order_points(homography_points[final_index].reshape(-1).reshape((-1, 2)))
        homography_results[file_name] = {
            'P1X': points[0][0], 'P1Y': points[0][1],
            'P2X': points[1][0], 'P2Y': points[1][1],
            'P3X': points[2][0], 'P3Y': points[2][1],
            'P4X': points[3][0], 'P4Y': points[3][1]
        }

        image_boxes = PIL.Image.fromarray(cv2.drawContours(original_image.copy(), [points], 0, (0, 0, 255), 2).astype("uint8"))
        image_results[file_name + ".jpg"] = image_boxes
        image_results[file_name + "_hough" + ".jpg"] = PIL.Image.fromarray(img_hough.astype("uint8"))

    return [pd.DataFrame(homography_results).T.reset_index(), image_results]

def homography_segmentation(masks, images):
    homography_results = dict()
    image_results = dict()

    for file_name, loader in masks.items():
        archive = loader()
        m = archive['masks'][np.argmax(archive['scores'])]
        gray = m.astype(np.uint8) * 255
        edged = cv2.Canny(gray, 75, 200)
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)
        chosen_contours = cnt[0]
        for i in range(1, min(len(cnt), 4)):
            chosen_contours = np.concatenate([chosen_contours, cnt[i]], axis=0)
        hull = np.intp(cv2.convexHull(chosen_contours))
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

        image_results[file_name + ".jpg"] = PIL.Image.fromarray(cv2.drawContours(np.array(images[file_name + "-receipt.jpg"]()), [points], 0, (0, 0, 255), 2).astype('uint8'))

    return [pd.DataFrame(homography_results).T.reset_index(), image_results]
