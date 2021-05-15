#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
from matplotlib import cm


def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3]) * 255)


# callback function
def mouse_callback(event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passed to the watershed
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)

        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)
        marks_updated = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, help="Path to image file", default="road_image.jpg", required=False)
    args = parser.parse_args()

    # load image
    img_path = args.file_path
    road = cv2.imread(img_path)

    road_copy = np.copy(road)
    marker_image = np.zeros(road.shape[:2], dtype=np.int32)
    segments = np.zeros(road.shape, dtype=np.uint8)

    colors = []
    for i in range(10):
        colors.append(create_rgb(i))

    # global
    current_marker = 1
    marks_updated = False
    n_markers = 10

    cv2.namedWindow("Road Image")
    cv2.setMouseCallback("Road Image", mouse_callback)

    while True:
        cv2.imshow("Watershed Segments", segments)
        cv2.imshow("Road Image", road_copy)

        # close
        k = cv2.waitKey(1)
        if k == 27:
            break

        # clearing all the colors press c key
        elif k == ord("c"):
            road_copy = road.copy()
            marker_image = np.zeros(road.shape[0:2], dtype=np.int32)
            segments = np.zeros(road.shape, dtype=np.uint8)

        # update color choice
        elif k > 0 and chr(k).isdigit():
            current_marker = int(chr(k))

        if marks_updated:
            marker_image_copy = marker_image.copy()
            cv2.watershed(road, marker_image_copy)

            segments = np.zeros(road.shape, dtype=np.uint8)

            for color_ind in range(n_markers):
                segments[marker_image_copy == color_ind] = colors[color_ind]

        # update markings

    cv2.destroyAllWindows()
