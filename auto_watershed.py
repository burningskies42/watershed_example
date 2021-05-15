import argparse

import cv2
import numpy as np


def mark_objects(img_path):
    """Use the watershed algo to automatically detect contours in an image

    Args:
        img_path (str): location of the target file
    """
    # load image
    img = cv2.imread(img_path)

    # apply strong blur to image
    blurred_img = cv2.medianBlur(img, 35)

    # convert to grayscale
    gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    # apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # apply Watershed algorithm
    markers = cv2.watershed(img, markers)
    contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # For every entry in contours
    for i in range(len(contours)):

        # last column in the array is -1 if an external contour (no contours inside of it)
        if hierarchy[0][i][3] == -1:

            # We can now draw the external contours from the list of contours
            cv2.drawContours(img, contours, i, (255, 0, 0), 10)

    cv2.imwrite("result.jpg", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", type=str, help="Path to image file", default="pennies.jpg", required=False)
    args = parser.parse_args()

    mark_objects(args.file_path)
