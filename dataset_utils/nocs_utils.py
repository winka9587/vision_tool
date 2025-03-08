import numpy as np
import cv2

def find_bounding_box_2d(mask):
    """
    Find the bounding box from a 2D mask image.

    Parameters:
    mask (numpy.array): A (h, w) mask image where points with value 1 are of interest.

    Returns:
    tuple: A tuple (y1, x1, y2, x2) representing the bounding box.
    """

    mask = mask[:, :, 2]

    # Identifying all points where the value is 1
    points = np.where(mask == 1)

    # If no points found, return None or an appropriate response
    if len(points[0]) == 0 or len(points[1]) == 0:
        return None

    # Find the bounding box coordinates
    y1, x1 = np.min(points[0]), np.min(points[1])
    y2, x2 = np.max(points[0]), np.max(points[1])

    return y1, x1, y2, x2

def draw_and_show_bbox(mask, bbox):
    """
    Draw a bounding box on the mask and display it.

    Parameters:
    mask (numpy.array): A (h, w) mask image.
    bbox (tuple): A tuple (y1, x1, y2, x2) representing the bounding box.
    """
    # Unpack the bounding box coordinates
    y1, x1, y2, x2 = bbox

    # Draw the rectangle on the mask
    # Note: OpenCV's rectangle function takes top-left and bottom-right coordinates
    cv2.rectangle(mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the mask with the bounding box
    cv2.imshow("Mask with Bounding Box", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()