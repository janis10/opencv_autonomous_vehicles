from typing import Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def selectWhiteYellow(image: np.ndarray, space: Optional[str] = 'RGB') -> np.ndarray: 
    """Filters and keeps yellow and white colors in the selected color space.

    Args:
      image: An image as a np.ndarray. 
      space: The space of the image's channels: RGB or HSL. 

    Returns:
      The thresholded image such that white and yellow colors are selected. 
    """
    if space == 'RGB':
        imageInSpace = image.copy()
        # white color mask
        lowerW = np.uint8([200, 200, 200])
        upperW = np.uint8([255, 255, 255])
        # yellow color mask
        lowerY = np.uint8([190, 190,   0])
        upperY = np.uint8([255, 255, 255])
    elif space == 'HSL':
        imageInSpace = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # white color mask
        lowerW = np.uint8([  0, 200,   0])
        upperW = np.uint8([255, 255, 255])
        # yellow color mask
        lowerY = np.uint8([ 10,   0, 100])
        upperY = np.uint8([ 40, 255, 255])
    else:
        raise ValueError(f'Space can be either "RGB" or "HSL".')
    # Get masks in given space
    white_mask = cv2.inRange(imageInSpace, lowerW, upperW) 
    yellow_mask = cv2.inRange(imageInSpace, lowerY, upperY)
    # Apply the masks on the original RGB image
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def getRegionOfInterest(image: np.ndarray) -> np.ndarray: 
    """Masks a ROI (proportional to image size) of road.

    Args:
      image: An image as a np.ndarray. 

    Returns:
      The masked image such that areas outside of the road are zeroed out.
    """
    height = image.shape[0]
    width = image.shape[1]
    BL = [width*0.1, height*0.95]
    TL = [width*0.4, height*0.6]
    BR = [width*0.9, height*0.95]
    TR = [width*0.6, height*0.6] 
    polygon = np.array([[BL, TL, TR, BR]], dtype=np.int32)

    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, polygon, 255)
    # in case, the input image has a channel dimension (not grayscale)
    else:
        cv2.fillPoly(mask, polygon, (255,) * mask.shape[2])
    return cv2.bitwise_and(image, mask)

########################################################################################################################

def line2points(y1: int, y2: int, line: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Convert a line represented in slope and intercept into pixel points. 

    Args:
      y1: Partial pixel coordinate for point 1.
      y2: Partial pixel coordinate for point 2.
      line: Line representation as a Tuple (slope, intercept).

    Returns:
      Line representation as two points (x1, y1, x2, y2).
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return (x1, y1, x2, y2)

def getEgoLane(image: np.ndarray, lines: list, method: Optional[str] = 'average') -> Tuple[Tuple[int,int,int,int], Tuple[int,int,int,int]]:
    """Averages out left and right lines, and returns left and right lane. 

    Args:
      image: Original image. 
      lines: List of detected lines.
      method: average or median.

    Returns:
      Returns left and right lines using 'method'.
    """
    if not (method == 'average' or method == 'median'):
        raise ValueError(f'Variable "method" must be either "average" or "median"')

    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
    # To average out / get median of lines, we need to convert a line from
    # two-points representation to (slope, intercept) representation.
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2==x1:
                continue # ignore a vertical line
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2+(x2-x1)**2)
            if slope < 0: # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
    
    if method == 'average':
        # add more weight to longer lines
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    elif method == 'median':
        left_slope = []
        left_intercept = []
        for slope, intercept in left_lines:
            left_slope.append(slope)
            left_intercept.append(intercept)
        right_slope = []
        right_intercept = []
        for slope, intercept in right_lines:
            right_slope.append(slope)
            right_intercept.append(intercept)
        left_lane = (np.median(left_slope), np.median(left_intercept))
        right_lane = (np.median(right_slope), np.median(right_intercept)) 

    # And now convert back to two-points representation
    y1 = image.shape[0] # bottom of the image
    y2 = y1 * 0.6         # slightly lower than the middle
    left_lane  = line2points(y1, y2, left_lane)
    right_lane = line2points(y1, y2, right_lane)

    return left_lane, right_lane

########################################################################################################################

def filterOutHorizontal(lines: list) -> list:
    """Removes detected lines that are close to being horizontal.

    Args:
      lines: List of detected lines.

    Returns:
      Refined list without lines that are very close to horizontal. 
    """
    filteredLines = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Calculating equation of the line: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        # theta will contain values in [-90,+90]
        theta = math.degrees(math.atan(m))
        # Remove lines of slope near to 0 degree or 90 degree and storing others
        REJECT_DEGREE_TH = 4.0
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            filteredLines.append(line)
    return filteredLines

########################################################################################################################

def drawLines(image: np.ndarray, lines: list) -> np.ndarray:
    """Draws all detected lines in an image.

    Args:
      image: Original image. 
      lines: List of detected lines.

    Returns:
      Image with plotted all detected lines.
    """
    overlaidImage = image.copy()
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(overlaidImage, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return overlaidImage

def drawEgoLane(image: np.ndarray, left_lane: Tuple[int,int,int,int], right_lane: Tuple[int,int,int,int]) -> np.ndarray:
    """Plot the detected ego lane.

    Args:
      image: Original image. 
      leftLane: Detected left lane. Line representation as two points (x1, y1, x2, y2).
      rightLane: Detected right lane. Line representation as two points (x1, y1, x2, y2).

    Returns:
      Image with plotted left and right lanes, and space in between them. 
    """
    line_image = image.copy()
    if left_lane is not None:
        x1, y1, x2, y2 = left_lane
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    if right_lane is not None:
        x3, y3, x4, y4 = right_lane
        cv2.line(line_image, (x3, y3), (x4, y4), (255, 0, 0), 10)
    cv2.addWeighted(line_image, 1.0, line_image, 0.95, 0.0)
    if left_lane is not None and right_lane is not None:
        # Create a copy of the original
        overlay = line_image.copy()
        # Draw trapezoid
        trapezoid = np.asarray([(x1, y1), (x2, y2), (x4, y4), (x3, y3)])
        trapezoid = np.expand_dims(trapezoid, 1).astype(np.int32)
        overlay = cv2.fillPoly(overlay,[trapezoid], color=(0, 255, 0))
        # Combine with original
        opacity = 0.7
        cv2.addWeighted(line_image, opacity, overlay, 1-opacity, 0, line_image)
    return line_image

########################################################################################################################

def getBEV(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """Returns BEV of image

    Args:
      image: An image as a np.ndarray. 

    Returns:
      The Homography transformation matrix H, the BEV image, and a concatenated [camera view, bev] image. 
    """
    height, width = image.shape[0:2]
    # Choose trapezoid vertices in camera view
    y0, y1 = int(height * 0.6), height - 50
    x_off = 50  # 280
    p1 = (0, y1)
    p2 = (int(width/2)-x_off, y0)
    p3 = (int(width/2)+x_off, y0)
    p4 = (width, y1)

    trapezoid = np.asarray([p1, p2, p3, p4])
    trapezoid = np.expand_dims(trapezoid, 1).astype(np.int32)
    trapezoidImage = np.zeros_like(image)
    trapezoidImage = cv2.fillPoly(trapezoidImage,[trapezoid], color=(0, 0, 255))
    # Camera view with trapezoid
    opacity = 0.4
    cv2.addWeighted(trapezoidImage, opacity, image, 1-opacity, 0, trapezoidImage)

    # Normally here we need some camera info such as:
    # * pixels_per_meter and vice versa
    # * info about the trapezoid's real word dimensions
    # This info we can get from camera parameters.
    ab_meters = 50
    ad_meters = 10

    pixels_per_meter = 40
    widthBEV = ad_meters * pixels_per_meter
    heightBEV = ab_meters * pixels_per_meter
    xoff = 500
    # dst_points are the transformed vertices of the trapezoid in the BEV space, 
    # where it is a parallelogram
    dst_points = np.asarray([(xoff + 0, heightBEV), (xoff + 0, 0), (xoff + widthBEV, 0), (xoff + widthBEV, heightBEV)])
    widthBEV += (2 * xoff)
    # cv2.findHomography returns the transformation 
    # that maps the points in 'trapezoid' to the points in 'dst_points'
    H, _ = cv2.findHomography(trapezoid, dst_points)

    # Finally, we can warp the full image to BEV using H
    imageBEV = cv2.warpPerspective(image, H, (widthBEV, heightBEV))

    # Now let's put image and imageBEV side by side for show
    ratio = trapezoidImage.shape[0] / imageBEV.shape[0]
    imageBEV_scaled = cv2.resize(imageBEV, dsize=None, fx=ratio, fy=ratio)
    side_by_side = np.concatenate([trapezoidImage, imageBEV_scaled], axis=1)

    return H, imageBEV, side_by_side