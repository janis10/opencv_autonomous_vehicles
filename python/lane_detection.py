import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def selectWhiteYellow(image, space = 'RGB'): 
    """
    Filters and keeps yellow and white colors in the selected color space.
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
        ValueError()
    # Get masks in given space
    white_mask = cv2.inRange(imageInSpace, lowerW, upperW) 
    yellow_mask = cv2.inRange(imageInSpace, lowerY, upperY)
    # Apply the masks on the original RGB image
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked

def getRegionOfInterest(image):
    """
    Returns ROI (proportional to image size) of road.
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


def line2points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return (x1, y1, x2, y2)

def average_slope_intercept(image, lines):
    left_lines    = [] # (slope, intercept)
    left_weights  = [] # (length,)
    right_lines   = [] # (slope, intercept)
    right_weights = [] # (length,)
    
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
    
    # add more weight to longer lines    
    left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
    right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    # left_lane, right_lane = (slope, intercept), (slope, intercept)

    y1 = image.shape[0] # bottom of the image
    y2 = y1*0.6         # slightly lower than the middle

    left_lane  = line2points(y1, y2, left_lane)
    right_lane = line2points(y1, y2, right_lane)
    return left_lane, right_lane
    

# Fitting the coordinates into our actual image and then returning the image with the detected line(road with the detected lane):
def plotDetectedEgoLane(image, leftLane, rightLane):
    """
    Plot the detected ego lane
    """
    line_image = image.copy()
    if leftLane is not None:
        x1, y1, x2, y2 = leftLane
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    if rightLane is not None:
        x3, y3, x4, y4 = rightLane
        cv2.line(line_image, (x3, y3), (x4, y4), (255, 0, 0), 10)
    cv2.addWeighted(line_image, 1.0, line_image, 0.95, 0.0)
    if leftLane is not None and rightLane is not None:
        pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int32)
        # Create a copy of the original
        overlay = line_image.copy()
        # Draw polygon based on pts
        cv2.fillConvexPoly(overlay, pts, (0, 255, 0))
        # Combine with original
        opacity = 0.4
        cv2.addWeighted(line_image, opacity, overlay, 1-opacity, 0, line_image)
    return line_image

# Remove 'almost' horizontal lines
def filterOutHorizontal(lines):
    filteredLines = []
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # Calculating equation of the line: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m * x2
        # theta will contain values in [-90,+90]
        theta = math.degrees(math.atan(m))
        # Remove lines of slope near to 0 degree or 90 degree and storing others
        REJECT_DEGREE_TH = 4.0
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            # l = math.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )    # length of the line
            # filteredLines.append([x1, y1, x2, y2, m, c, l])
            filteredLines.append(line)
    # Keep the longest N, lines to increase speed. 
    # N = 15
    # if len(filteredLines) > N:
    #     filteredLines = sorted(filteredLines, key=lambda x: x[-1], reverse=True)
    #     filteredLines = filteredLines[:N]
    return filteredLines

# Split detected lines into left and right 'lanes', and return median. 
def getMedianLaneParams(lines):
    rho_left = []
    theta_left = []
    rho_right = []
    theta_right = []
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                # left lanes
                if theta < np.pi/2 and theta > np.pi/4:
                    rho_left.append(rho)
                    theta_left.append(theta)
                # right lanes
                if theta > np.pi/2 and theta < 3*np.pi/4:
                    rho_right.append(rho)
                    theta_right.append(theta)
    # Get medians
    left_rho = np.median(rho_left)
    left_theta = np.median(theta_left)
    right_rho = np.median(rho_right)
    right_theta = np.median(theta_right)
    return left_rho, left_theta, right_rho, right_theta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input_file)
    while(cap.isOpened()):
        _, frame = cap.read()

        if frame is not None:
            # 1. Select white and yellow pixels (lanes)
            maskedWhiteYellow = selectWhiteYellow(frame, space = 'HSL')
            # 2. Convert to grayscale
            grayscaled = cv2.cvtColor(maskedWhiteYellow, cv2.COLOR_RGB2GRAY)
            # 3. Gaussian blurring
            kernelSize = 15 # must be postivie and odd
            blurred = cv2.GaussianBlur(grayscaled, (kernelSize, kernelSize), 0)
            # 4. Canny edge detector
            upper_thresh = 150
            lower_thresh = 50
            canny = cv2.Canny(blurred, lower_thresh, upper_thresh)
            # 5. Select ROI
            roiImage = getRegionOfInterest(canny)
            # 6. Detect lines
            lines = cv2.HoughLinesP(roiImage, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
            # overlaidImage = drawLines(image, lines)
            # 7. Compute ego lane
            leftLane, rightLane = average_slope_intercept(frame, lines)
            overlaidImage = plotDetectedEgoLane(frame, leftLane, rightLane)
            # Plot
            cv2.imshow("Processed", overlaidImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):	
            break

    # close the video file
    cap.release()
    cv2.destroyAllWindows()

