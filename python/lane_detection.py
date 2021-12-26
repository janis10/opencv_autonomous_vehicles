import cv2
import numpy as np
import argparse
from cv_utils import selectWhiteYellow, getRegionOfInterest, getEgoLane, drawEgoLane

"""Lane detection module.

Classic lane detection:
1. Color selection (yellow, white for lanes)
2. Convert to grayscale
3. Gaussian blurring
4. Canny edge detection
5. Region of interest selection
6. Hough lines transformation to detect lines
7. Compute ego lane from detected lines
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input_file)
    while(cap.isOpened()):
        _, frame = cap.read()

        if frame is not None:
            # 0. OpenCV reads in BGR format, we convert to RBG. 
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 1. Select white and yellow pixels (lanes)
            maskedWhiteYellow = selectWhiteYellow(frameRGB, space = 'HSL')
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
            leftLane, rightLane = getEgoLane(frameRGB, lines, method='average')
            overlaidImage = drawEgoLane(frameRGB, leftLane, rightLane)
            # Plot
            overlaidImage = cv2.cvtColor(overlaidImage, cv2.COLOR_RGB2BGR)
            cv2.imshow("Processed", overlaidImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):	
            break

    # close the video file
    cap.release()
    cv2.destroyAllWindows()

