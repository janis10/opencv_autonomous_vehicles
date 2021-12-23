import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

def getBEV(image): 
    """
    Returns BEV of image
    """
    height, width = image.shape[:-1]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.input_file)
    while(cap.isOpened()):
        _, frame = cap.read()

        if frame is not None:
            _, _, side_by_side = getBEV(frame)
            cv2.imshow("Camera view vs BEV", side_by_side)

        if cv2.waitKey(1) & 0xFF == ord('q'):	
            break

    # close the video file
    cap.release()
    cv2.destroyAllWindows()

