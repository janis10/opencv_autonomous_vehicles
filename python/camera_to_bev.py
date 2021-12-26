import cv2
import argparse
from cv_utils import getBEV

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

