import cv2 

for i in range(4):
    cap =cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    print(i, "opened:" , cap.isOpened(), "got_frame:" , ok , "frame_none:", frame is None)
    cap.release()

    