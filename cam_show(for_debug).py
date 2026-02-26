import cv2

def show(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    ok, frame = cap.read()
    print("idx", idx, "ok", ok, "shape", None if frame is None else frame.shape)
    if ok:
        cv2.imshow(f"cam {idx}", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    cap.release()

show(0)
show(1)
