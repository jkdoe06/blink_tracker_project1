import cv2
import mediapipe as mp

RIGHT_EYE_P1_TO_P6 = [33, 160, 158, 133, 153, 144]
LEFT_EYE_P1_TO_P6 = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.21

# setup facemesh
face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# build eye index set
left_eye_connections = mp.solutions.face_mesh.FACEMESH_LEFT_EYE
right_eye_connections = mp.solutions.face_mesh.FACEMESH_RIGHT_EYE
eye_landmark_indices = set()
for start_landmark_index, end_landmark_index in left_eye_connections:
    eye_landmark_indices.add(start_landmark_index)
    eye_landmark_indices.add(end_landmark_index)
for start_landmark_index, end_landmark_index in right_eye_connections:
    eye_landmark_indices.add(start_landmark_index)
    eye_landmark_indices.add(end_landmark_index)

# camera

def_cam = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

blink_count = 0
closed_frames = 0
min_closed_frames = 2
eye_was_closed = False

while True:
    # read frame
    got_frame, frame_data = def_cam.read()
    if not got_frame:
        print("No frame received from camera, so the program cannot continue processing.")
        break

    # bgr to rgb
    rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

    # process facemesh
    results = face_mesh_detector.process(rgb_frame)

    ear_avg = 0.0

    if results.multi_face_landmarks:
        face0 = results.multi_face_landmarks[0]
        frame_height, frame_width, frame_channels = frame_data.shape

        # convert normalized to pixel
        right_eye_p1 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[0]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[0]].y * frame_height))
        right_eye_p2 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[1]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[1]].y * frame_height))
        right_eye_p3 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[2]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[2]].y * frame_height))
        right_eye_p4 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[3]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[3]].y * frame_height))
        right_eye_p5 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[4]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[4]].y * frame_height))
        right_eye_p6 = (int(face0.landmark[RIGHT_EYE_P1_TO_P6[5]].x * frame_width), int(face0.landmark[RIGHT_EYE_P1_TO_P6[5]].y * frame_height))

        left_eye_p1 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[0]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[0]].y * frame_height))
        left_eye_p2 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[1]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[1]].y * frame_height))
        left_eye_p3 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[2]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[2]].y * frame_height))
        left_eye_p4 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[3]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[3]].y * frame_height))
        left_eye_p5 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[4]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[4]].y * frame_height))
        left_eye_p6 = (int(face0.landmark[LEFT_EYE_P1_TO_P6[5]].x * frame_width), int(face0.landmark[LEFT_EYE_P1_TO_P6[5]].y * frame_height))

        # compute ear math
        right_d_p2_p6 = ((right_eye_p2[0] - right_eye_p6[0]) ** 2 + (right_eye_p2[1] - right_eye_p6[1]) ** 2) ** 0.5
        right_d_p3_p5 = ((right_eye_p3[0] - right_eye_p5[0]) ** 2 + (right_eye_p3[1] - right_eye_p5[1]) ** 2) ** 0.5
        right_d_p1_p4 = ((right_eye_p1[0] - right_eye_p4[0]) ** 2 + (right_eye_p1[1] - right_eye_p4[1]) ** 2) ** 0.5
        if right_d_p1_p4 == 0:
            ear_right = 0.0
        else:
            ear_right = (right_d_p2_p6 + right_d_p3_p5) / (2 * right_d_p1_p4)

        left_d_p2_p6 = ((left_eye_p2[0] - left_eye_p6[0]) ** 2 + (left_eye_p2[1] - left_eye_p6[1]) ** 2) ** 0.5
        left_d_p3_p5 = ((left_eye_p3[0] - left_eye_p5[0]) ** 2 + (left_eye_p3[1] - left_eye_p5[1]) ** 2) ** 0.5
        left_d_p1_p4 = ((left_eye_p1[0] - left_eye_p4[0]) ** 2 + (left_eye_p1[1] - left_eye_p4[1]) ** 2) ** 0.5
        if left_d_p1_p4 == 0:
            ear_left = 0.0
        else:
            ear_left = (left_d_p2_p6 + left_d_p3_p5) / (2 * left_d_p1_p4)

        ear_avg = (ear_right + ear_left) / 2

        # blink logic
        if ear_avg < EAR_THRESHOLD:
            closed_frames += 1
        else:
            if eye_was_closed:
                blink_count += 1
            closed_frames = 0

        eye_was_closed = closed_frames >= min_closed_frames

        # draw circles
        for eye_landmark_index in eye_landmark_indices:
            current_landmark = face0.landmark[eye_landmark_index]
            pixel_x = int(current_landmark.x * frame_width)
            pixel_y = int(current_landmark.y * frame_height)
            cv2.circle(frame_data, (pixel_x, pixel_y), 2, (0, 255, 0), -1)
    else:
        closed_frames = 0
        eye_was_closed = False

    # overlay text
    cv2.putText(frame_data, f"EAR: {ear_avg:.3f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_data, f"Blinks: {blink_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame_data)

    # quit and cleanup
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# quit and cleanup
face_mesh_detector.close()
def_cam.release()
cv2.destroyAllWindows()
