import cv2
import numpy as np
import dlib
import math

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_1.dat")
font = cv2.FONT_HERSHEY_SIMPLEX
keyboard = np.zeros((1000, 1500, 3), np.uint8)

keys_set_1 = {0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
              5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
              10: "Z", 11: "X", 12: "C", 13: "V", 14: "B"}
blinking_board=np.zeros((500,500),np.uint8)
blinking_board[:]=255

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def letter(letter_index,text,light):
    # Keys
    if letter_index == 0:
        x = 0
        y = 0
    elif letter_index == 1:
        x = 200
        y = 0
    elif letter_index == 2:
        x = 400
        y = 0
    elif letter_index == 3:
        x = 600
        y = 0
    elif letter_index == 4:
        x = 800
        y = 0
    elif letter_index == 5:
        x = 0
        y = 200
    elif letter_index == 6:
        x = 200
        y = 200
    elif letter_index == 7:
        x = 400
        y = 200
    elif letter_index == 8:
        x = 600
        y = 200
    elif letter_index == 9:
        x = 800
        y = 200
    elif letter_index == 10:
        x = 0
        y = 400
    elif letter_index == 11:
        x = 200
        y = 400
    elif letter_index == 12:
        x = 400
        y = 400
    elif letter_index == 13:
        x = 600
        y = 400
    elif letter_index == 14:
        x = 800
        y = 400



    width = 200
    height = 200
    th = 3  # thickness
    if light is True:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255,255), -1)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)
    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)



def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = math.hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = math.hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


def get_gaze_ratio(eye_points,landmarks):
    left_eye_region = np.array([(landmarks.part(eye_points[0]).x, landmarks.part(eye_points[0]).y),
                                (landmarks.part(eye_points[1]).x, landmarks.part(eye_points[1]).y),
                                (landmarks.part(eye_points[2]).x, landmarks.part(eye_points[2]).y),
                                (landmarks.part(eye_points[3]).x, landmarks.part(eye_points[3]).y),
                                (landmarks.part(eye_points[4]).x, landmarks.part(eye_points[4]).y),
                                (landmarks.part(eye_points[5]).x, landmarks.part(eye_points[5]).y)], np.int32)
    # create mask
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    left_eye = cv2.bitwise_and(gray, gray, mask=mask)

    # get extreme point
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    # threshold
    gray_eye = left_eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]  # left side of the eye
    left_side_white = cv2.countNonZero(left_side_threshold)  # count 1 in left side => white part of th eye

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]  # right side of the eye
    right_side_white = cv2.countNonZero(right_side_threshold)  # count white pixels in the right part of the eye
    if left_side_white==0:
        gaze_ratio=0.5
    elif right_side_white==0:
        gaze_ratio=9
    else:
        gaze_ratio = left_side_white / right_side_white  # divide the left white pixels to right


    return gaze_ratio

#counters
frames=0
letter_index=0
blinking=0
text=""
while True:
    _, frame = cap.read()
    frames+=1
    keyboard[:]=(0,0,0) # reset keyboard for update
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = np.zeros((500, 500, 3), np.uint8)
    active_letter=keys_set_1[letter_index]
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
            blinking+=1
            frames-=1
            if blinking == 5:
                text += active_letter
        else:
            blinking=0



        # Gaze rato left to write
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        if gaze_ratio <= 1:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
            new_frame[:] = (0, 0, 255)
        elif 1 < gaze_ratio < 5.5:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            new_frame[:] = (255, 0, 0) # more that 5.5
            #  is looking at left
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)


        # threshold_eye = cv2.resize(threshold_eye, None, fx=5, fy=5)
        # only_eye_big = cv2.resize(gray_eye, None, fx=5, fy=5)
        # cv2.imshow("Eye", only_eye_big)
        # cv2.imshow("Threshold", threshold_eye)
        # cv2.imshow("Left eye", left_eye)

        if frames==10:
            letter_index+=1
            frames=0
        if letter_index==15:
            letter_index=0


        for i in range(15):
            if i == letter_index:
                light = True
            else:
                light = False
            letter(i, keys_set_1[i], light)

        cv2.putText(blinking_board, text, (10, 100), font, 4, 0, 3)

        cv2.imshow("keyboard", keyboard)
        cv2.imshow("blinking_board",blinking_board)
        # cv2.imshow("new frame",new_frame)
        # cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()