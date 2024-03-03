import math
import cv2
from time import sleep, time
import mediapipe as mp
import pyttsx3
import tensorflow as tf

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def calculateHipAngle(left_hip, right_hip):
    x1, y1, _ = left_hip
    x2, y2, _ = right_hip
    hip_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if hip_angle < 0:
        hip_angle += 360
    return hip_angle

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), circle_radius=2, thickness=2))

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))

    if display:
        cv2.imshow('Pose Detection', output_image)
        cv2.waitKey(1)

    return output_image, landmarks

def classifyPose(landmarks, output_image, display=False):
    label = 'Unknown Pose'
    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])


    namas_condition = (
        left_elbow_angle > 160 and left_elbow_angle < 200 and
        right_elbow_angle > 160 and right_elbow_angle < 200 and
        left_shoulder_angle > 170 and left_shoulder_angle < 190 and
        right_shoulder_angle > 170 and right_shoulder_angle < 190
    )

    t_pose_condition = (
        left_elbow_angle > 160 and left_elbow_angle < 200 and
        right_elbow_angle > 160 and right_elbow_angle < 200 and
        left_shoulder_angle > 80 and left_shoulder_angle < 110 and
        right_shoulder_angle > 80 and right_shoulder_angle < 110
    )

    praying_hand_pose_condition = (
        left_elbow_angle > 80 and left_elbow_angle < 120 and
        right_elbow_angle > 80 and right_elbow_angle < 120 and
        left_shoulder_angle > 80 and left_shoulder_angle < 110 and
        right_shoulder_angle > 80 and right_shoulder_angle < 110
    )
    
    still_condition = (
        left_elbow_angle > 160 and left_elbow_angle < 200 and
        right_elbow_angle > 160 and right_elbow_angle < 200 and
        left_shoulder_angle > 5 and left_shoulder_angle < 40 and
        right_shoulder_angle > 5 and right_shoulder_angle < 40
    )

    
    warrior_pose_condition = (
        left_elbow_angle > 160 and left_elbow_angle < 200 and
        right_elbow_angle > 160 and right_elbow_angle < 200 and
        left_shoulder_angle > 90 and left_shoulder_angle < 120 and
        right_shoulder_angle > 60 and right_shoulder_angle < 90
    )

    
    h_pose_condition = (
        left_elbow_angle > 120 and left_elbow_angle < 180 and
        right_elbow_angle > 120 and right_elbow_angle < 180 and
        left_shoulder_angle > 160 and left_shoulder_angle < 200 and
        right_shoulder_angle > 160 and right_shoulder_angle < 200
    )

    # Cobra Pose
    cobra_pose_condition = (
        left_elbow_angle > 60 and left_elbow_angle < 100 and
        right_elbow_angle > 60 and right_elbow_angle < 100 and
        left_shoulder_angle > 160 and left_shoulder_angle < 200 and
        right_shoulder_angle > 160 and right_shoulder_angle < 200
    )

    if still_condition:
        label = 'Still Pose'
        speak_text("Still Pose detected, sometimes it's better to stand relaxed for better circulation of the blood")  # Speak the pose

    elif warrior_pose_condition:
        label = 'Warrior Pose'
        speak_text("Warrior Pose, Warrior I strengthens and stretches your legs and buttocks, the front of your hips, and shins")  # Speak the pose

    elif h_pose_condition:
        label = 'Vrikshasana Pose'
        speak_text("Vrikshasana Pose, calms and relaxes the central nervous system and stretches the entire body")  # Speak the pose

    elif cobra_pose_condition:
        label = 'Cobra Pose'
        speak_text("Cobra Pose detected") 

    elif namas_condition:
        label = 'Namaskar Pose'
        speak_text("Namaskar pose, physical and mental strength, better command over your body, calmness of the mind, balanced energies, and inner peace")  # Speak the pose

    elif t_pose_condition:
        label = 'T Pose'
        speak_text("T Pose detected, stretches the muscles of the shoulders, neck, chest, abdomen, psoas, and strengthens hips, quads, and knees")  # Speak the pose

    elif praying_hand_pose_condition:
        label = 'Praying Hand Pose'
        speak_text("Praying Hand Pose detected")  

    if label != 'Unknown Pose':
        color = (0, 255, 0)
        
        sleep(3)

    cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    if display:
        cv2.imshow('Pose Classification', output_image)
        cv2.waitKey(1)
    else:
        return output_image, label

def speak_text(text, rate=100, pause_duration=1):
    engine.setProperty('rate', rate)  
    engine.say(text)
    engine.runAndWait()
    sleep(pause_duration)  

pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)
video = cv2.VideoCapture(1)
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

while video.isOpened():
    ok, frame = video.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    frame = cv2.resize(frame, (int(frame_width * (720 / frame_height)), 720))
    
    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    if landmarks:
        frame, _ = classifyPose(landmarks, frame, display=False)

    cv2.imshow('Pose Classification', frame)
    sleep(0.01)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
