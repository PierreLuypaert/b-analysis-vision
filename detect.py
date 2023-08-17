# Avi Shah - Basketball Shot Detector/Tracker - July 2023

from ultralytics import YOLO
import cv2
import numpy as np
import math
import cvzone
from utils.utils import display_ball_velocity, draw_ball_direction, goal
BABY_FOOT_WIDTH_IN_CM = 115
BABY_FOOT_HEIGHT_IN_CM = 68
CAMERA_FRAME_RATE = 60
DEFENSIVE_ZONE_WIDTH = 400
GOAL_DETECTION_PROPORTION_OF_THE_WALL_IN_PERCENTAGE = 57

from enum import Enum

# class syntax
class BallState(Enum):
    CHILLING = 0
    DANGEROUS = 1
    GOAL = 2

class ShotDetector:
    def __init__(self, video_url):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("weights/best_ball_and_table_m_100.pt")
        self.class_names = ['ball','baby-foot-table']
        self.last_ball_state = [] #check if goal or not
        self.ball_detected_in_frame = False

        self.last_seen_ball_x = 0
        self.last_seen_ball_y = 0

        self.direction = None
        # Use video - replace text with your video path
        #self.cap = cv2.VideoCapture("ressources/baby.mp4")
        self.cap = cv2.VideoCapture(video_url)
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the duration in seconds
        self.Duration = int(self.total_frames / 30)
        self.frame_count = 0
        self.frame_count_ball_undetected = 0
        self.frame = None

        self.ball_velocity = 0
        self.table_border_left_x  = 0
        self.table_border_right_x = 0

        # Output video file name
        self.output_video_file = "ressources/output.mp4"

        #Scale between reality and camera environment
        self.scale = 1
        self.BallSpeed = []
        self.BallPosition = []
        self.BabyDelimitation=[]
        self.PredictedGoals=[]

    def calculate_distance(self, x, y, x2, y2):
        # Calculate the differences in x and y coordinates
        delta_x = x2 - x
        delta_y = y2 - y

        # Square the differences
        delta_x_squared = delta_x ** 2
        delta_y_squared = delta_y ** 2

        # Calculate the sum of squared differences
        sum_squared_differences = delta_x_squared + delta_y_squared

        # Take the square root to get the distance
        distance = math.sqrt(sum_squared_differences)

        return distance


    def run(self):
        new_center=None
        previous_center=None
        while True:
            ret, self.frame = self.cap.read()
            if self.frame_count==0:
                # Create a video writer object
                width, height, _= self.frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(self.output_video_file, fourcc, 30, (height, width))
            if not ret:
                # End of the video or an error occurred
                break

            self.ball_detected_in_frame = False
            self.frame_count += 1
            results = self.model.track(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Class Name
                    cls = int(box.cls[0])
                    current_class = self.class_names[cls]                        
                    
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    if current_class=='ball':
                        self.frame_count_ball_undetected=0
                        self.ball_detected_in_frame = True
                        if new_center!=None:
                            previous_center=new_center
                        new_center = (int(x1 + w / 2), int(y1 + h / 2))
                        self.last_seen_ball_x = new_center[0]
                        self.last_seen_ball_y = new_center[1]
                        if new_center[0] < self.table_border_left_x-100: 
                            if self.direction != None:
                                self.BallPosition.append([new_center[0]-self.table_border_left_x, new_center[1]-y1, self.ball_velocity, self.direction.value, BallState.GOAL.value, int(self.frame_count/self.total_frames*self.Duration)])
                            goal(self.frame)
                        elif new_center[0] > self.table_border_right_x+100:
                            if self.direction != None:
                                self.BallPosition.append([new_center[0]-self.table_border_left_x, new_center[1]-y1, self.ball_velocity, self.direction.value, BallState.GOAL.value, int(self.frame_count/self.total_frames*self.Duration)])
                            goal(self.frame)
                        
                            

                    elif current_class=='baby-foot-table':  
                        baby_foot_diagonal_in_cm = math.sqrt(BABY_FOOT_WIDTH_IN_CM**2 + BABY_FOOT_HEIGHT_IN_CM**2)
                        baby_foot_diagonal_in_px = math.sqrt(w**2 + h**2)
                        self.scale = baby_foot_diagonal_in_cm / baby_foot_diagonal_in_px
                        self.table_border_left_x = x1
                        self.table_border_right_x = x2
                        if len(self.BabyDelimitation)==0:
                            self.BabyDelimitation.append([x1, y1])
                            self.BabyDelimitation.append([x2, y2])
                        if previous_center!=None:
                            self.direction, dangerous = draw_ball_direction(self.frame, previous_center,new_center, x1, x2, y1, y2, h, GOAL_DETECTION_PROPORTION_OF_THE_WALL_IN_PERCENTAGE, self.ball_velocity)
                            state = BallState.CHILLING
                            if dangerous:
                                state = BallState.DANGEROUS
                                self.last_ball_state.append(self.direction)
                            else:
                                self.last_ball_state=[]
                            self.BallPosition.append([new_center[0]-self.table_border_left_x, new_center[1]-y1, self.ball_velocity, self.direction.value, state.value, int(self.frame_count/self.total_frames*self.Duration)])
                
                    cvzone.cornerRect(self.frame, (x1, y1, w, h))

            if not(self.ball_detected_in_frame):
                previous_center=None
                new_center=None

            if self.ball_detected_in_frame and previous_center!=None:
                traveled_cm = self.calculate_distance(new_center[0], new_center[1], previous_center[0], previous_center[1])*self.scale
                self.ball_velocity = (traveled_cm / (1/CAMERA_FRAME_RATE) ) * 0.036 #convert to km/h
                self.BallSpeed.append(self.ball_velocity)
                display_ball_velocity(self.frame, self.ball_velocity)

            elif not(self.ball_detected_in_frame):
                self.frame_count_ball_undetected+=1
                #if we dont see the ball for 5 frames, it means goal
                if self.frame_count_ball_undetected>=5 and len(self.last_ball_state)>1:
                    if self.direction != None:
                        self.BallPosition.append([self.last_seen_ball_x-self.table_border_left_x, self.last_seen_ball_y-y1, self.ball_velocity, self.direction.value, BallState.GOAL.value, int(self.frame_count/self.total_frames*self.Duration)])
                    goal(self.frame)

            
            self.video_writer.write(self.frame)

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        print("close process")
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        result = {
            'BallSpeed': self.BallSpeed,
            'BallPosition': self.BallPosition,
            'BabyDelimitation': self.BabyDelimitation,
            'MatchDuration': self.Duration,
            'PredictedGoals': self.PredictedGoals
        }
        return result