import cv2
import mediapipe as mp
import numpy as np
import time

class ExerciseTracker:
    def __init__(self, exercise_id=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.exercise_id = exercise_id
        self.exercise_list = ["Push-up", "Plank", "Pull-up", "Hammer Curl", "Tricep Dip", "Tricep Pull-down"]
        self.rep_count = 0
        self.in_progress = False
        self.last_rep_time = 0  # Track last successful rep time
        self.min_rep_time = 0.2 #Threshold time to ignore false rep occuring due to sudden changes in the media pipes posture 
        self.plank_start_time = 0
        self.plank_timer_running = False
        self.elapsed_time = 0
        self.bad_posture = False  # Track bad posture
        self.exercise_type = self.exercise_list[self.exercise_id]
        self.calories_burned = 0
        self.start_time = None
        self.exercise_duration = 0

    #Calculating angle between the points
    def calculate_angle_3d(self, a, v, b):
        a = np.array(a)
        b = np.array(b)
        v = np.array(v)
        av = a-v
        bv = b-v
        radians = np.dot(av,bv) / (np.linalg.norm(av) * np.linalg.norm(bv))
        angle = np.degrees(np.arccos(np.clip(radians,-1.0,1.0)))
        return angle

    def is_push_up(self, angles):
        left_status = "Down" if angles['left_elbow'] < 95 else "Up" if angles['left_elbow'] > 130 else "In Progress"
        right_status = "Down" if angles['right_elbow'] < 95 else "Up" if angles['right_elbow'] > 130 else "In Progress"
        
        if left_status != right_status:
            self.bad_posture = True
            return "Asymmetrical Movement"
        self.bad_posture = False
        return left_status

    def is_plank(self, angles):
        if 140 < angles['left_hip_angle'] < 180 and 140 < angles['right_hip_angle'] < 180:
            self.bad_posture = False
            return "Plank"
        self.bad_posture = True
        return "Not Plank"

    def is_pull_up(self, angles):
        left_up = angles['left_elbow'] < 30 and angles['left_shoulder'] < 50
        right_up = angles['right_elbow'] < 30 and angles['right_shoulder'] < 50
        left_down = angles['left_elbow'] > 150
        right_down = angles['right_elbow'] > 150

        if left_up and right_up:
            self.bad_posture = False
            return "Up"
        elif left_down and right_down:
            self.bad_posture = False
            return "Down"
        self.bad_posture = True
        return "In Progress"

    def is_hammer_curl(self, angles):
        left_status = "Up" if angles['left_elbow'] < 60 else "Down" if angles['left_elbow'] > 120 else "In Progress"
        right_status = "Up" if angles['right_elbow'] < 60 else "Down" if angles['right_elbow'] > 120 else "In Progress"
        
        if left_status != right_status:
            self.bad_posture = True
            return "Asymmetrical Movement"
        self.bad_posture = False
        return left_status

    def is_tricep_dip(self, angles):
        left_status = "Down" if angles['left_elbow'] < 110 else "Up" if angles['left_elbow'] > 130 else "In Progress"
        right_status = "Down" if angles['right_elbow'] < 110 else "Up" if angles['right_elbow'] > 130 else "In Progress"
        
        if left_status != right_status:
            self.bad_posture = True
            return "Asymmetrical Movement"
        self.bad_posture = False
        return left_status

    def is_tricep_pull_down(self, angles):
        left_elbow_down = angles['left_elbow'] > 130
        right_elbow_down = angles['right_elbow'] > 130
        left_elbow_up = angles['left_elbow'] < 90
        right_elbow_up = angles['right_elbow'] < 90

        if left_elbow_down and right_elbow_down:
            self.bad_posture = False
            return "Down"
        elif left_elbow_up and right_elbow_up:
            self.bad_posture = False
            return "Up"
        
        self.bad_posture = True
        return "In Progress"

    def calculate_calories(self, duration):
        """
        Estimate calories burned based on exercise type and duration (in seconds).
        Assumes average calorie burn rates for different exercises.
        """
        # Random calorie burn rates
        calorie_burn_rates = {
            "Push-up": 7, 
            "Plank": 4,
            "Pull-up": 8,
            "Hammer Curl": 5,
            "Tricep Dip": 6,
            "Tricep Pull-down": 6
        }

        # Get the calorie burn rate for the selected exercise
        rate = calorie_burn_rates.get(self.exercise_type, 5)  # Default to 5 kcal/min

        # Convert duration from seconds to minutes
        duration_minutes = duration / 60
        return rate * duration_minutes

    def detect_exercise_state(self, angles):
        state_functions = {
            "Push-up": self.is_push_up,
            "Plank": self.is_plank,
            "Pull-up": self.is_pull_up,
            "Hammer Curl": self.is_hammer_curl,
            "Tricep Dip": self.is_tricep_dip,
            "Tricep Pull-down": self.is_tricep_pull_down
        }
        return state_functions[self.exercise_type[self.exercise_id]](angles)

    def check_good_posture(self, angles, exercise_type):
        if exercise_type == "Push-up":
            #See whether the body angles are within the threshold or not.
            if not (60 < angles['left_elbow'] < 160 and 60 < angles['right_elbow'] < 160):
                return False #bad posture 

        elif exercise_type == "Pull-up":
            if not (angles['left_shoulder'] > 20 and angles['right_shoulder'] > 20):
                return False #bad posture

        elif exercise_type == "Hammer Curl":
            
            if not (0 < angles['left_elbow'] < 180 and 0 < angles['right_elbow'] < 180):
                return False #bad posture
            
            angle_diff = abs(angles['left_elbow'] - angles['right_elbow'])
            if angle_diff > 40:  # Tolerance of 10 degrees for minor differences
                return False #bad posture
            return True # correct Posture


        elif exercise_type == "Tricep Dip":
            if not (40 < angles['left_elbow'] < 180 and 40 < angles['right_elbow'] < 180):
                return False  #bad posture

        elif exercise_type == "Plank":
            if not (140 < angles['left_hip_angle'] < 180 and 140 < angles['right_hip_angle'] < 180):
                return False  #bad posture

        elif exercise_type == "Tricep Pull-down":
            if not (50 < angles['left_elbow'] < 180 and 50 < angles['right_elbow'] < 180):
                return False  #bad posture

        return True  # Posture is good

    def count_reps(self, frame, angles,result):
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        exercise_state = ""

        if not self.check_good_posture(angles, self.exercise_type):
            cv2.putText(frame, 'Posture Incorrect', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            return  # Don't count rep if posture is bad
        else:
            cv2.putText(frame, 'Posture Correct', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if self.exercise_type == "Push-up":
            exercise_state = self.is_push_up(angles)
        elif self.exercise_type == "Plank":
            exercise_state = self.is_plank(angles)
        elif self.exercise_type == "Pull-up":
            exercise_state = self.is_pull_up(angles)
        elif self.exercise_type == "Hammer Curl":
            exercise_state = self.is_hammer_curl(angles)
        elif self.exercise_type == "Tricep Dip":
            exercise_state = self.is_tricep_dip(angles)
        elif self.exercise_type == "Tricep Pull-down":
            exercise_state = self.is_tricep_pull_down(angles)

        # Plank Timer 
        if self.exercise_type == "Plank":
            if exercise_state == "Plank" and not self.plank_timer_running:
                self.plank_start_time = time.time()  # Start the timer when plank position is detected
                self.plank_timer_running = True
            elif exercise_state != "Plank" and self.plank_timer_running:
                self.plank_timer_running = False  # Reset the timer if the plank position is lost

            if self.plank_timer_running:
                self.elapsed_time = int(time.time() - self.plank_start_time)
                self.rep_count = self.elapsed_time
                cv2.putText(frame, f'Time: {self.elapsed_time}s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # rep count logic
        elif self.exercise_type == "Pull-up":
            if exercise_state == "Up":
                self.in_progress = True  # Only allow counting if "Up" is detected first
            
            if exercise_state == "Down" and self.in_progress:
                self.rep_count += 1  # Count a rep when moving from Up to Down
                self.in_progress = False #reset

        elif self.exercise_type == "Push-up":
            if exercise_state == "Down" and not self.in_progress:
                # Only count rep when moving from up to down
                self.rep_count += 1
                self.in_progress = True  # Set in_progress to True after counting a rep
            elif exercise_state == "Up" and self.in_progress:
                # Set in_progress to False when moving from Down to up
                self.in_progress = False

        elif self.exercise_type == "Tricep Pull-down":
            if exercise_state == "Down" and not self.in_progress:
                self.rep_count += 1
                self.in_progress = True
            elif exercise_state == "Up" and self.in_progress:
                self.in_progress = False

        elif self.exercise_type == "Hammer Curl":
            if exercise_state == "Up" and not self.in_progress:
                self.in_progress = True  # tracking the curl motion
            elif exercise_state == "Down" and self.in_progress:
                self.rep_count += 1  # Count a rep when returning from curl to extended position
                self.in_progress = False #reset

        elif self.exercise_type == "Tricep Dip":
            if exercise_state == "Down" and not self.in_progress:
                self.rep_count += 1
                self.in_progress = True
            elif exercise_state == "Up" and self.in_progress:
                self.in_progress = False

        else:
            #to handle unexpected inputs
            if exercise_state == f"{self.exercise_type} Down" and not self.in_progress:
                self.rep_count += 1
                self.in_progress = True
            elif exercise_state == f"{self.exercise_type} Up" and self.in_progress:
                self.in_progress = False

        if self.start_time is None:
            self.start_time = time.time() #start the timer after first rep

        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time  #time recorded
            self.exercise_duration = elapsed_time
            self.calories_burned = self.calculate_calories(elapsed_time) #estimating calories based on time duration not that accurate though will update it on the basis of reps later!
        
        if self.exercise_type != "Plank":
            cv2.putText(frame, f'Reps: {self.rep_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    def get_angles_from_landmarks(self, landmarks):
        keypoints = {part: [landmarks[getattr(self.mp_pose.PoseLandmark, part).value].x,
                            landmarks[getattr(self.mp_pose.PoseLandmark, part).value].y] 
                    for part in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", 
                                "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
                                "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"]}

        angles = {}

        # left elbow angle
        if all(k in keypoints for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"]):
            angles["left_elbow"] = self.calculate_angle_3d(
                keypoints["LEFT_SHOULDER"], keypoints["LEFT_ELBOW"], keypoints["LEFT_WRIST"]
            )

        # right elbow angle
        if all(k in keypoints for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"]):
            angles["right_elbow"] = self.calculate_angle_3d(
                keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_ELBOW"], keypoints["RIGHT_WRIST"]
            )

        # left shoulder angle
        if all(k in keypoints for k in ["LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_HIP"]):
            angles["left_shoulder"] = self.calculate_angle_3d(
                keypoints["LEFT_HIP"], keypoints["LEFT_SHOULDER"], keypoints["LEFT_ELBOW"]
            )

        # right shoulder angle
        if all(k in keypoints for k in ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_HIP"]):
            angles["right_shoulder"] = self.calculate_angle_3d(
                keypoints["RIGHT_HIP"], keypoints["RIGHT_SHOULDER"], keypoints["RIGHT_ELBOW"]
            )

        # body angle not needed but kept it for now
        if all(k in keypoints for k in ["LEFT_HIP", "LEFT_SHOULDER"]):
            angles["body_angle"] = self.calculate_angle_3d(
                keypoints["LEFT_HIP"], keypoints["LEFT_SHOULDER"], 
                [keypoints["LEFT_SHOULDER"][0], keypoints["LEFT_SHOULDER"][1] - 1]
            )

        # left hip angle 
        if all(k in keypoints for k in ["LEFT_KNEE", "LEFT_HIP", "LEFT_SHOULDER"]):
            angles["left_hip_angle"] = self.calculate_angle_3d(
                keypoints["LEFT_KNEE"], keypoints["LEFT_HIP"], keypoints["LEFT_SHOULDER"]
            )

        # right hip angle
        if all(k in keypoints for k in ["RIGHT_KNEE", "RIGHT_HIP", "RIGHT_SHOULDER"]):
            angles["right_hip_angle"] = self.calculate_angle_3d(
                keypoints["RIGHT_KNEE"], keypoints["RIGHT_HIP"], keypoints["RIGHT_SHOULDER"]
            )

        return angles


    def process_videos(self, filename):
        vid = cv2.VideoCapture(filename) 
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            #opencv works on BGR 
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_frame)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                angles = self. get_angles_from_landmarks(landmarks)
                self.count_reps(frame, angles, result)
                mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                cv2.imshow('Exercise Tracker', cv2.resize(frame, (700, 700)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        vid.release()
        cv2.destroyAllWindows()
        if self.exercise_type[self.exercise_id] == "Plank":
            self.rep_count = self.elapsed_time
        return self.rep_count

#for my reference
# if __name__ == "__main__":
#     filename = "dataset\plank\plank_5.mp4"
#     tracker = ExerciseTracker(exercise_id=1)  # Exercise ID for Tricep Pull-down
#     reps = tracker.process_videos(filename)
#     print(reps, tracker.calories_burned, tracker.exercise_duration)


    # state_functions = {
    #         "Push-up": self.is_push_up,
    #         "Plank": self.is_plank,
    #         "Pull-up": self.is_pull_up,
    #         "Hammer Curl": self.is_hammer_curl,
    #         "Tricep Dip": self.is_tricep_dip,
    #         "Tricep Pull-down": self.is_tricep_pull_down  # Add new exercise here
    #     }