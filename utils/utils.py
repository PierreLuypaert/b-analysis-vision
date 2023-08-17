import cv2
from utils.ball_direction import BallDirection
def intersect(point1, point2, target_point3, target_point4):
        # Lines can be defined by two points each
        # Point1 and Point2 define the first line
        # Point3 and Point4 define the second line

        # Check if the lines are parallel (have the same slope)
        m1 = (point2[1] - point1[1]) / (point2[0] - point1[0]) if point2[0] != point1[0] else None
        m2 = (target_point4[1] - target_point3[1]) / (target_point4[0] - target_point3[0]) if target_point4[0] != target_point3[0] else None
        if m1 == m2:
            return False

        # Calculate the y-intercepts (b) for the equations of the lines
        b1 = point1[1] - m1 * point1[0] if m1 is not None else point1[0]
        b2 = target_point3[1] - m2 * target_point3[0] if m2 is not None else target_point3[0]

        # Calculate the point of intersection by solving the equation for x
        if m1 is not None and m2 is not None:
            x_intersection = (b2 - b1) / (m1 - m2)
            y_intersection = m1 * x_intersection + b1
        elif m1 is not None:
            x_intersection = target_point3[0]
            y_intersection = m1 * x_intersection + b1
        else:
            x_intersection = point1[0]
            y_intersection = m2 * x_intersection + b2

        print(x_intersection, y_intersection)

        if x_intersection >= target_point3[0] and x_intersection <= target_point4[0] and y_intersection >= target_point3[1] and y_intersection <= target_point4[1]:
            return True
        return False

def display_ball_velocity(frame, ball_velocity):
    cv2.putText(frame, str(round(ball_velocity)) + "km/h", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

def draw_ball_direction(frame, previous_center, new_center, x1, x2, y1, y2, h, proportion_percentage, ball_velocity):
    direction = new_center[0] - previous_center[0]
    detection_height = h * proportion_percentage/100
    difference_each_side = int((h - detection_height)/2)
    go_to_left_goal = intersect(previous_center,new_center,(x1,y1+difference_each_side),(x1,y2-difference_each_side)) and direction < 0 and (abs(new_center[0] - x1)<200 or ball_velocity>17)
    go_to_right_goal = intersect(previous_center,new_center,(x2,y1+difference_each_side),(x2,y2-difference_each_side)) and direction > 0 and (abs(new_center[0] - x2)<200 or ball_velocity>17)
    line_color = (0,255,0) if go_to_left_goal or go_to_right_goal else (0,0,0)
    left_color = (0,255,0) if go_to_left_goal else (0,0,0)
    right_color = (0,255,0) if go_to_right_goal else (0,0,0)
    cv2.line(frame, (x1,(y1+difference_each_side)),(x1,(y2-difference_each_side)), color=left_color, thickness=10)
    cv2.line(frame, (x2,(y1+difference_each_side)),(x2,(y2-difference_each_side)), color=right_color, thickness=10)
    cv2.line(frame, previous_center, new_center, color=line_color, thickness=6)
    return BallDirection.RED_GOAL if direction<0 else BallDirection.BLUE_GOAL, go_to_left_goal or go_to_right_goal
    '''
    cvzone.cornerRect(self.frame, (x1, y1, DEFENSIVE_ZONE_WIDTH, h), colorC=colorLeft, colorR=(0,200,16), t=10)
    cvzone.cornerRect(self.frame, (x2-DEFENSIVE_ZONE_WIDTH, y1, DEFENSIVE_ZONE_WIDTH, h), colorC=colorRight, colorR=(0,200,16), t=10)
    '''

def goal(frame):
    cv2.putText(frame, "GOAL!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)