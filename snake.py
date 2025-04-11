import cv2
import mediapipe as mp
import numpy as np
import math
import random

# Hand tracker
hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mp.solutions.drawing_utils

# Snake Game
class SnakeGame:
    def __init__(self):
        self.points, self.lengths = [], []
        self.length_limit = 150
        self.total_length = 0
        self.prev = (0, 0)
        self.score = 0
        self.food = self.random_food()
        self.game_over = False

    def random_food(self):
        return random.randint(100, 500), random.randint(100, 400)

    def update(self, img, head):
        if self.game_over: return img

        dist = math.hypot(head[0] - self.prev[0], head[1] - self.prev[1])
        self.points.append(head)
        self.lengths.append(dist)
        self.total_length += dist
        self.prev = head

        while self.total_length > self.length_limit:
            self.total_length -= self.lengths.pop(0)
            self.points.pop(0)

        # Draw snake
        for i in range(1, len(self.points)):
            cv2.line(img, self.points[i-1], self.points[i], (0, 255, 0), 15)

        # Draw food
        fx, fy = self.food
        cv2.circle(img, (fx, fy), 20, (0, 0, 255), -1)

        # Eat food
        if math.hypot(head[0] - fx, head[1] - fy) < 20:
            self.length_limit += 50
            self.score += 1
            self.food = self.random_food()

        # Score text
        cv2.putText(img, f"Score: {self.score}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Collision
        if len(self.points) > 10:
            pts = np.array(self.points[:-10], np.int32).reshape((-1, 1, 2))
            if cv2.pointPolygonTest(pts, head, True) >= -1:
                self.game_over = True

        return img

    def game_over_screen(self):
        img = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(img, "GAME OVER", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(img, f"Score: {self.score}", (200, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, "Press R to Restart or Q to Quit", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return img

# Main loop
cap = cv2.VideoCapture(0)
game = SnakeGame()

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and not game.game_over:
        for hand in result.multi_hand_landmarks:
            h, w = img.shape[:2]
            lm = hand.landmark[8]  # Index tip
            point = int(lm.x * w), int(lm.y * h)
            img = game.update(img, point)
            draw.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)

    elif game.game_over:
        img = game.game_over_screen()

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r') and game.game_over: game = SnakeGame()

cap.release()
cv2.destroyAllWindows()
