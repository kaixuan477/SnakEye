import cv2
import mediapipe as mp
import numpy as np
import time
import multiprocessing
import pygame
import sys
import random
from pygame.locals import Color
from multiprocessing import Manager

# Colors
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)

# Define global variables
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
FPS = 10

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
FORWARD = (0, 0)

# Eye tracker function
def eye_tracker(shared_direction):
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        if not success:
            continue

        start = time.time()

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # See where the user's head is tilting
                if y < -5:
                    text = "Looking Left"
                elif y > 5:
                    text = "Looking Right"
                elif x < -2:
                    text = "Looking Down"
                elif x > 5:
                    text = "Looking Up"
                else:
                    text = "Forward"
                
                # Update the shared direction
                shared_direction.value = text

                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                # Add the text on the image
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("Eye Tracker FPS: ", fps)

        cv2.imshow('Eye Tracker', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# Snake class
class Snake:
    def __init__(self, shared_direction):
        self.length = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = FORWARD  # Initial direction
        self.color = GREEN
        self.shared_direction = shared_direction

    def get_head_position(self):
        return self.positions[0]

    def update(self):
        current = self.get_head_position()

        # Adjust direction based on eye movement information
        eye_movement = self.shared_direction.value
        if eye_movement == "Looking Up":
            self.direction = UP
        elif eye_movement == "Looking Down":
            self.direction = DOWN
        elif eye_movement == "Looking Left":
            self.direction = LEFT
        elif eye_movement == "Looking Right":
            self.direction = RIGHT
        elif eye_movement == "Forward":
            self.direction = FORWARD

        x, y = self.direction
        new = (((current[0] + (x * GRID_SIZE)) % WIDTH), (current[1] + (y * GRID_SIZE)) % HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            self.reset()
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

    def reset(self):
        self.length = 1
        self.positions = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = FORWARD  # Reset to initial direction

    def render(self, surface):
        for p in self.positions:
            pygame.draw.rect(surface, self.color, (p[0], p[1], GRID_SIZE, GRID_SIZE))

# Fruit class
class Fruit:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, (WIDTH // GRID_SIZE) - 1) * GRID_SIZE,
                         random.randint(0, (HEIGHT // GRID_SIZE) - 1) * GRID_SIZE)

    def render(self, surface):
        pygame.draw.rect(surface, self.color, (self.position[0], self.position[1], GRID_SIZE, GRID_SIZE))

# Main function for the snake game
def snake_game(shared_direction):
    pygame.init()

    # Set up the screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    pygame.display.set_caption("Snake Game")
    clock = pygame.time.Clock()

    # Create an instance of the Snake class with the shared direction
    snake = Snake(shared_direction)
    fruit = Fruit()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        snake.update()
        if snake.get_head_position() == fruit.position:
            snake.length += 1
            fruit.randomize_position()

        screen.fill((0, 0, 0))
        snake.render(screen)
        fruit.render(screen)

        pygame.display.update()
        clock.tick(FPS)

# Main process
if __name__ == "__main__":
    # Create a Manager to share data between processes
    with Manager() as manager:
        # Create a shared direction value
        shared_direction = manager.Value(str, "Forward")

        # Create processes for eye tracker and snake game
        eye_tracker_process = multiprocessing.Process(target=eye_tracker, args=(shared_direction,))
        snake_game_process = multiprocessing.Process(target=snake_game, args=(shared_direction,))

        # Start both processes
        eye_tracker_process.start()
        snake_game_process.start()

        # Wait for both processes to finish
        eye_tracker_process.join()
        snake_game_process.join()
