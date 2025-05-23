import cv2
import mediapipe
import pyautogui
import time

capture_hands = mediapipe.solutions.hands.Hands()
drawing_option = mediapipe.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(1)  #select input camera

# Set camera properties for FPS and resolution 
camera.set(cv2.CAP_PROP_FPS, 60)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

x1 = y1 = x2 = y2 = 0
last_click_time = 0
click_cooldown = 1  # seconds

while True:
    ret, image = camera.read()
    if not ret or image is None:
        print("Error: Failed to grab frame")
        break

    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands:
            drawing_option.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                if id == 8:  # Index finger tip
                    mouse_x = int(screen_width / image_width * x)
                    mouse_y = int(screen_height / image_height * y)
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1, y1 = x, y

                if id == 4:  # Thumb tip
                    x2, y2 = x, y
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)

        dist = abs(y2 - y1)
        #print(dist)  

        # Click if distance is less than threshold and cooldown passed
        if dist < 40 and (time.time() - last_click_time) > click_cooldown:
            pyautogui.click()
            print("Clicked")
            last_click_time = time.time()

    cv2.imshow("Hand movement video capture", image)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to quit
        break

camera.release()
cv2.destroyAllWindows()
