from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def getHandMove(hand_landmarks):
    landmarks = hand_landmarks.landmark
    if all([landmarks[i].y < landmarks[i+3].y for i in range(9,20,4)]): 
        return "rock"
    elif landmarks[13].y < landmarks[16].y and landmarks[17].y < landmarks[20].y: 
        return "scissors"
    else: 
        return "paper"

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    frame_width = 1280
    frame_height = 720
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    bg_image = cv2.imread("./static/assets/img/bg.png")

    font_path = './static/assets/font/rubik.ttf'
    font = ImageFont.truetype(font_path, size=30)

    p1_move = p2_move = None
    gameText = ""
    gameText1 = ""
    gameText2 = ""
    clock = 0
    hands_detected = False

    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            player1_hand = None
            player2_hand = None

            bg_image_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))

            mask = np.zeros_like(frame, dtype=np.uint8)

            if result.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame,
                                              hand_landmarks,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

                    box_top_left_1 = (79, 280)
                    box_bottom_right_1 = (box_top_left_1[0] + 400, box_top_left_1[1] + 400)
                    cv2.rectangle(mask, box_top_left_1, box_bottom_right_1, (255, 255, 255), thickness=cv2.FILLED)

                    width = box_bottom_right_1[0] - box_top_left_1[0]

                    box_top_left_2 = (frame_width - width - 79 - 28, 280)
                    box_bottom_right_2 = (frame_width - 107, box_top_left_2[1] + 400)
                    cv2.rectangle(mask, box_top_left_2, box_bottom_right_2, (255, 255, 255), thickness=cv2.FILLED)

                    frame[box_top_left_1[1]:box_bottom_right_1[1], box_top_left_1[0]:box_bottom_right_1[0]] = frame[box_top_left_1[1]:box_bottom_right_1[1], box_top_left_1[0]:box_bottom_right_1[0]]
                    frame[box_top_left_2[1]:box_bottom_right_2[1], box_top_left_2[0]:box_bottom_right_2[0]] = frame[box_top_left_2[1]:box_bottom_right_2[1], box_top_left_2[0]:box_bottom_right_2[0]]

                    frame[~mask.astype(bool)] = bg_image_resized[~mask.astype(bool)]

                    box_color = (255, 255, 255)
                    box_thickness = 2
                    cv2.rectangle(frame, box_top_left_1, box_bottom_right_1, box_color, box_thickness)
                    cv2.rectangle(frame, box_top_left_2, box_bottom_right_2, box_color, box_thickness)

                    if box_top_left_1[0] > frame.shape[1] / 2:
                        player1_hand = hand_landmarks
                    else:
                        player2_hand = hand_landmarks

            if player1_hand and player2_hand:
                p1_move = getHandMove(player1_hand)
                p2_move = getHandMove(player2_hand)

            if result.multi_hand_landmarks:
                hands_detected = True
                clock += 1
                if 0 <= clock < 10:
                    success = True
                    gameText = "Ready?"
                elif clock < 15: gameText = "3..."
                elif clock < 30: gameText = "2..."
                elif clock < 45: gameText = "1..."
                elif clock < 60: gameText = "GO!!!"
                elif clock == 75:
                    hls = result.multi_hand_landmarks
                    if hls and len(hls) == 2:
                        p1_move = getHandMove(hls[0])
                        p2_move = getHandMove(hls[1])
                    else:
                        success = False
                elif clock < 115:
                    if success:
                        gameText1 = f"Player 1 : {p1_move}"
                        gameText2 = f"Player 2 : {p2_move}"
                        if p1_move == p2_move: gameText = f"Game is tied."
                        elif p1_move == "paper" and p2_move == "rock": gameText = f"Player 1 wins."
                        elif p1_move == "rock" and p2_move == "scissors": gameText = f"Player 1 wins."
                        elif p1_move == "scissors" and p2_move == "paper": gameText = f"Player 1 wins."
                        else: gameText = f"Player 2 wins."
                    else:
                        gameText = "Didn't play properly!"
            else:
                hands_detected = False
                clock = 0

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            draw.text((80, 100), gameText1, font=font, fill=(0, 255, 255))
            draw.text((80, 130), gameText2, font=font, fill=(0, 255, 255))
            draw.text((80, 170), gameText, font=font, fill=(0, 255, 255))

            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            if not result.multi_hand_landmarks:
                bg_no_hand = cv2.imread('./static/assets/img/bg_nohands.png')
                frame = bg_no_hand

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run()
