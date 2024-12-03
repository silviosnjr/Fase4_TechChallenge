import cv2
import mediapipe as mp
from deepface import DeepFace
import os
from tqdm import tqdm

# Funções para verificar ações específicas
def detect_movements(landmarks, mp_pose):
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    movements = []
    
    # Sentado
    if left_knee.y > left_ankle.y and right_knee.y > right_ankle.y:
        movements.append("Sentado")
    
    # Em pé
    elif left_knee.y < left_ankle.y and right_knee.y < right_ankle.y and left_elbow.y > left_wrist.y:
        movements.append("Em pé")

    # Acenando
    if right_wrist.y < nose.y and right_elbow.y < nose.y:
        movements.append("Acenando com a mão")

    # Mãos no rosto
    if left_wrist.y < nose.y and right_wrist.y < nose.y:
        movements.append("Mãos no rosto")

    # Cumprimento
    if abs(left_wrist.x - right_wrist.x) < 0.1 and abs(left_wrist.y - right_wrist.y) < 0.1:
        movements.append("Cumprimento")

    # Escrevendo
    if left_wrist.y > nose.y and right_wrist.y > nose.y and nose.y < left_elbow.y:
        movements.append("Escrevendo")

    return movements

# Função principal
def analyze_video(video_path, output_path, report_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    emotion_counts = {}
    movement_counts = {}
    anomaly_count = 0

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processamento de emoções
        emotions = []
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            for face in result:
                dominant_emotion = face['dominant_emotion']
                emotions.append(dominant_emotion)
                emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1

                # Desenhar retângulo ao redor do rosto
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except:
            pass

        # Processamento de movimentos
        movements = []
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            movements = detect_movements(results.pose_landmarks.landmark, mp_pose)
            for movement in movements:
                movement_counts[movement] = movement_counts.get(movement, 0) + 1

        # Análise de anomalias
        if not emotions and not movements:
            anomaly_count += 1

        # Exibir informações no frame
        if emotions:
            cv2.putText(frame, f"Emocoes: {', '.join(emotions)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if movements:
            cv2.putText(frame, f"Movimentos: {', '.join(movements)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Gerar relatório
    with open(report_path, 'w') as report:
        report.write(f"Quantidade de frames analisados: {total_frames}\n")
        report.write(f"Quantidade de anomalias detectadas: {anomaly_count}\n")
        report.write("Frequencia de emocoes detectadas:\n")
        for emotion, count in emotion_counts.items():
            report.write(f"  {emotion}: {count}\n")
        report.write("Frequência de movimentos detectados:\n")
        for movement, count in movement_counts.items():
            report.write(f"  {movement}: {count}\n")

# Caminhos para entrada, saída e relatório
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
output_video_path = os.path.join(script_dir, 'output_video.mp4')
report_path = os.path.join(script_dir, 'report.txt')

# Analisar o vídeo
analyze_video(input_video_path, output_video_path, report_path)
