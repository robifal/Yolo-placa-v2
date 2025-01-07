import cv2
import numpy as np
from ultralytics import YOLO

model_path = '/home/estagiario-tarde/Área de Trabalho/Yolo-placa-v2/Placas-dec 2.v1i.yolov11/runs/detect/train2/weights/best.pt'
model = YOLO(model_path)

ultimo_texto_placa = "Nenhuma placa detectada"
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

cv2.namedWindow("Detecção de Placas", cv2.WINDOW_NORMAL)

# Ajustes de exposição e brilho
cap.set(cv2.CAP_PROP_EXPOSURE, 0.5)  # Ajuste a exposição
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)  # Ajuste o brilho

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam.")
        break

    # Aplicando equalização do histograma para melhorar o contraste
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    frame_yuv[:, :, 0] = cv2.equalizeHist(frame_yuv[:, :, 0])  # Equaliza a componente de brilho (Y)
    frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

    # Ajustando o contraste
    frame = cv2.convertScaleAbs(frame, alpha=0.8,beta==0)  # Aumentando o contraste

    # Redimensionamento para melhorar a detecção
    scale_percent = 150
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convertendo a imagem para RGB para a detecção
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Ajuste dos thresholds de confiança e IOU
    results = model(img, conf=0.3, iou=0.3)

    confidence_threshold = 0.1
    detections = results[0].boxes
    detected = False

    for box in detections:
        conf = box.conf[0].cpu().numpy()
        if conf < confidence_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().numpy())

        if cls in model.names:
            label = f"{model.names[cls]} {conf:.2f}"
            ultimo_texto_placa = model.names[cls]
            detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Placa: {ultimo_texto_placa}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if not detected:
        cv2.putText(frame, "Nenhuma placa detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Detecção de Placas", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
