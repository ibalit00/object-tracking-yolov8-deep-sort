import os
import random
import cv2
from ultralytics import YOLO
from tracker import Tracker

# Basisverzeichnis
base_dir = '/home/test/object-tracking-yolov8-deep-sort'

# Zusammenfügen des Pfades für das Eingabevideo und das Ausgabevideo
video_path = os.path.join(base_dir, 'data', 'people.mp4')
video_out_path = os.path.join(base_dir, 'out.mp4')

# Initialisiere Video-Capture und lese den ersten Frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Initialisiere das Ausgabevideo, falls erforderlich
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

# Lade YOLO-Modell und initialisiere Tracker
model = YOLO("yolov8n.pt")
tracker = Tracker()

# Farben für die Track-IDs
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Erkennungsschwelle
detection_threshold = 0.5

# Hauptschleife zum Lesen und Verarbeiten der Frames
while ret:
    # YOLO-Modell auf den Frame anwenden
    results = model(frame)

    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > detection_threshold:
                detections.append([int(x1), int(y1), int(x2), int(y2), score])

        # Aktualisiere den Tracker mit den Erkennungen
        tracker.update(frame, detections)

        # Visualisierung der Tracks vor und nach dem Update
        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            color = colors[track_id % len(colors)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    # Zeige das aktuelle Frame
    cv2.imshow("nachher", frame)

    # Schreibe das Frame ins Ausgabevideo
    cap_out.write(frame)

    # Aktualisiere das nächste Frame und beende die Schleife bei Bedarf
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

# Ressourcen freigeben
cap.release()
cap_out.release()
cv2.destroyAllWindows()
