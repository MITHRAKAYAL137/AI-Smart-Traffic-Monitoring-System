import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import time

#  PAGE 
st.set_page_config(layout="wide")
st.title("🚦 AI Smart City Traffic Analyzer")

left, right = st.columns([1,2])

# MODEL 
model = YOLO("yolov8n.pt")

#  VIDEO 
cap = cv2.VideoCapture("traffic_video.mp4")

frame_skip = 6
frame_count = 0

line_position = 250

vehicle_count = 0
counted_ids = set()

start_time = time.time()

data_log = []

# DASHBOARD
total_box = left.empty()
live_box = left.empty()
traffic_box = left.empty()
vpm_box = left.empty()

video_placeholder = right.empty()

#  LOOP
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    frame = cv2.resize(frame,(640,360))

    results = model.track(frame, persist=True, conf=0.6)

    vehicles_in_frame = 0

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu()
        ids = results[0].boxes.id.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, ids, classes):

            if int(cls) not in [1,2,3,5,7]:
                continue

            x1,y1,x2,y2 = map(int,box)

            cy = int((y1+y2)/2)

            vehicles_in_frame += 1

            if track_id not in counted_ids and cy > line_position:

                counted_ids.add(track_id)
                vehicle_count += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # TRAFFIC STATUS 
    if vehicles_in_frame < 5:
        traffic_status = "LOW 🟢"
    elif vehicles_in_frame < 12:
        traffic_status = "MEDIUM 🟡"
    else:
        traffic_status = "HIGH 🔴"

    

    #  DATA
    data_log.append({
        "Time": round(time.time() - start_time,2),
        "Vehicles_in_frame": vehicles_in_frame,
        "Total_Vehicles": vehicle_count
    })

    #  DASHBOARD 
    total_box.metric("Total Vehicles Passed", vehicle_count)
    live_box.metric("Current Vehicles", vehicles_in_frame)
    traffic_box.metric("Traffic Condition", traffic_status)

    cv2.line(frame,(0,line_position),(frame.shape[1],line_position),(255,0,0),3)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    video_placeholder.image(frame)

cap.release()

# SAVE DATA
df = pd.DataFrame(data_log)
df.to_csv("traffic_data.csv", index=False)

st.success("Traffic data saved to traffic_data.csv")