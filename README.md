#  AI Smart Traffic Monitoring System

This project uses YOLOv8 and Streamlit to detect and analyze traffic from video.

## Features
- Real-time vehicle detection
- Traffic density monitoring
- Vehicles in frame counter
- Vehicles passed counter
- AI dashboard using Streamlit
- Traffic analytics data logging

## Technologies Used
- Python
- YOLOv8
- Computer Vision
- Streamlit
- OpenCV
- Pandas

## Project Structure
traffic-ai-project
│
├── app.py
├── traffic_video.mp4
├── yolov8n.pt
├── traffic_data.csv
├── requirements.txt
└── README.md

## Installation

1. Clone the repository

2. Install dependencies
pip install -r requirements.txt

3. Run the application
streamlit run app.py

## Output Dashboard
- Total Vehicles
- Vehicles in Frame
- Traffic Density
- Vehicles per Minute

## Future Improvements
- Traffic trend analysis
- Congestion alert system
- Smart city traffic analytics