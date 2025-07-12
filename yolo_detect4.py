import os
import sys
import argparse
import glob
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import gradio as gr  # <-- import gradio here

# Define item info: calories and price per unit
item_info = {
    'MMs_peanut': {'calories': 250, 'price': 10},
    'MMs_regular': {'calories': 230, 'price': 10},
    'airheads': {'calories': 120, 'price': 12},
    'gummy_worms': {'calories': 150, 'price': 15},
    'milky_way': {'calories': 280, 'price': 20},
    'nerds': {'calories': 100, 'price': 8},
    'skittles': {'calories': 210, 'price': 10},
    'snickers': {'calories': 300, 'price': 25},
    'starbust': {'calories': 200, 'price': 15},
    'three_musketeers': {'calories': 270, 'price': 22},
    'twizzlers': {'calories': 180, 'price': 14},
}

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g., "best.pt")')
parser.add_argument('--source', required=True, help='Image source: image file, folder, video file, usb0, etc.')
parser.add_argument('--thresh', default=0.5, type=float, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='Resolution WxH for display (e.g., 640x480)')
parser.add_argument('--record', action='store_true', help='Record video output')
parser.add_argument('--stability-frames', default=15, type=int, help='Frames to wait before adding item (stability check)')
parser.add_argument('--absence-frames', default=30, type=int, help='Frames item must be absent before allowing re-detection')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
stability_frames = args.stability_frames
absence_frames = args.absence_frames

if not os.path.exists(model_path):
    print(f'ERROR: Model file {model_path} not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg','.jpeg','.png','.bmp','.JPG','.JPEG','.PNG','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension {ext}')
        sys.exit(0)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Invalid source {img_source}')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video or USB camera sources.')
        sys.exit(0)
    if not user_res:
        print('Resolution must be specified to record video.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
                (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

img_count = 0
fps_avg_len = 200
frame_rate_buffer = []
avg_frame_rate = 0
cart_items = {}

class ItemTracker:
    def __init__(self, stability_frames=15, absence_frames=30):
        self.stability_frames = stability_frames
        self.absence_frames = absence_frames
        self.detection_history = defaultdict(deque)
        self.last_seen = defaultdict(int)
        self.added_items = set()
        self.frame_count = 0

    def update_frame(self):
        self.frame_count += 1

    def process_detections(self, detected_items):
        items_to_add = []
        for item, count in detected_items.items():
            self.detection_history[item].append(count)
            self.last_seen[item] = self.frame_count
            if len(self.detection_history[item]) > self.stability_frames:
                self.detection_history[item].popleft()

        for item in detected_items:
            if self.is_stable_detection(item) and item not in self.added_items:
                recent_counts = list(self.detection_history[item])
                if recent_counts:
                    stable_count = max(set(recent_counts), key=recent_counts.count)
                    items_to_add.append((item, stable_count))
                    self.added_items.add(item)

        items_to_remove = []
        for item in self.added_items:
            if self.frame_count - self.last_seen[item] > self.absence_frames:
                items_to_remove.append(item)

        for item in items_to_remove:
            self.added_items.remove(item)
            self.detection_history[item].clear()

        return items_to_add

    def is_stable_detection(self, item):
        if item not in self.detection_history:
            return False
        history = self.detection_history[item]
        if len(history) < self.stability_frames:
            return False
        non_zero_detections = sum(1 for count in history if count > 0)
        return non_zero_detections / len(history) >= 0.8

    def get_detection_status(self, item):
        if item not in self.detection_history:
            return "New"
        history = self.detection_history[item]
        if len(history) < self.stability_frames:
            return f"Tracking {len(history)}/{self.stability_frames}"
        if item in self.added_items:
            return "Added"
        if self.is_stable_detection(item):
            return "Ready to Add"
        return "Unstable"

tracker = ItemTracker(stability_frames, absence_frames)

# --- Main loop for video or images ---

def main_loop():
    global img_count, cart_items

    while True:
        t_start = time.perf_counter()
        tracker.update_frame()

        if source_type in ['image','folder']:
            if img_count >= len(imgs_list):
                break
            frame = cv2.imread(imgs_list[img_count])
            img_count += 1
        elif source_type in ['video', 'usb']:
            ret, frame = cap.read()
            if not ret:
                break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        results = model(frame, verbose=False)
        detections = results[0].boxes

        current_detected_items = defaultdict(int)
        detection_boxes = []

        for i in range(len(detections)):
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf >= min_thresh:
                current_detected_items[classname] += 1
                detection_boxes.append({
                    'bbox': (xmin, ymin, xmax, ymax),
                    'class': classname,
                    'confidence': conf,
                    'color': bbox_colors[classidx % len(bbox_colors)]
                })

        items_to_add = tracker.process_detections(current_detected_items)
        for item, count in items_to_add:
            cart_items[item] = cart_items.get(item, 0) + count
            print(f"Added to cart: {item} x{count}")

        for det in detection_boxes:
            xmin, ymin, xmax, ymax = det['bbox']
            classname = det['class']
            conf = det['confidence']
            color = det['color']

            info = item_info.get(classname, {'price': 0, 'calories': 0})
            status = tracker.get_detection_status(classname)

            label = f"{classname}: {int(conf*100)}%"
            price_cal_label = f"INR {info['price']}, {info['calories']} cal"
            status_label = f"Status: {status}"

            label_ymin = max(ymin, 60)
            cv2.rectangle(frame, (xmin, label_ymin-60), (xmin + 220, label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            cv2.putText(frame, price_cal_label, (xmin, label_ymin-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
            cv2.putText(frame, status_label, (xmin, label_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

        total_price = sum(item_info.get(k, {'price': 0})['price'] * v for k, v in cart_items.items())
        total_calories = sum(item_info.get(k, {'calories': 0})['calories'] * v for k, v in cart_items.items())

        cv2.putText(frame, f'Cart Total: INR {total_price:.2f} | {total_calories} cal', (10, 20), cv2.FONT
