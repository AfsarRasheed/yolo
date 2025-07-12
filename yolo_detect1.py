import os
import sys
import argparse
import glob
import time
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Item info: class name -> price + calories
item_info = {
    'mms_peanut': {"price": 15.0, "calories": 180},
    'mms_regular': {"price": 10.0, "calories": 160},
    'airheads': {"price": 12.0, "calories": 140},
    'gummy_worms': {"price": 20.0, "calories": 200},
    'milky_way': {"price": 25.0, "calories": 220},
    'nerds': {"price": 8.0, "calories": 120},
    'skittles': {"price": 10.0, "calories": 160},
    'snickers': {"price": 25.0, "calories": 250},
    'starbust': {"price": 10.0, "calories": 150},
    'three_musketeers': {"price": 18.0, "calories": 190},
    'twizzlers': {"price": 14.0, "calories": 170},
}

# PDF generator

def generate_pdf_bill(cart_items, item_info, total_price, total_calories):
    filename = "final_bill.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 750, "📟 Smart Checkout Bill")
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 690, "Item")
    c.drawString(250, 690, "Qty")
    c.drawString(320, 690, "Price")
    c.drawString(400, 690, "Calories")
    y = 670
    c.setFont("Helvetica", 12)
    for item, qty in cart_items.items():
        if item in item_info:
            price = item_info[item]["price"]
            cal = item_info[item]["calories"]
            c.drawString(50, y, f"{item}")
            c.drawString(250, y, f"x{qty}")
            c.drawString(320, y, f"₹{price * qty:.2f}")
            c.drawString(400, y, f"{cal * qty} cal")
            y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y - 10, f"Total Price: ₹{total_price:.2f}")
    c.drawString(250, y - 10, f"Total Calories: {total_calories} cal")
    c.save()
    print(f"\n📆 PDF bill saved as '{filename}'")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model not found.')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names
img_ext_list = ['.jpg', '.png', '.jpeg', '.bmp']
vid_ext_list = ['.avi', '.mp4', '.mov']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = 'image' if ext in img_ext_list else 'video' if ext in vid_ext_list else sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print('Invalid source input')
    sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video', 'usb'] or not user_res:
        print('Recording requires video/camera source and resolution')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type == 'video':
    cap = cv2.VideoCapture(img_source)
elif source_type == 'usb':
    cap = cv2.VideoCapture(usb_idx)

    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

cart_items = {}
total_price = 0
total_calories = 0

while True:
    t_start = time.perf_counter()
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret:
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            label_ymin = max(ymin, 20)
            cv2.rectangle(frame, (xmin, label_ymin-20), (xmin+150, label_ymin), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

            if classname in cart_items:
                cart_items[classname] += 1
            else:
                cart_items[classname] = 1

            object_count += 1

    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    if source_type in ['video', 'usb']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record: recorder.write(frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# Summarize bill
if cart_items:
    print("\n===== \U0001f9fe Final Bill Summary =====")
    for item, qty in cart_items.items():
        price = item_info[item]['price']
        cal = item_info[item]['calories']
        item_total = qty * price
        item_cal = qty * cal
        total_price += item_total
        total_calories += item_cal
        print(f"{item:<20} x{qty:<2} @ ₹{price:.2f} = ₹{item_total:.2f}")
    print(f"Total Price: ₹{total_price:.2f}\nTotal Calories: {total_calories} cal\n============================")
    generate_pdf_bill(cart_items, item_info, total_price, total_calories)

if source_type in ['video', 'usb']: cap.release()
cv2.destroyAllWindows()
if record: recorder.release()
print(f"\nAverage FPS: {avg_frame_rate:.2f}")
