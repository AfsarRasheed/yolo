# 🍬 Real-Time Candy Detection With Calorie Tracking and Automated Checkout 

## Train YOLO Models

**Option 1. With Google Colab**

Click below to acces a Colab notebook for training YOLO models. It makes training a custom YOLO model as easy as uploading an image dataset and running a few blocks of code.

<a href="https://colab.research.google.com/drive/1U_NUt5hiEhyv2O_4YI1VK510fSwuRCr3?usp=sharing " target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## 🏷️ Introduction

This project presents a **smart object detection pipeline** powered by the **YOLOv8** deep learning model, designed to automatically **classify and detect 11 types of candies** in both images and videos. The main goal is to create a **virtual "smart cart"** that not only recognizes each candy but also keeps track of all candies detected, dynamically computing the **total price** and **caloric content** for a seamless and interactive retail experience.

---

### 🔍 What Does This Project Do?

- **Detects Candies:** Accurately identifies 11 different candy types in real time from various sources (images, live camera feeds, videos).
- **Tracks Selections:** Adds recognized candies to a "cart", ensuring that only stable, confirmed detections are counted.
- **Calculates Total:** Continuously updates and displays the total cost and calories based on candies detected.
- **Visualizes Results:** Overlays detection boxes, labels, and cart summaries right onto the video/image feeds for easy understanding.

---

### 🚀 Key Features

- **Model:** Utilizes Ultralytics' YOLOv8, a highly efficient, state-of-the-art object detection model.
- **Dataset:** Consists of labeled images featuring 11 candy varieties (e.g., M&Ms, Skittles, Airheads, etc.).
- **Environment:** Runs entirely in **Google Colab** for free GPU acceleration (Tesla T4), ensuring rapid training and inference.
- **User-Friendly:** Includes visual overlays, real-time stats, and interactive controls for practical usability.

---

### 📚 How It Works

1. **Setup:**
   - All code executes in Google Colab, leveraging free access to a Tesla T4 GPU.
   - Installation of required libraries including Ultralytics YOLOv8 and OpenCV.

2. **Detection Pipeline:**
   - **Input:** Accepts images, videos, webcam feeds, or entire image folders.
   - **YOLOv8 Model:** Loaded and used to make predictions frame-by-frame.
   - **Candy Metadata:** Maintains a lookup for price and calories for each candy type.
   - **Smart Tracking:** Uses a robust tracking mechanism to guarantee only consistent, stable detections are added to the cart.

3. **Visualization:**
   - **BBox & Labels:** Draws colored bounding boxes with candy names, confidence, price, and calories right on each detection.
   - **Cart Overlay:** Continuously shows a summary panel with current cart totals and a breakdown of itemized candies.
   - **Keyboard Controls:** 
     - `q` = Quit
     - `c` = Clear cart
     - `r` = Reset detection tracker
     - `p` = Screenshot the current frame

4. **Result Summary:**
   - When the session ends, a neat printout displays each candy’s count, its subtotal, and the overall cart price/calories.

---

### 🍭 Example Use Case

Imagine using a camera-enabled checkout system:
- Place candies on the tray,
- The system detects and displays what you’ve picked,
- Instantly see what you’ll pay and what you’ll consume,
- Enjoy a checkout process that’s fast, accurate, and fun!






