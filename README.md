# 🚗 YOLOv5 Driver Behavior & Fatigue Detection

## 📌 Project Overview
This project is based on **YOLOv5** and **dlib**, designed to detect **unsafe driving behaviors** (such as smoking, drinking water, and using a phone), **recognize driver emotions** (such as anger and surprise), and **detect driver fatigue** (such as eye closure and yawning). The model can be applied to intelligent monitoring systems to enhance road safety.

## ⚙️ Features
- **Real-time Detection**: Fast and accurate behavior recognition using YOLOv5 and dlib
- **Multi-Class Recognition**: Supports multiple dangerous driving behaviors, emotions, and fatigue detection
- **Fatigue Monitoring**: Uses dlib to track eye blink frequency and yawning to detect drowsy drivers
## 🚀 Installation & Execution

### 1️⃣ **Clone the Project**
```bash
git git@github.com:LiZihao98/ACP_Behavior-EmotionDetection_Dlib-Yolov5.git
```

### 2️⃣ **Create Virtual Environment & Install Dependencies**
```bash
conda create -n yolov5-env python=3.8 -y
conda activate yolov5-env
pip install -r requirements.txt
```

### 3️⃣ Run the system
```bash
python application.py
```
you will see the following user interface, press Start Video detection to run the system.
![Logo](weight\ui1.png)

## 📢License

- **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE) file for more details.
- **Enterprise License**: Designed for commercial use, this license permits seamless integration of Ultralytics software and AI models into commercial goods and services, bypassing the open-source requirements of AGPL-3.0. If your scenario involves embedding our solutions into a commercial offering, reach out through [Ultralytics Licensing](https://www.ultralytics.com/license).

## Conclusion
This project integrates YOLOv5 for driver behavior and emotion detection, along with dlib-based fatigue detection. The system can detect unsafe driving behaviors, emotional states, and fatigue-related signs such as eye closure and yawning, making it highly applicable in smart transportation and driver safety monitoring.

If you're interested in this project, feel free to Star ⭐, Fork 🍴, and Contribute 💡!
