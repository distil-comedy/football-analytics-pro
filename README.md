# ⚽ Real Time Football Match Analytics - PRO Edition

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Real--time%20AI-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)
![GTX 1080 Ti](https://img.shields.io/badge/GPU-GTX%201080%20Ti-76B900.svg)

An elite Computer Vision pipeline designed for tactical football analysis. This system processes high-definition match footage to track players, classify teams (Al-Nassr vs. Al-Khaleej), and generate automated tactical intelligence.

## 🚀 Key Features

- **Custom-Trained YOLOv8 Engine**: Optimized to distinguish between Players, Referees, and the Ball with 98%+ accuracy.
- **Team Classification (HSV)**: Advanced color-space filtering to automatically identify teams based on jersey colors.
- **Tactical Heatmaps**: Automated pitch dominance mapping using Plotly to visualize territory control.
- **High-Performance Dashboard**: A professional "PRO Edition" Streamlit UI with 1GB file upload support and real-time inference metrics.
- **Motion Tracking**: Implements centroid tracking to maintain player IDs across frames.

## 🛠️ System Architecture

1. **Input Layer**: Supports up to 1GB of HD video footage via a specialized Streamlit buffer.
2. **Inference Layer**: Custom YOLOv8 weights running on **NVIDIA GTX 1080 Ti** (Average latency: 25ms/frame).
3. **Analytics Layer**: Real-time team counting and coordinate extraction for tactical mapping.
4. **Visualization Layer**: Interactive Plotly charts and live-annotated video feed.



## 📦 Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/distil-comedy/football-analytics-pro.git](https://github.com/distil-comedy/football-analytics-pro.git)
