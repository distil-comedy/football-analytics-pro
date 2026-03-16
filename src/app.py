import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Pro-Edition Configuration ---
st.set_page_config(page_title="PRO Edition Analytics", page_icon="⚽", layout="wide")

# Custom CSS for the Pro Dashboard
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background: #161b22; border: 1px solid #30363d; border-radius: 10px; }
    .main-title { font-size: 3rem; font-weight: 800; color: #e3b341; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

def classify_team(player_crop):
    hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(mask)
    
    if yellow_pixels > (player_crop.shape[0] * player_crop.shape[1] * 0.15):
        return "Al-Nassr", (0, 255, 255) 
    else:
        return "Al-Khaleej", (0, 255, 0)

def main():
    st.markdown('<h1 class="main-title">Real Time Football Match Analytics - PRO Edition</h1>', unsafe_allow_html=True)
    st.divider()

    model = YOLO("models/best.pt")
    uploaded_file = st.file_uploader("📂 Upload HD Footage (Up to 1GB)", type=['mp4', 'mov'])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        
        col_vid, col_stats = st.columns([3, 1])
        st_frame = col_vid.empty()
        
        with col_stats:
            st.markdown("### 📊 Live Analytics")
            m_nassr = st.metric("Al-Nassr Players", "0")
            m_khaleej = st.metric("Al-Khaleej Players", "0")
            m_fps = st.metric("Processing Speed", "0 FPS")
            heatmap_placeholder = st.empty()

        if st.button("🚀 EXECUTE ELITE ANALYSIS"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prog_bar = st.progress(0)
            
            # Data storage for Heatmap
            tracking_history = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                t1 = time.time()
                results = model.track(frame, persist=True, device=0, verbose=False) #
                t2 = time.time()
                
                nassr_count = 0
                khaleej_count = 0
                
                if results[0].boxes:
                    for box in results[0].boxes:
                        if int(box.cls[0]) == 0: # Player
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Team Classification
                            player_crop = frame[y1:y2, x1:x2]
                            if player_crop.size > 0:
                                team, color = classify_team(player_crop)
                                if team == "Al-Nassr": nassr_count += 1
                                else: khaleej_count += 1
                                
                                # Store coordinates for heatmap
                                center_x = (x1 + x2) / 2
                                center_y = y2 # Bottom of the box represents pitch position
                                tracking_history.append({"Team": team, "x": center_x, "y": center_y})

                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, team, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                st_frame.image(frame, channels="BGR", use_column_width=True)
                
                # Update Metrics
                fps = 1/(t2-t1)
                m_nassr.metric("Al-Nassr Players", nassr_count)
                m_khaleej.metric("Al-Khaleej Players", khaleej_count)
                m_fps.metric("Processing Speed", f"{fps:.1f} FPS")
                prog_bar.progress(len(tracking_history)/(total_frames * 11) if total_frames > 0 else 0)

            cap.release()
            st.success("Match Analysis Exported.")

            # --- Generate Heatmap ---
            if tracking_history:
                st.divider()
                st.subheader("🏟️ Tactical Heatmap: Pitch Dominance")
                df = pd.DataFrame(tracking_history)
                
                # High-Profile Heatmap using Plotly
                fig = px.density_contour(df, x="x", y="y", color="Team", 
                                        title="Territory Control Map",
                                        template="plotly_dark",
                                        nbinsx=30, nbinsy=30)
                fig.update_traces(contours_coloring="fill", contours_showlabels=False)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()