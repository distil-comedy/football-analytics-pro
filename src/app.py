import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time
import pandas as pd
import numpy as np
import os
import plotly.express as px

# ==========================================
# 🚨 CRITICAL FIX: SET PATH BEFORE IMPORT
# ==========================================
MODEL_STORAGE_PATH = r"D:\football_analytics_pro\ocr_models"
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH)

# We MUST tell Windows to use D: before EasyOCR even wakes up
os.environ["EASYOCR_MODULE_PATH"] = MODEL_STORAGE_PATH

# NOW we can import it safely
import easyocr
# ==========================================

# --- Pro-Edition Configuration ---
st.set_page_config(page_title="PRO Edition Analytics", page_icon="⚽", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=True)

# Load the reader
reader = load_ocr()

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
        return "Al-Nassr", (0, 255, 255) # Yellow
    else:
        return "Al-Khaleej", (255, 0, 0) # Blue

def main():
    st.markdown('<h1 class="main-title">Real Time Football Match Analytics - PRO Edition</h1>', unsafe_allow_html=True)
    st.divider()

    model = YOLO("models/best.pt")
    uploaded_file = st.file_uploader("📂 Upload HD Footage", type=['mp4', 'mov'])

    if uploaded_file:
        target_temp_dir = r"D:\football_analytics_pro\temp_data"
        if not os.path.exists(target_temp_dir): os.makedirs(target_temp_dir)

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=target_temp_dir)
        tfile.write(uploaded_file.read())
        
        # 🚨 THE FIX: Close the file writer so Windows releases the lock!
        tfile.close()
        
        col_vid, col_stats = st.columns([3, 1])
        st_frame = col_vid.empty()
        
        with col_stats:
            st.markdown("### 📊 Live Analytics")
            m_nassr = st.metric("Al-Nassr Players", "0")
            m_khaleej = st.metric("Al-Khaleej Players", "0")
            m_possession = st.metric("Ball Possession", "None")
            m_fps = st.metric("Processing Speed", "0 FPS")

        if st.button("🚀 EXECUTE ELITE ANALYSIS"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            prog_bar = st.progress(0)
            
            # MEMORY MODULES
            tracking_history = []
            previous_positions = {} 
            possession_stats = {"Al-Nassr": 0, "Al-Khaleej": 0, "None": 0} 
            jersey_memory = {} # {track_id: "number"}
            processed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                t1 = time.time()
                results = model.track(frame, persist=True, device=0, verbose=False)
                t2 = time.time()
                
                n_count, k_count = 0, 0
                frame_possessor_team = "None"

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    # --- A. FIND BALL & SPEED ---
                    ball_pos, ball_speed_label = None, "stationary"
                    for b, c, t_id in zip(boxes, class_ids, track_ids):
                        if model.names[c].upper() == "BALL":
                            cx_b, cy_b = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                            ball_pos = (cx_b, cy_b)
                            if t_id in previous_positions:
                                b_dist = np.sqrt((cx_b - previous_positions[t_id][0])**2 + (cy_b - previous_positions[t_id][1])**2)
                                # REFINED BALL SPEED LABELS
                                if b_dist > 15: ball_speed_label = "fast"
                                elif b_dist > 5: ball_speed_label = "medium"
                                elif b_dist > 1: ball_speed_label = "slow"
                            break
                    
                    possession_player_id = None
                    if ball_pos:
                        min_dist = 60 
                        for b, t_id, c_id in zip(boxes, track_ids, class_ids):
                            if model.names[c_id].upper() == "PLAYER":
                                p_center = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                                dist = np.sqrt((p_center[0]-ball_pos[0])**2 + (p_center[1]-ball_pos[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    possession_player_id = t_id

                    # --- B. PLAYER PROCESSING ---
                    player_data = [] 
                    for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                        x1, y1, x2, y2 = box
                        class_name = model.names[cls_id].upper()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        move_dist = 0
                        if track_id in previous_positions:
                            move_dist = np.sqrt((cx - previous_positions[track_id][0])**2 + (cy - previous_positions[track_id][1])**2)
                        previous_positions[track_id] = (cx, cy)

                        color = (255, 255, 255)
                        display_texts = [f"{class_name} #{track_id}"]

                        if class_name == "PLAYER":
                            # Action Mapping based on your annotation requirements
                            act = "standing"
                            if move_dist > 2.0: act = "running"
                            if track_id == possession_player_id:
                                act = "dribbling"
                                if ball_speed_label == "fast": act = "passing"
                            
                            display_texts.append(f"action: {act}")
                            
                            player_crop = frame[y1:y2, x1:x2]
                            if player_crop.size > 0:
                                # 1. Team Classification
                                team, t_color = classify_team(player_crop)
                                color = t_color
                                display_texts.append(f"team: {team}")
                                
                                # 2. OCR Jersey Number (Once every 30 frames to save FPS)
                                if track_id not in jersey_memory or processed_frames % 30 == 0:
                                    ocr_result = reader.readtext(player_crop, allowlist='0123456789')
                                    if ocr_result:
                                        jersey_memory[track_id] = ocr_result[0][1]
                                
                                if track_id in jersey_memory:
                                    display_texts.append(f"jersey_number: {jersey_memory[track_id]}")

                                player_data.append({'id': track_id, 'team': team, 'pos': (cx, cy)})
                                if team == "Al-Nassr": n_count += 1
                                else: k_count += 1
                                tracking_history.append({"Team": team, "x": cx, "y": y2})
                                if track_id == possession_player_id:
                                    frame_possessor_team = team
                                    color = (0, 0, 255)
                        
                        elif class_name == "BALL":
                            # REFINED: Ball only shows SPEED, not ACTION
                            display_texts.append(f"speed: {ball_speed_label}")
                            color = (255, 0, 255)
                        
                        elif class_name == "REFEREE": color = (0, 255, 0)

                        # TACKLING LOGIC
                        if class_name == "PLAYER":
                            for other in player_data:
                                if other['team'] != team and np.sqrt((cx - other['pos'][0])**2 + (cy - other['pos'][1])**2) < 35:
                                    display_texts[1] = "action: tackling"

                        # DRAW UI
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        max_w = max([cv2.getTextSize(t, font, 0.4, 1)[0][0] for t in display_texts])
                        bg_h = len(display_texts) * 15 + 5
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (x1, y1 - bg_h), (x1 + max_w + 10, y1), color, -1)
                        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                        curr_y = y1 - bg_h + 12
                        for t in display_texts:
                            cv2.putText(frame, t, (x1 + 5, curr_y), font, 0.4, (255, 255, 255), 1)
                            curr_y += 15

                    possession_stats[frame_possessor_team] += 1

                st_frame.image(frame, channels="BGR", use_container_width=True)
                
                # LIVE DASHBOARD
                fps = 1/(t2-t1) if (t2-t1) > 0 else 0
                m_nassr.metric("Al-Nassr Players", n_count)
                m_khaleej.metric("Al-Khaleej Players", k_count)
                m_possession.metric("Ball Possession", frame_possessor_team)
                m_fps.metric("Processing Speed", f"{fps:.1f} FPS")
                processed_frames += 1
                prog_bar.progress(min(processed_frames / total_frames, 1.0))

            cap.release()
            if os.path.exists(tfile.name): os.remove(tfile.name)
            st.success("Elite Analysis Complete.")

            # --- ANALYTICS ---
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if tracking_history:
                    st.subheader("🏟️ Tactical Heatmap")
                    fig = px.density_contour(pd.DataFrame(tracking_history), x="x", y="y", color="Team", template="plotly_dark")
                    fig.update_traces(contours_coloring="fill")
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("⏱️ Possession Share")
                pie_data = pd.DataFrame({"Team": ["Al-Nassr", "Al-Khaleej"], "Frames": [possession_stats["Al-Nassr"], possession_stats["Al-Khaleej"]]})
                fig_pie = px.pie(pie_data, values="Frames", names="Team", hole=0.4, template="plotly_dark", color_discrete_map={"Al-Nassr":"#E3B341", "Al-Khaleej":"#0000FF"})
                st.plotly_chart(fig_pie, use_container_width=True)

if __name__ == "__main__":
    main()