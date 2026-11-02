import time
import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from PIL import Image

import emotion_utils

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🇨🇭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SWISS MINIMAL DESIGN SYSTEM (INTERNATIONAL TYPOGRAPHIC STYLE)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Canvas & Base Typography */
    html, body, .stApp {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        letter-spacing: -0.015em;
    }

    /* Streamlit Main Container */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* Streamlit Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }

    /* Streamlit Sidebar (Swiss Light Neutral) */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    section[data-testid="stSidebar"] .swiss-mono-tag {
        color: #64748B !important;
    }

    /* Headings & Text Color Overrides */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
    }

    p, span, label, div {
        color: #1E293B;
    }

    /* Swiss Masthead Layout */
    .swiss-masthead {
        border-bottom: 2px solid #000000;
        padding-bottom: 22px;
        margin-bottom: 28px;
    }

    .swiss-top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .swiss-brand-badge {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .swiss-cross-box {
        background-color: #DC2626;
        color: #FFFFFF !important;
        font-weight: 900;
        font-size: 0.72rem;
        padding: 3px 7px;
        border-radius: 2px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.05em;
    }

    .swiss-kicker {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #64748B !important;
    }

    .swiss-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #DC2626 !important;
        border: 1px solid #FCA5A5;
        background: #FEF2F2;
        padding: 3px 10px;
        border-radius: 2px;
        letter-spacing: 0.06em;
    }

    .swiss-dot {
        width: 6px;
        height: 6px;
        background-color: #DC2626;
        border-radius: 50%;
    }

    .swiss-title {
        font-size: 2.85rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.04em !important;
        color: #000000 !important;
        text-transform: uppercase;
        margin: 0 !important;
        line-height: 1.05;
    }

    .swiss-lead {
        font-family: 'Inter', sans-serif;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        color: #64748B !important;
        margin-top: 8px !important;
        margin-bottom: 0 !important;
        letter-spacing: -0.01em;
    }

    /* Swiss Cards (Crisp Minimalist Rectangular Grid) */
    .swiss-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 2px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #000000 !important;
        letter-spacing: -0.04em;
        margin-top: 4px;
    }

    /* Swiss Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background-color: #FFFFFF !important;
        padding: 0 !important;
        border-bottom: 2px solid #E2E8F0 !important;
        border-radius: 0 !important;
        margin-bottom: 24px !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        padding: 12px 26px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #64748B !important;
        transition: all 0.12s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #000000 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom: 2px solid #DC2626 !important;
        background-color: transparent !important;
    }

    /* Buttons (Swiss Red Primary, Crisp Minimalist 2px Radius) */
    button[kind="primary"] {
        background-color: #DC2626 !important;
        border: 1px solid #B91C1C !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        border-radius: 2px !important;
        padding: 8px 18px !important;
        box-shadow: none !important;
    }

    button[kind="primary"]:hover {
        background-color: #B91C1C !important;
        border-color: #991B1B !important;
    }

    button[kind="secondary"] {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        color: #0F172A !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border-radius: 2px !important;
        box-shadow: none !important;
    }

    button[kind="secondary"]:hover {
        background-color: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }

    /* Minimal Technical Monospace Tag */
    .swiss-mono-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B !important;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    /* Streamlit Expander */
    .streamlit-expanderHeader {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 2px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        color: #0F172A !important;
    }

    /* Streamlit File Uploader */
    div[data-testid="stFileUploader"] {
        border: 1px dashed #CBD5E1 !important;
        border-radius: 2px !important;
        background: #F8FAFC !important;
        padding: 12px !important;
    }

    /* Slider track accents */
    div[data-baseweb="slider"] {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# RESOURCE CACHING (MODEL & FACE DETECTOR)
# ==========================================
@st.cache_resource(show_spinner=False)
def get_model():
    """Cached TensorFlow emotion classification model."""
    return emotion_utils.load_emotion_model("models/emotion_cnn.keras")


@st.cache_resource(show_spinner=False)
def get_face_detector():
    """Cached OpenCV Haar cascade classifier."""
    return emotion_utils.load_face_detector()


# Initialize neural model
with st.spinner("Initializing Swiss Neural Pipeline..."):
    model = get_model()
    face_detector = get_face_detector()

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

# ==========================================
# SIDEBAR CONFIGURATION (SWISS MINIMAL)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 16px; border-bottom: 2px solid #000000; margin-bottom: 20px;">
        <div class="swiss-brand-badge" style="margin-bottom: 6px;">
            <span class="swiss-cross-box">CH</span>
            <span class="swiss-kicker">VISION LAB // FER-2.4</span>
        </div>
        <h3 style="margin: 0; font-weight: 900; font-size: 1.25rem; letter-spacing: -0.03em; color: #000000; text-transform: uppercase;">
            Facial Emotion Recognition
        </h3>
        <div style="margin-top: 10px;">
            <span class="swiss-status"><span class="swiss-dot"></span> PIPELINE ONLINE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Optimized detection parameters
    scale_factor = 1.25
    min_neighbors = 5

    with st.expander("ℹ️ Model Architecture", expanded=False):
        st.markdown("""
        - **Input Layer:** 48×48 Normalized Grayscale Tensor
        - **Classifier:** Deep Convolutional Neural Network
        - **Native File:** `emotion_cnn.keras`
        - **Classes:** 7 Emotional Affect Dimensions
        - **Face Detector:** OpenCV Haar Feature Cascade
        """)

    st.markdown("---")
    st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 10px;">AFFECT TAXONOMY (7 CLASSES)</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    emotions_list = list(emotion_utils.EMOTION_META.items())
    for i, (name, meta) in enumerate(emotions_list):
        col = cols[i % 2]
        col.markdown(
            f"<div style='margin-bottom: 6px; font-size: 0.85rem;'>"
            f"<span>{meta['emoji']}</span> "
            f"<strong style='color: #0F172A;'>{meta['label'].upper()}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    if st.button("Clear Detection Records", use_container_width=True):
        st.session_state.detection_history = []
        st.rerun()


# ==========================================
# SWISS MASTHEAD
# ==========================================
st.markdown("""
<div class="swiss-masthead">
    <div class="swiss-top-bar">
        <div class="swiss-brand-badge">
            <span class="swiss-cross-box">CH</span>
            <span class="swiss-kicker">INTERNATIONAL TYPOGRAPHIC SYSTEM // 2026</span>
        </div>
        <div class="swiss-status">
            <span class="swiss-dot"></span> LIVE INFERENCE ENGINE READY
        </div>
    </div>
    <h1 class="swiss-title">Facial Emotion Recognition</h1>
    <p class="swiss-lead">Real-time deep learning facial affect analysis, multi-face classification, and affective telemetry.</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# HELPER FUNCTIONS FOR UI VISUALIZATIONS
# ==========================================
def render_emotion_bars(probabilities: dict, threshold: float = 0.0):
    """Render authentic Swiss minimal precision progress bars."""
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_emotion = sorted_probs[0][0] if sorted_probs else None

    for emotion, prob in sorted_probs:
        meta = emotion_utils.EMOTION_META.get(emotion, {"label": emotion.capitalize(), "emoji": "", "color": "#DC2626"})
        is_top = (emotion == top_emotion)
        bar_color = "#DC2626" if is_top else "#334155"
        text_weight = "700" if is_top else "500"
        text_color = "#000000" if is_top else "#475569"

        st.markdown(
            f"""
            <div style="margin-bottom: 9px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline; font-size: 0.84rem; margin-bottom: 3px;">
                    <span style="font-weight: {text_weight}; color: {text_color};">
                        {meta['emoji']} {meta['label'].upper()}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 700; color: {'#DC2626' if is_top else '#64748B'};">
                        {prob:.1f}%
                    </span>
                </div>
                <div style="background: #E2E8F0; border-radius: 1px; height: 5px; width: 100%; overflow: hidden;">
                    <div style="background: {bar_color}; width: {max(prob, 1.2):.1f}%; height: 100%; border-radius: 1px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def create_radar_chart(probabilities: dict):
    """Generate a clean, light Swiss Plotly polar radar chart of emotional dimensions."""
    categories = [emotion_utils.EMOTION_META[k]["label"].upper() for k in emotion_utils.CLASS_NAMES]
    values = [probabilities.get(k, 0.0) for k in emotion_utils.CLASS_NAMES]
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(220, 38, 38, 0.08)',
        line=dict(color='#DC2626', width=2),
        marker=dict(size=4, color='#DC2626'),
        hoverinfo='theta+r'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(100, max(values) + 10)],
                tickfont=dict(size=8, color='#64748B', family='JetBrains Mono'),
                gridcolor='#E2E8F0',
                linecolor='#CBD5E1'
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color='#0F172A', family='Inter', weight='bold'),
                gridcolor='#E2E8F0',
                linecolor='#CBD5E1'
            ),
            bgcolor='#FFFFFF'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        height=280
    )
    return fig


def log_detection(face_data: dict, source: str):
    """Log detection to session history (latest / last come displayed first)."""
    st.session_state.detection_history.insert(0, {
        "Timestamp": datetime.now().strftime("%H:%M:%S"),
        "Source": source,
        "Face ID": face_data["face_id"],
        "Dominant Emotion": face_data["dominant_emotion"].upper(),
        "Confidence (%)": round(face_data["confidence"], 1),
        "Valence": face_data["metadata"]["valence"].upper()
    })


# ==========================================
# MAIN NAVIGATION TABS (SAMPLE GALLERY REMOVED)
# ==========================================
tab_webcam, tab_upload, tab_analytics = st.tabs([
    "01 // Live Camera",
    "02 // Image Analysis",
    "03 // Session Logs"
])

# -------------------------------------------------------------
# TAB 1: LIVE CAMERA STUDIO
# -------------------------------------------------------------
with tab_webcam:
    cam_mode = st.radio(
        "Camera Mode Selection",
        ["Real-Time Webcam Feed (Continuous)", "Instant Snapshot Capture (Browser Camera)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if cam_mode == "Real-Time Webcam Feed (Continuous)":
        col_stream, col_metrics = st.columns([3, 2])

        with col_stream:
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                start_btn = st.button("Start Live Stream", use_container_width=True, type="primary")
            with btn_col2:
                stop_btn = st.button("Stop Stream", use_container_width=True)

            stream_placeholder = st.empty()

            if start_btn:
                st.session_state.webcam_running = True
            if stop_btn:
                st.session_state.webcam_running = False

        with col_metrics:
            metrics_placeholder = st.empty()
            if not st.session_state.webcam_running:
                metrics_placeholder.markdown("""
                <div class="swiss-card" style="border-left: 4px solid #000000;">
                    <div class="metric-label">STREAM STATUS // STANDBY</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #000000; margin: 6px 0 10px 0;">Camera Inactive</div>
                    <p style="font-size: 0.88rem; color: #64748B; margin: 0; line-height: 1.5;">
                        Click <strong>Start Live Stream</strong> to activate continuous neural inference. Face bounding boxes and affect telemetry will update in real time.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access webcam device. Please verify camera permissions.")
                st.session_state.webcam_running = False
            else:
                prev_time = time.time()
                frame_count = 0
                last_logged_emotion = None
                last_log_time = 0

                while st.session_state.webcam_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Webcam feed disconnected.")
                        break

                    # Process frame
                    result = emotion_utils.process_facial_emotions(
                        img_bgr=frame,
                        model=model,
                        face_detector=face_detector,
                        scale_factor=scale_factor,
                        min_neighbors=min_neighbors
                    )

                    # Calculate FPS
                    curr_time = time.time()
                    fps = 1.0 / (curr_time - prev_time + 1e-6)
                    prev_time = curr_time
                    frame_count += 1

                    # Render live frame
                    stream_placeholder.image(
                        result["annotated_rgb"],
                        channels="RGB",
                        use_container_width=True,
                        caption=f"01.1 // LIVE FEED • {result['face_count']} FACE(S) DETECTED • {fps:.1f} FPS"
                    )

                    # Update metrics container: latest (last come) is ALWAYS displayed first
                    with metrics_placeholder.container():
                        st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 8px;">01.2 // AFFECT TELEMETRY (LATEST FIRST)</div>', unsafe_allow_html=True)
                        if result["faces"]:
                            for idx, face in enumerate(result["faces"]):
                                meta = face["metadata"]
                                face_title = f"Face #{face['face_id']}" if len(result["faces"]) > 1 else "Primary Face"
                                st.markdown(
                                    f"""
                                    <div class="swiss-card" style="border-left: 4px solid #DC2626;">
                                        <div style="display: flex; justify-content: space-between; align-items: center;">
                                            <span class="metric-label">{face_title} Dominant Emotion</span>
                                            <span class="swiss-status" style="font-size: 0.68rem; padding: 1px 6px;">
                                                <span class="swiss-dot"></span> LIVE
                                            </span>
                                        </div>
                                        <div style="font-size: 2.1rem; font-weight: 900; color: #000000; letter-spacing: -0.03em; margin: 4px 0;">
                                            {meta['emoji']} {meta['label'].upper()}
                                        </div>
                                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #64748B;">
                                            CONFIDENCE: <strong style="color: #000000;">{face['confidence']:.1f}%</strong> • {meta['valence'].upper()} VALENCE
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                render_emotion_bars(face["probabilities"])

                            # Log emotion transitions
                            top_face = result["faces"][0]
                            if (top_face["dominant_emotion"] != last_logged_emotion or (curr_time - last_log_time) > 4.0):
                                log_detection(top_face, source="Webcam Stream")
                                last_logged_emotion = top_face["dominant_emotion"]
                                last_log_time = curr_time
                        else:
                            st.markdown("""
                            <div class="swiss-card" style="border-left: 4px solid #94A3B8;">
                                <div class="metric-label">STATUS // SCANNING</div>
                                <div style="font-size: 1rem; color: #64748B; margin-top: 6px;">
                                    No face detected in current frame. Position face directly toward the camera.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    time.sleep(0.01)

                cap.release()
        else:
            stream_placeholder.info("Webcam in standby. Click Start Live Stream to begin.")

    else:
        # Instant Photo Snapshot
        st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 8px;">01.2 // BROWSER CAMERA SNAPSHOT</div>', unsafe_allow_html=True)
        camera_photo = st.camera_input("Capture photo for facial affect inference")

        if camera_photo is not None:
            pil_img = Image.open(camera_photo)
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Processing snapshot..."):
                result = emotion_utils.process_facial_emotions(
                    img_bgr=img_bgr,
                    model=model,
                    face_detector=face_detector,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors
                )

            col_res_img, col_res_data = st.columns([3, 2])
            with col_res_img:
                st.image(
                    result["annotated_rgb"],
                    caption=f"SNAPSHOT OUTPUT // {result['face_count']} FACE(S) IDENTIFIED",
                    use_container_width=True
                )

            with col_res_data:
                if result["face_count"] == 0:
                    st.warning("No face detected in photo. Ensure adequate lighting and frontal face orientation.")
                else:
                    for face in result["faces"]:
                        log_detection(face, source="Camera Snapshot")
                        meta = face["metadata"]
                        st.markdown(
                            f"""
                            <div class="swiss-card" style="border-left: 4px solid #DC2626;">
                                <div class="metric-label">Face #{face['face_id']} Classification</div>
                                <div style="font-size: 2rem; font-weight: 900; color: #000000; margin: 4px 0;">
                                    {meta['emoji']} {meta['label'].upper()}
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #64748B;">
                                    CONFIDENCE: <strong style="color: #000000;">{face['confidence']:.1f}%</strong> • {meta['valence'].upper()} VALENCE
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        render_emotion_bars(face["probabilities"])
                        st.plotly_chart(create_radar_chart(face["probabilities"]), use_container_width=True)


# -------------------------------------------------------------
# TAB 2: IMAGE ANALYSIS & MULTI-FACE INSPECTION
# -------------------------------------------------------------
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload image file (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload frontal portrait or group photographs for multi-face affect recognition."
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        img_np = np.array(pil_img)
        if len(img_np.shape) == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        elif img_np.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        with st.spinner("Executing neural inference on uploaded image..."):
            start_t = time.time()
            result = emotion_utils.process_facial_emotions(
                img_bgr=img_bgr,
                model=model,
                face_detector=face_detector,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors
            )
            proc_time = (time.time() - start_t) * 1000

        # Minimal KPI Row
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">02.1 // FACES DETECTED</div>
                    <div class="metric-value">{result['face_count']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi2:
            top_em = result["faces"][0]["dominant_emotion"].upper() if result["faces"] else "N/A"
            top_emoji = result["faces"][0]["metadata"]["emoji"] if result["faces"] else "—"
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">02.2 // PRIMARY AFFECT</div>
                    <div class="metric-value" style="font-size: 1.6rem;">{top_emoji} {top_em}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi3:
            conf = f"{result['faces'][0]['confidence']:.1f}%" if result["faces"] else "N/A"
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">02.3 // CONFIDENCE</div>
                    <div class="metric-value">{conf}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">02.4 // LATENCY</div>
                    <div class="metric-value">{proc_time:.0f} ms</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col_img, col_details = st.columns([3, 2])

        with col_img:
            st.image(
                result["annotated_rgb"],
                caption=f"INSPECTION OUTPUT // {result['face_count']} FACE(S) IDENTIFIED",
                use_container_width=True
            )

            # Download annotated image button
            _, encoded_img = cv2.imencode(".jpg", result["annotated_bgr"])
            st.download_button(
                label="Download Annotated Image",
                data=encoded_img.tobytes(),
                file_name="facial_emotion_recognition_annotated.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

        with col_details:
            if result["face_count"] == 0:
                st.warning("No face detected. Adjust Cascade Scale Factor or Min Neighbors in sidebar.")
            else:
                for face in result["faces"]:
                    log_detection(face, source="Uploaded Image")
                    meta = face["metadata"]

                    st.markdown(f'<div class="swiss-mono-tag" style="margin-top: 10px;">FACE #{face["face_id"]} METRICS</div>', unsafe_allow_html=True)
                    if face["crop_rgb"] is not None:
                        f_col1, f_col2 = st.columns([1, 2])
                        with f_col1:
                            st.image(face["crop_rgb"], width=95)
                        with f_col2:
                            st.markdown(
                                f"""
                                <div style="font-size: 1.4rem; font-weight: 800; color: #000000;">
                                    {meta['emoji']} {meta['label'].upper()}
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #64748B; margin-top: 2px;">
                                    CONFIDENCE: <strong style="color: #000000;">{face['confidence']:.1f}%</strong>
                                </div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #64748B;">
                                    VALENCE: <strong style="color: #DC2626;">{meta['valence'].upper()}</strong>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    render_emotion_bars(face["probabilities"])
                    st.plotly_chart(create_radar_chart(face["probabilities"]), use_container_width=True)


# -------------------------------------------------------------
# TAB 3: SESSION ANALYTICS & LOG RECORDS
# -------------------------------------------------------------
with tab_analytics:
    st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 8px;">03.0 // HISTORICAL SESSION METRICS</div>', unsafe_allow_html=True)

    if not st.session_state.detection_history:
        st.info("No session records logged yet. Run a live camera stream or inspect an image to populate telemetry logs.")
    else:
        df = pd.DataFrame(st.session_state.detection_history)

        col_kpi_a, col_kpi_b, col_kpi_c = st.columns(3)
        with col_kpi_a:
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">03.1 // TOTAL DETECTIONS</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_kpi_b:
            most_common = df["Dominant Emotion"].mode()[0] if not df.empty else "N/A"
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">03.2 // DOMINANT MODAL EMOTION</div>
                    <div class="metric-value">{most_common}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_kpi_c:
            avg_conf = f"{df['Confidence (%)'].mean():.1f}%" if not df.empty else "N/A"
            st.markdown(
                f"""
                <div class="swiss-card">
                    <div class="metric-label">03.3 // MEAN CONFIDENCE</div>
                    <div class="metric-value">{avg_conf}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 6px;">03.4 // EMOTION PROPORTION</div>', unsafe_allow_html=True)
            emotion_counts = df["Dominant Emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]

            fig_pie = px.pie(
                emotion_counts,
                values="Count",
                names="Emotion",
                hole=0.6,
                color_discrete_sequence=['#DC2626', '#0F172A', '#475569', '#64748B', '#94A3B8', '#CBD5E1', '#E2E8F0']
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0F172A', family='Inter'),
                margin=dict(l=15, r=15, t=15, b=15),
                legend=dict(font=dict(color='#475569', family='JetBrains Mono', size=11))
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 6px;">03.5 // AFFECTIVE VALENCE SPLIT</div>', unsafe_allow_html=True)
            valence_counts = df["Valence"].value_counts().reset_index()
            valence_counts.columns = ["Valence", "Count"]

            color_map = {
                "POSITIVE": "#16A34A",
                "NEUTRAL": "#64748B",
                "NEGATIVE": "#DC2626"
            }

            fig_bar = px.bar(
                valence_counts,
                x="Valence",
                y="Count",
                color="Valence",
                color_discrete_map=color_map,
                text="Count"
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#0F172A', family='Inter'),
                margin=dict(l=15, r=15, t=15, b=15),
                xaxis=dict(gridcolor='#E2E8F0', tickfont=dict(family='JetBrains Mono')),
                yaxis=dict(gridcolor='#E2E8F0', tickfont=dict(family='JetBrains Mono')),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('<div class="swiss-mono-tag" style="margin-bottom: 8px;">03.6 // SESSION DETECTION LOG (LATEST FIRST)</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)

        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Session Log (CSV)",
            data=csv_data,
            file_name=f"facial_emotion_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==========================================
# FOOTER (SWISS MINIMAL)
# ==========================================
st.markdown("""
<div style="border-top: 2px solid #000000; padding-top: 16px; margin-top: 36px; display: flex; justify-content: space-between; align-items: center; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748B;">
    <div>FACIAL EMOTION RECOGNITION // INTERNATIONAL TYPOGRAPHIC SYSTEM</div>
    <div>TENSORFLOW • OPENCV • PYTHON 3.x</div>
</div>
""", unsafe_allow_html=True)
