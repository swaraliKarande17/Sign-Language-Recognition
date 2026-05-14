import streamlit as st	
import tempfile, os, cv2, time	
import numpy as np	
import torch	
	
# Page config — MUST be the first Streamlit command in the file	
st.set_page_config(	
    page_title="Sign Language Translator",	
    page_icon="🤟",	
    layout="wide",	
    initial_sidebar_state="expanded"	
)	
	
# Custom CSS for nicer styling	
st.markdown("""	
<style>	
.result-box { background:#f0fdf4; border-left:4px solid #22c55e;	
              padding:16px; border-radius:8px; margin:12px 0; }	
.gloss-box  { background:#eff6ff; border-left:4px solid #3b82f6;	
              padding:16px; border-radius:8px; margin:12px 0; }	
</style>	
""", unsafe_allow_html=True)	
	
	
# Load model once and cache it	
# @st.cache_resource means this only runs once even if page refreshes	
@st.cache_resource	
def load_model():	
    checkpoint = "checkpoints/best_model.pt"	
    if not os.path.exists(checkpoint):	
        return None	
    from src.inference.video_inference import VideoInference	
    device = "cuda" if torch.cuda.is_available() else "cpu"	
    return VideoInference(checkpoint, device=device)	
	
	
# ── Header ──────────────────────────────────────────────────────────	
st.title("🤟 Sign Language Translator")	
st.caption("PHOENIX-2014T · German Sign Language → English · ResNet-50 + BiLSTM + CTC")	
st.divider()	
	
# ── Sidebar ─────────────────────────────────────────────────────────	
with st.sidebar:	
    st.header("⚙️  Settings")	
    max_frames   = st.slider("Max frames to process", 50, 300, 150)	
    show_glosses = st.toggle("Show German glosses", value=True)	
    st.divider()	
    st.markdown("**Model:** ResNet-50 + BiLSTM")	
    st.markdown("**Dataset:** RWTH PHOENIX-2014T")	
    st.markdown("**Translation:** Helsinki-NLP/opus-mt-de-en")	
	
# ── Two tabs: video upload and live webcam ──────────────────────────	
tab1, tab2 = st.tabs(["📹 Upload Video", "📷 Live Webcam"])	
	
engine = load_model()	
	
# ════════════════════════════════════════════════════════════════════	
# TAB 1: VIDEO UPLOAD	
# ════════════════════════════════════════════════════════════════════	
with tab1:	
    uploaded = st.file_uploader(	
        "Upload a sign language video",	
        type=["mp4", "avi", "mov", "mkv"],	
        help="Upload a video of continuous German Sign Language (DGS)"	
    )	
	
    if uploaded:	
        # Save to temp file because OpenCV needs a file path	
        suffix = "." + uploaded.name.split(".")[-1]	
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:	
            tmp.write(uploaded.read())	
            tmp_path = tmp.name	
	
        col1, col2 = st.columns([1, 1])	
	
        with col1:	
            st.video(tmp_path)	
	
        with col2:	
            if engine is None:	
                st.warning(	
                    "No trained model found. "	
                    "Train the model first and place best_model.pt "	
                    "in the checkpoints/ folder."	
                )	
            else:	
                if st.button("🔍 Recognize Signs", type="primary"):	
                    with st.spinner("Analyzing video frames..."):	
                        t_start = time.time()	
                        result  = engine.run(tmp_path, max_frames)	
                        elapsed = time.time() - t_start	
	
                    if "error" in result:	
                        st.error(result["error"])	
                    else:	
                        st.success(	
                            f"Processed {result['num_frames']} frames in {elapsed:.1f}s"	
                        )	
                        if show_glosses:	
                            st.markdown("**German Glosses Detected**")	
                            gloss_text = " · ".join(result["glosses"])	
                            st.markdown(	
                                f'<div class="gloss-box">{gloss_text}</div>',	
                                unsafe_allow_html=True	
                            )	
                        st.markdown("**English Translation**")	
                        st.markdown(	
                            f'<div class="result-box"><strong>{result["english"]}</strong></div>',	
                            unsafe_allow_html=True	
                        )	
	
        os.unlink(tmp_path)  # clean up temp file	
	
# ════════════════════════════════════════════════════════════════════	
# TAB 2: LIVE WEBCAM	
# ════════════════════════════════════════════════════════════════════	
with tab2:	
    st.info("Webcam mode: translates continuously every 3 seconds of signing.")	
	
    col1, col2 = st.columns([3, 2])	
    with col1:	
        run_webcam = st.toggle("▶ Start Webcam")	
        frame_ph   = st.empty()	
    with col2:	
        st.markdown("**Live Translation**")	
        result_ph = st.empty()	
	
    if run_webcam:	
        cap = cv2.VideoCapture(0)  # 0 = default webcam	
        frames_buffer = []	
        last_translate = time.time()	
	
        while run_webcam:	
            ret, frame = cap.read()	
            if not ret:	
                st.error("Cannot access webcam. Check camera permissions.")	
                break	
	
            # Show live feed in RGB	
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)	
            frame_ph.image(frame_rgb, channels="RGB", use_column_width=True)	
            frames_buffer.append(cv2.resize(frame_rgb, (224, 224)))	
	
            # Every 3 seconds, run inference on buffered frames	
            if (time.time() - last_translate > 3	
                    and engine is not None	
                    and len(frames_buffer) >= 30):	
	
                # Save buffer as temp video file	
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:	
                    out = cv2.VideoWriter(	
                        tmp.name,	
                        cv2.VideoWriter_fourcc(*"mp4v"),	
                        10, (224, 224)	
                    )	
                    for f in frames_buffer[-90:]:  # last 9 seconds at 10fps	
                        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))	
                    out.release()	
	
                    result = engine.run(tmp.name, max_frames=90)	
                    os.unlink(tmp.name)	
	
                if "english" in result:	
                    result_ph.markdown(	
                        f'<div style="background:#f0fdf4;border-left:4px solid #22c55e;'	
                        f'padding:16px;border-radius:8px;margin:12px 0;">'	
                        f'<strong>{result["english"]}</strong></div>',	
                        unsafe_allow_html=True	
                    )	
	
                last_translate = time.time()	
                frames_buffer = frames_buffer[-30:]  # keep overlap	
	
        cap.release()	