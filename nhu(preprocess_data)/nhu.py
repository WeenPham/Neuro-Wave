import streamlit as st
import mne
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import tempfile
import os

warnings.filterwarnings("ignore")

# ==========================================================
# 1️⃣ LOAD MÔ HÌNH RANDOM FOREST
# ==========================================================
MODEL_PATH = "RandomForest_pipeline.pkl"
try:
    rf_model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False

# ==========================================================
# 2️⃣ HÀM XỬ LÝ EEG (.bdf)
# ==========================================================
def load_eeg_bdf(file_path):
    raw = mne.io.read_raw_bdf(file_path, preload=True)
    raw.pick_types(eeg=True)
    bad_channels = ['EXG1','EXG2','EXG3','EXG4','EXG5','EXG6','EXG7']
    existing_bad = [ch for ch in bad_channels if ch in raw.ch_names]
    if existing_bad:
        raw.drop_channels(existing_bad)
    raw.filter(1., 40., fir_design='firwin')
    raw.resample(128)
    data = raw.get_data()
    return data, raw.info['sfreq'], len(raw.ch_names), raw.n_times, raw.ch_names

def extract_features(eeg_data):
    feats = []
    for ch in eeg_data:
        feats.extend([np.mean(ch), np.std(ch)])
    return np.array(feats)

# ==========================================================
# 3️⃣ PHÂN TÍCH DẢI TẦN EEG
# ==========================================================
def compute_band_powers(eeg_data, sfreq):
    freqs = np.fft.rfftfreq(eeg_data.shape[1], 1/sfreq)
    psd = np.abs(np.fft.rfft(eeg_data, axis=1)) ** 2

    bands = {
        "Delta (0.5–4 Hz)": (0.5, 4),
        "Theta (4–8 Hz)": (4, 8),
        "Alpha (8–13 Hz)": (8, 13),
        "Beta (13–30 Hz)": (13, 30),
        "Gamma (30–40 Hz)": (30, 40)
    }

    band_powers = {}
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        band_powers[band_name] = np.mean(psd[:, idx], axis=1)
    return band_powers, freqs, psd

# ==========================================================
# 4️⃣ HÀM DỰ ĐOÁN
# ==========================================================
def predict_eeg(file_path):
    eeg_data, sfreq, n_ch, n_samples, ch_names = load_eeg_bdf(file_path)
    feats = extract_features(eeg_data).reshape(1, -1)
    if feats.shape[1] != rf_model.n_features_in_:
        raise ValueError(f"Feature mismatch: có {feats.shape[1]}, model cần {rf_model.n_features_in_}")

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)
    pred = rf_model.predict(feats_scaled)[0]
    prob = rf_model.predict_proba(feats_scaled)[0]
    label = "🧠 **Healthy (HC)**" if pred == 0 else "⚠️ **Parkinson (PD)**"
    return label, prob, sfreq, n_ch, n_samples, eeg_data, ch_names

# ==========================================================
# 5️⃣ GIAO DIỆN STREAMLIT
# ==========================================================
st.set_page_config(page_title="EEG Visual Analyzer", page_icon="🧠", layout="wide")

# === CSS tùy chỉnh (nền đen – cam, font đẹp) ===
st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: #f5f5f5;
    }
    .stApp {
        background-color: #0d0d0d;
    }
    h1, h2, h3, h4 {
        color: #ff914d !important;
    }
    .block-container {
        padding-top: 1rem;
    }
    .stButton>button {
        background-color: #ff6f3c;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff8c5a;
    }
    .css-1v3fvcr, .css-18e3th9, .css-1d391kg {
        background-color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("""
    <h1 style="text-align:center;">⚡ EEG Visual Analyzer</h1>
    <p style="text-align:center; color:#ffae70;">
        Phân tích sóng não EEG – Dự đoán Parkinson và trực quan hóa theo dải tần.
    </p>
    <hr style="border: 1px solid #ff6f3c;">
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Tải file EEG (.bdf)", type=["bdf"])

if not model_loaded:
    st.error("❌ Không thể tải model. Hãy kiểm tra lại file `.pkl`.")
elif uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Đang xử lý EEG và dự đoán..."):
        try:
            label, prob, sfreq, n_ch, n_samples, eeg_data, ch_names = predict_eeg(tmp_path)
            band_powers, freqs, psd = compute_band_powers(eeg_data, sfreq)
            st.success("✅ Dự đoán hoàn tất!")

            # === Giao diện kết quả ===
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🧾 Kết quả dự đoán")
                st.markdown(label)
                st.metric("Xác suất HC", f"{prob[0]:.3f}")
                st.metric("Xác suất PD", f"{prob[1]:.3f}")
                st.write(f"**EEG:** {n_ch} kênh | {n_samples} mẫu | {sfreq:.1f} Hz")

            with col2:
                st.subheader("📊 Trung bình công suất theo dải tần")
                avg_powers = {band: np.mean(val) for band, val in band_powers.items()}
                st.bar_chart(avg_powers)

            st.markdown("---")

            # === Chọn tương tác kênh và dải sóng ===
            st.markdown("### 🎛️ Tùy chọn hiển thị")
            col_ch, col_band = st.columns(2)
            with col_ch:
                selected_channel = st.selectbox("Chọn kênh EEG:", ch_names, index=0)
                ch_idx = ch_names.index(selected_channel)
            with col_band:
                selected_band = st.selectbox("Chọn dải tần hiển thị:", list(band_powers.keys()), index=2)

            # === Hiển thị tín hiệu EEG ===
            st.markdown(f"### 🧩 Tín hiệu EEG – **{selected_channel}**")
            st.line_chart(eeg_data[ch_idx])

            # === Biểu đồ PSD với highlight dải tần ===
            st.markdown("### 🔬 Phổ tần EEG")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(freqs, psd[ch_idx], color="#ff6f3c", lw=1.5)
            ax.set_facecolor("#111")
            ax.set_title(f"Power Spectral Density – {selected_channel}", color="#ff914d")
            ax.set_xlabel("Tần số (Hz)", color="white")
            ax.set_ylabel("Công suất (Power)", color="white")

            # Highlight dải được chọn
            for band_name, (low, high) in {
                "Delta (0.5–4 Hz)": (0.5, 4),
                "Theta (4–8 Hz)": (4, 8),
                "Alpha (8–13 Hz)": (8, 13),
                "Beta (13–30 Hz)": (13, 30),
                "Gamma (30–40 Hz)": (30, 40)
            }.items():
                color = "#ff6f3c" if band_name == selected_band else "gray"
                alpha = 0.25 if band_name == selected_band else 0.08
                ax.axvspan(low, high, alpha=alpha, color=color, label=band_name)
            ax.legend(facecolor="#111", edgecolor="gray", labelcolor="white")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
        finally:
            os.remove(tmp_path)
else:
    st.info("👆 Hãy upload file EEG (.bdf) để bắt đầu phân tích.")
