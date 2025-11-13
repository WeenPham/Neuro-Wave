
import streamlit as st
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mne
import os
import tempfile

# ==============================================================================
# 0️⃣ Device & Model (SỬ DỤNG MÔ HÌNH CỦA BẠN VỚI TRỌNG SỐ ĐÃ TẢI)
# ==============================================================================

# Dán định nghĩa mô hình của bạn vào đây
class EEGCNN1D(nn.Module):
    def __init__(self, n_channels=32, n_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x

# --- Cấu hình và khởi tạo model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ### <<< SỬA LỖI TẠI ĐÂY >>> ###
# THAY ĐỔI SỐ KÊNH TỪ 32 THÀNH 40 ĐỂ KHỚP VỚI MODEL ĐÃ HUẤN LUYỆN
N_CHANNELS_MODEL_EXPECTS = 40
N_CLASSES = 2

# Khởi tạo kiến trúc model
model = EEGCNN1D(n_channels=N_CHANNELS_MODEL_EXPECTS, n_classes=N_CLASSES)

MODEL_PATH = "eeg_cnn_model_weights.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    st.session_state.model_loaded = True
except FileNotFoundError:
    st.session_state.model_loaded = False
except Exception as e:
    # Thêm một thông báo lỗi cụ thể hơn trên web
    st.error(f"LỖI NGHIÊM TRỌNG KHI TẢI MODEL: {e}")
    st.error("Rất có thể số kênh `N_CHANNELS_MODEL_EXPECTS` trong code không khớp với số kênh của model đã được huấn luyện. Vui lòng kiểm tra lại.")
    st.stop()


model.to(device)
model.eval()

# ==============================================================================
# CÁC PHẦN CÒN LẠI CỦA FILE GIỮ NGUYÊN
# (Copy-paste toàn bộ các hàm process_eeg, plot_attributions, và giao diện Streamlit vào đây)
# ==============================================================================

# 1️⃣ Cấu hình chung
channels_of_interest = ['C3','Cz','C4','F3','F4','P3','P4']
n_steps_ig = 50
resample_rate = 128
cmap = plt.cm.inferno

# 2️⃣ Hàm xử lý EEG + Integrated Gradients
def process_eeg(file_path, model, device, n_steps, resample_rate, n_channels_expected):
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False).crop(0,30)
    n_chans_raw = len(mne.pick_types(raw.info, eeg=True))
    raw.filter(1., 40., fir_design='firwin', verbose=False)
    ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800, verbose=False)
    ica.fit(raw)
    eog_indices, _ = ica.find_bads_eog(raw, ch_name='Fp1')
    ica.exclude = eog_indices
    raw_clean = ica.apply(raw.copy(), verbose=False)
    raw_eeg = raw_clean.copy().pick_types(eeg=True)
    eeg_data_original = raw_eeg.get_data()
    eeg_data_adjusted = eeg_data_original
    if n_chans_raw != n_channels_expected:
        if n_chans_raw < n_channels_expected:
            repeat_factor = (n_channels_expected // n_chans_raw) + 1
            eeg_data_adjusted = np.tile(eeg_data_original, (repeat_factor, 1))[:n_channels_expected, :]
        else:
            eeg_data_adjusted = eeg_data_original[:n_channels_expected, :]
    x_tensor = torch.tensor(eeg_data_adjusted, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class_idx = torch.max(probabilities, 1)
    ig = IntegratedGradients(model)
    attr = ig.attribute(x_tensor, target=predicted_class_idx.item(), n_steps=n_steps).detach().cpu().numpy()[0]
    attr = attr[:n_chans_raw, :]
    raw_resampled = raw_eeg.copy().resample(resample_rate)
    x_data_resampled = raw_resampled.get_data()
    factor = eeg_data_original.shape[1] // x_data_resampled.shape[1]
    attr_ds = attr[:, ::factor]
    attr_norm = np.abs(attr_ds)**0.5
    max_attr = attr_norm.max()
    if max_attr > 0:
        attr_norm /= max_attr
    return x_data_resampled, attr_norm, raw_resampled.ch_names, predicted_class_idx.item(), confidence.item(), n_chans_raw

# 3️⃣ Hàm vẽ biểu đồ
def plot_attributions(file_name, x_data, attr_data, ch_names, channels_of_interest):
    n_per_fig = len(channels_of_interest)
    plt.style.use('default')
    fig, axes = plt.subplots(n_per_fig, 1, figsize=(18, 2.5 * n_per_fig), sharex=True)
    if n_per_fig == 1: axes = [axes]
    for j, ch_name in enumerate(channels_of_interest):
        ax = axes[j]
        if ch_name in ch_names:
            ch_idx = ch_names.index(ch_name)
            y_signal, attr_signal = x_data[ch_idx], attr_data[ch_idx]
            segments = np.array([[[k, y_signal[k]], [k+1, y_signal[k+1]]] for k in range(y_signal.shape[0]-1)])
            lc = LineCollection(segments, colors=cmap(attr_signal[:-1]), linewidths=1.5)
            ax.add_collection(lc)
            y_range = y_signal.max() - y_signal.min()
            padding = y_range * 0.1
            ax.set_xlim(0, y_signal.shape[0]); ax.set_ylim(y_signal.min() - padding, y_signal.max() + padding)
            ax.set_ylabel(ch_name, fontsize=12); ax.grid(True, linestyle='--', alpha=0.4)
        else:
            ax.set_ylabel(f"{ch_name}\n(not found)")
            ax.text(0.5, 0.5, 'Kênh không có trong file', ha='center', va='center', transform=ax.transAxes)
    axes[-1].set_xlabel("Thời gian (Mẫu)", fontsize=12)
    fig.suptitle(f"Phân Tích AI Trên Tín Hiệu EEG\nFile: {file_name}", fontsize=18)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
    cbar.set_label('Mức Độ Quan Trọng (Integrated Gradients)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.98, 0.95]); return fig

# 4️⃣ Xây dựng giao diện Streamlit
st.set_page_config(layout="wide")
st.title("🔬 Ứng dụng Phân tích và Giải thích AI cho EEG")

if 'model_loaded' not in st.session_state or not st.session_state.model_loaded:
    st.error(f"Không tìm thấy file trọng số '{MODEL_PATH}'. Vui lòng đảm bảo bạn đã lưu model và file này tồn tại trong môi trường Colab.")
    st.stop()

if st.session_state.get('model_loaded', False):
  st.success(f"Đã tải thành công model từ file '{MODEL_PATH}'.")

st.markdown(f"""
Ứng dụng này sử dụng mô hình **`{model.__class__.__name__}`** để dự đoán trên dữ liệu EEG.
- **Model mong đợi đầu vào có `{N_CHANNELS_MODEL_EXPECTS}` kênh.**
- **Giải thích (XAI)** được thực hiện bằng kỹ thuật **Integrated Gradients**.
""")

st.header("1. Tải lên file EEG (.bdf)")
uploaded_file = st.file_uploader("Chọn một file .bdf", type=["bdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.bdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    st.info(f"Đã tải lên file: **{uploaded_file.name}**. Bắt đầu xử lý...")
    try:
        raw_info = mne.io.read_raw_bdf(tmp_file_path, preload=False, verbose=False)
        n_chans_in_file = len(mne.pick_types(raw_info.info, eeg=True))
        if n_chans_in_file != N_CHANNELS_MODEL_EXPECTS:
            st.warning(f"⚠️ **Cảnh báo:** Model được huấn luyện với **{N_CHANNELS_MODEL_EXPECTS}** kênh, nhưng file của bạn có **{n_chans_in_file}** kênh EEG. Ứng dụng sẽ tự động điều chỉnh dữ liệu, nhưng kết quả có thể không tối ưu.")
        with st.spinner('Đang chạy mô hình AI và tính toán XAI... Vui lòng chờ.'):
            x_data, attr_data, ch_names, pred_idx, conf, _ = process_eeg(tmp_file_path, model, device, n_steps_ig, resample_rate, N_CHANNELS_MODEL_EXPECTS)
            fig = plot_attributions(uploaded_file.name, x_data, attr_data, ch_names, channels_of_interest)
        st.success("Xử lý hoàn tất!")
        st.header("2. Kết quả dự đoán")
        class_names = [f"Lớp {i}" for i in range(N_CLASSES)]
        st.metric(label="Dự đoán của mô hình", value=class_names[pred_idx])
        st.progress(conf)
        st.write(f"Độ tin cậy (Confidence): **{conf:.2%}**")
        st.header("3. Phân tích Explainable AI (XAI)")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
        st.error("Vui lòng kiểm tra định dạng file BDF. Kênh 'Fp1' phải tồn tại để thuật toán ICA loại bỏ nhiễu mắt có thể hoạt động.")
    finally:
        os.remove(tmp_file_path)
else:
    st.info("Vui lòng tải lên một file .bdf để bắt đầu.")
