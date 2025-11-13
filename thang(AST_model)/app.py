from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import numpy as np
import mne
import torch
from transformers import ASTForAudioClassification, AutoFeatureExtractor
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Global storage for visualization data
app_data = {}

# Cấu hình
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'bdf'}
MODEL_PATH = 'models/ast-finetuned-pd'  # Đường dẫn model đã train

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Load model và feature extractor
print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASTForAudioClassification.from_pretrained(MODEL_PATH).to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_bdf_file(file_path, fs_target=16000):
    """
    Xử lý file BDF thành các epochs 2 giây
    """
    # Load EEG
    raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
    raw.resample(fs_target)
    raw.filter(0.5, 50)
    
    # Chọn channel Cz (hoặc channel đầu tiên)
    ch_name = "Cz" if "Cz" in raw.ch_names else raw.ch_names[0]
    ch_idx = raw.ch_names.index(ch_name)
    signal_data = raw.get_data()[ch_idx]
    
    # Chia thành epochs 2 giây
    epoch_len = int(2 * fs_target)
    n_epochs = len(signal_data) // epoch_len
    
    epochs = []
    for i in range(n_epochs):
        seg = signal_data[i*epoch_len:(i+1)*epoch_len]
        # Chuẩn hóa [-1,1]
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        epochs.append(seg.astype(np.float32))
    
    return epochs, ch_name, n_epochs

def predict_epochs(epochs):
    """
    Dự đoán cho từng epoch và tổng hợp kết quả
    """
    predictions = []
    probabilities = []

    with torch.no_grad():
        for epoch in epochs:
            # Feature extraction
            features = feature_extractor(
                epoch,
                sampling_rate=16000,
                return_tensors="pt"
            )

            # Inference
            input_values = features["input_values"].to(device)
            outputs = model(input_values)
            logits = outputs.logits

            # Softmax để lấy probability
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            pred = np.argmax(probs)

            predictions.append(int(pred))
            probabilities.append(probs.tolist())

    return predictions, probabilities

def analyze_results(predictions, probabilities):
    """
    Phân tích kết quả và đưa ra nhận định
    """
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    
    # Tính tỷ lệ PD
    pd_ratio = np.mean(predictions == 1) * 100
    
    # Confidence trung bình cho mỗi class
    avg_prob_hc = np.mean(probabilities[:, 0]) * 100
    avg_prob_pd = np.mean(probabilities[:, 1]) * 100
    
    # Xác định các khoảng thời gian có dấu hiệu PD
    pd_segments = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            start_time = i * 2  # Mỗi epoch 2 giây
            end_time = start_time + 2
            confidence = probabilities[i, 1] * 100
            pd_segments.append({
                "epoch": i,
                "start_time": start_time,
                "end_time": end_time,
                "confidence": round(confidence, 2)
            })
    
    # Đưa ra kết luận
    if pd_ratio >= 70:
        diagnosis = "Có dấu hiệu Parkinson's rõ rệt"
        risk_level = "high"
    elif pd_ratio >= 40:
        diagnosis = "Có dấu hiệu Parkinson's, cần theo dõi thêm"
        risk_level = "medium"
    elif pd_ratio >= 20:
        diagnosis = "Có một số dấu hiệu bất thường, nên kiểm tra thêm"
        risk_level = "low"
    else:
        diagnosis = "Không có dấu hiệu Parkinson's rõ rệt"
        risk_level = "normal"
    
    return {
        "diagnosis": diagnosis,
        "risk_level": risk_level,
        "pd_ratio": round(pd_ratio, 2),
        "avg_prob_hc": round(avg_prob_hc, 2),
        "avg_prob_pd": round(avg_prob_pd, 2),
        "total_epochs": len(predictions),
        "pd_epochs": int(np.sum(predictions == 1)),
        "hc_epochs": int(np.sum(predictions == 0)),
        "pd_segments": pd_segments
    }



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only .bdf files are allowed'}), 400
    
    try:
        # Lưu file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Xử lý file BDF
        print(f"Processing file: {filename}")
        epochs, channel_used, n_epochs = process_bdf_file(filepath)
        
        # Dự đoán
        print("Making predictions...")
        predictions, probabilities = predict_epochs(epochs)
        
        # Phân tích kết quả
        analysis = analyze_results(predictions, probabilities)
        
        # Thêm thông tin file
        analysis['filename'] = filename
        analysis['channel_used'] = channel_used
        analysis['duration_seconds'] = n_epochs * 2
        
        # Lưu dữ liệu cho visualization
        app_data['epochs'] = epochs
        app_data['predictions'] = predictions

        # Xóa file sau khi xử lý
        os.remove(filepath)

        return jsonify({
            'success': True,
            'result': analysis,
            'predictions': predictions,
            'probabilities': probabilities
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2004)


