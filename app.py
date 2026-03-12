import os
import requests
from flask import Flask, request, jsonify, render_template
from disease_data import get_disease_info

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

HF_API_URL = "https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG, PNG, or WEBP image.'}), 400

    image_data = file.read()

    hf_token = os.environ.get('HF_API_TOKEN', '')
    headers = {'Authorization': f'Bearer {hf_token}'} if hf_token else {}

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=image_data,
            timeout=45
        )
    except requests.Timeout:
        return jsonify({'error': 'Request timed out. Please try again.'}), 504
    except requests.ConnectionError:
        return jsonify({'error': 'Unable to reach the AI model. Check your connection.'}), 503

    if response.status_code == 503:
        resp_json = response.json() if response.content else {}
        estimated_time = resp_json.get('estimated_time', 20)
        return jsonify({
            'error': 'Model is warming up',
            'loading': True,
            'estimated_time': int(estimated_time)
        }), 503

    if response.status_code == 401:
        return jsonify({'error': 'Invalid or missing Hugging Face API token. Set HF_API_TOKEN env variable.'}), 401

    if response.status_code != 200:
        return jsonify({'error': f'API returned status {response.status_code}. Please try again.'}), 500

    try:
        predictions = response.json()
    except Exception:
        return jsonify({'error': 'Invalid response from model API.'}), 500

    if not isinstance(predictions, list) or len(predictions) == 0:
        return jsonify({'error': 'No predictions returned. Ensure the image is a clear leaf photograph.'}), 422

    results = []
    for pred in predictions[:5]:
        label = pred.get('label', '')
        score = pred.get('score', 0.0)
        info = get_disease_info(label)
        results.append({
            'label': label,
            'display_name': info['display_name'],
            'score': round(score * 100, 1),
            'plant': info['plant'],
            'disease': info['disease'],
            'is_healthy': info['is_healthy'],
            'severity': info['severity'],
            'severity_color': info['severity_color'],
            'description': info['description'],
            'treatment': info['treatment'],
            'prevention': info['prevention']
        })

    return jsonify({'predictions': results, 'top': results[0] if results else None})


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
