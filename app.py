import os
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sklearn.cluster import KMeans
from kneed import KneeLocator
import joblib
from PIL import Image
import io
import base64

app = Flask(__name__)
# Update CORS configuration to allow all origins
CORS(app, resources={r"/api/*": {"origins": ["https://gel-card-classifier-60152oeqg-albaraazains-projects.vercel.app", "http://localhost:8081"]}})

# Load the trained model, scaler, and label encoder
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load dataset
with open('ReactionGradeDataset_version2.json', 'r') as f:
    dataset = json.load(f)

IMAGE_FOLDER = 'Reaction_grades_dataset_new'


def elbow_method(data, max_clusters=10):
    inertias = []
    k_values = range(1, min(max_clusters + 1, len(data) + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    kneedle = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    optimal_k = kneedle.elbow
    if optimal_k is None:
        optimal_k = 2  # Default to 2 if no clear elbow is found

    return optimal_k

def extract_gel_card_features_dynamic(image, n_sections=20, red_threshold=130):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    red_mask = cr > red_threshold
    cr_red = np.where(red_mask, cr, 0)
    cr_normalized = cv2.normalize(cr_red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pixel_values = cr_normalized[cr_normalized > 0].reshape((-1, 1))

    if len(pixel_values) == 0:
        return np.zeros(n_sections * 2 + 2)  # Return zero features if no red pixels

    n_clusters = elbow_method(pixel_values)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixel_values)
    labels = np.zeros_like(cr_normalized)
    labels[cr_normalized > 0] = kmeans.labels_ + 1

    height, width = cr_normalized.shape
    section_height = height // n_sections
    features = []
    for i in range(n_sections):
        start_row = i * section_height
        end_row = (i + 1) * section_height if i < n_sections - 1 else height
        section = labels[start_row:end_row, :]

        total_pixels = section.size
        cluster_ratios = [np.sum(section == j) / total_pixels for j in range(1, n_clusters + 1)]
        features.extend(cluster_ratios)

    global_ratios = [np.sum(labels == j) / labels.size for j in range(1, n_clusters + 1)]
    features.extend(global_ratios)

    # Add top and bottom cluster sizes
    top_cluster = np.argmax(np.bincount(labels[0])[1:]) + 1 if np.any(labels[0] > 0) else 0
    bottom_cluster = np.argmax(np.bincount(labels[-1])[1:]) + 1 if np.any(labels[-1] > 0) else 0
    features.extend([np.sum(labels == top_cluster) / labels.size,
                     np.sum(labels == bottom_cluster) / labels.size])

    return np.array(features).flatten(), n_clusters  # Ensure we return a 1D array

def pad_features(features, max_length):
    features = np.array(features).flatten()  # Ensure features is a 1D array
    if len(features) < max_length:
        return np.pad(features, (0, max_length - len(features)), 'constant')
    elif len(features) > max_length:
        return features[:max_length]
    else:
        return features

def generate_processing_stages(image):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    red_threshold = 130

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    _, cr, _ = cv2.split(ycrcb)
    red_mask = cr > red_threshold
    cr_red = np.where(red_mask, cr, 0)
    cr_normalized = cv2.normalize(cr_red, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pixel_values = cr_normalized[cr_normalized > 0].reshape((-1, 1))

    if len(pixel_values) > 0:
        optimal_k = elbow_method(pixel_values)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(pixel_values)
        labels = np.zeros_like(cr_normalized)
        labels[cr_normalized > 0] = kmeans.labels_ + 1
    else:
        optimal_k = 1
        labels = np.zeros_like(cr_normalized)

    stages = []

    # Original Image
    stages.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Cr Channel
    stages.append(cv2.applyColorMap(cr, cv2.COLORMAP_JET))

    # Normalized Red Regions (Cr)
    stages.append(cv2.applyColorMap(cr_normalized, cv2.COLORMAP_JET))

    # K-means Segmented Red Regions
    segmented = np.zeros((cr_normalized.shape[0], cr_normalized.shape[1], 3), dtype=np.uint8)
    for i in range(1, optimal_k + 1):
        segmented[labels == i] = [np.random.randint(0, 255) for _ in range(3)]
    stages.append(segmented)

    # Segmentation Overlay
    overlay = cv2.addWeighted(image, 0.7, segmented, 0.3, 0)
    stages.append(overlay)

    return stages


@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify(dataset)

@app.route('/api/classify_image/<int:image_id>', methods=['GET'])
def classify_image(image_id):
    item = next((item for item in dataset if item['id'] == image_id), None)
    if item is None:
        return jsonify({'error': 'Image not found'}), 404

    image_path = os.path.join(IMAGE_FOLDER, os.path.basename(item['data']['image']))
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 500

    image = cv2.resize(image, (224, 224))
    features, _ = extract_gel_card_features_dynamic(image)
    max_length = model.n_features_in_
    padded_features = pad_features(features, max_length)

    scaled_features = scaler.transform(padded_features.reshape(1, -1))

    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)

    predicted_class = label_encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities) * 100)

    return jsonify({
        'predicted_class': predicted_class,
        'confidence': confidence
    })

@app.route('/api/process_image/<int:image_id>', methods=['GET'])
def process_image(image_id):
    item = next((item for item in dataset if item['id'] == image_id), None)
    if item is None:
        return jsonify({'error': 'Image not found'}), 404

    image_path = os.path.join(IMAGE_FOLDER, os.path.basename(item['data']['image']))
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 500

    image = cv2.resize(image, (224, 224))
    processed_stages = generate_processing_stages(image)

    features, n_clusters = extract_gel_card_features_dynamic(image)
    max_length = model.n_features_in_
    padded_features = pad_features(features, max_length)

    feature_descriptions = [f"Feature {i+1}" for i in range(len(padded_features))]

    response_data = {
        'feature_vector': padded_features.tolist(),  # Convert numpy array to list
        'feature_descriptions': feature_descriptions,
        'n_clusters': int(n_clusters)  # Ensure this is JSON serializable
    }

    return jsonify(response_data)

@app.route('/api/image/<int:image_id>/<int:stage>', methods=['GET'])
def get_processed_image(image_id, stage):
    item = next((item for item in dataset if item['id'] == image_id), None)
    if item is None:
        return jsonify({'error': 'Image not found'}), 404

    image_path = os.path.join(IMAGE_FOLDER, os.path.basename(item['data']['image']))
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': 'Failed to read image'}), 500

    image = cv2.resize(image, (224, 224))
    processed_stages = generate_processing_stages(image)

    if stage < 0 or stage >= len(processed_stages):
        return jsonify({'error': 'Invalid stage'}), 400

    _, buffer = cv2.imencode('.png', processed_stages[stage])
    return send_file(io.BytesIO(buffer), mimetype='image/png')

@app.route('/api/save_label/<int:image_id>', methods=['POST'])
def save_label(image_id):
    new_label = request.json.get('label')
    if new_label is None:
        return jsonify({'error': 'No label provided'}), 400

    item = next((item for item in dataset if item['id'] == image_id), None)
    if item is None:
        return jsonify({'error': 'Image not found'}), 404

    item['annotations'][0]['result'][0]['value']['choices'][0] = new_label
    return jsonify({'message': 'Label updated successfully'})

@app.route('/api/remove_image/<int:image_id>', methods=['DELETE'])
def remove_image(image_id):
    global dataset
    dataset = [item for item in dataset if item['id'] != image_id]
    return jsonify({'message': 'Image removed successfully'})

@app.route('/api/update_dataset', methods=['POST'])
def update_dataset():
    with open('ReactionGradeDataset_version2.json', 'w') as f:
        json.dump(dataset, f)
    return jsonify({'message': 'Dataset updated successfully'})

@app.route('/api/dataset_stats', methods=['GET'])
def get_dataset_stats():
    total_images = len(dataset)
    label_distribution = {}
    for item in dataset:
        label = item['annotations'][0]['result'][0]['value']['choices'][0]
        label_distribution[label] = label_distribution.get(label, 0) + 1

    return jsonify({
        'total_images': total_images,
        'label_distribution': label_distribution
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
