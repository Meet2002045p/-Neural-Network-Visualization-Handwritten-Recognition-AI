from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'emnist_model.h5')
model = None

EMNIST_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'g', 41: 'h', 42: 'n', 43: 'q', 44: 'r', 45: 't'
}

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            dummy_input = np.zeros((1, 28, 28, 1))
            model.predict(dummy_input)
            print("Model loaded and initialized successfully.")
            model.summary()
            print("Layer names:", [layer.name for layer in model.layers])
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model file not found. Please run train_model.py first.")

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        load_model()
        if model is None:
             return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        image_data = data['image']
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        img = img.convert('L')
        
        img_array = np.array(img)

        if img_array.mean() > 128:
             img_array = 255 - img_array
             
        rows = np.any(img_array > 40, axis=1)
        cols = np.any(img_array > 40, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            digit = img_array[rmin:rmax+1, cmin:cmax+1]
            
            h, w = digit.shape
            scale = 20.0 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            digit_pil = Image.fromarray(digit).resize((new_w, new_h), Image.Resampling.LANCZOS)
            digit_scaled = np.array(digit_pil)
            
            new_img = np.zeros((28, 28), dtype=np.float32)
            
            r_offset = (28 - new_h) // 2
            c_offset = (28 - new_w) // 2
            
            new_img[r_offset:r_offset+new_h, c_offset:c_offset+new_w] = digit_scaled
            img_array = new_img
        else:
             img_array = np.array(Image.fromarray(img_array).resize((28, 28)))

        img_array = img_array.T
        
        img_array = img_array.astype('float32') / 255.0
        
        img_array = img_array.reshape(1, 28, 28, 1)

        try:
            inputs = tf.keras.Input(shape=(28, 28, 1))
            x = inputs
            
            layer_outputs = {}
            target_layers = ['layer1', 'layer2', 'output']
            
            for layer in model.layers:
                x = layer(x)
                if layer.name in target_layers:
                    layer_outputs[layer.name] = x
            
            final_outputs = [layer_outputs.get(name) for name in target_layers]
            
            intermediate_layer_model = tf.keras.Model(inputs=inputs, outputs=final_outputs)
            intermediate_output = intermediate_layer_model.predict(img_array)
            
            layer1_activation = intermediate_output[0][0].tolist() 
            layer2_activation = intermediate_output[1][0].tolist() 
            prediction = intermediate_output[2] 
            
        except Exception as e:
            print(f"Error building intermediate model: {e}")
            prediction = model.predict(img_array)
            layer1_activation = [0] * 256
            layer2_activation = [0] * 128
        
        predicted_index = np.argmax(prediction)
        predicted_char = EMNIST_MAPPING.get(int(predicted_index), '?')
        probabilities = prediction[0].tolist()
        
        top_indices = np.argsort(prediction[0])[-5:][::-1]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'char': EMNIST_MAPPING.get(int(idx), '?'),
                'probability': float(prediction[0][idx])
            })
        
        return jsonify({
            'digit': predicted_char,
            'probabilities': probabilities, 
            'top_predictions': top_predictions,
            'activations': [layer1_activation, layer2_activation]
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
