import requests
import base64
import json
import numpy as np
from PIL import Image
import io

img = Image.new('L', (280, 280), color=0)
for x in range(100, 180):
    for y in range(100, 180):
        img.putpixel((x, y), 255)

buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()
data_url = f"data:image/png;base64,{img_str}"

payload = {'image': data_url}

try:
    response = requests.post('http://localhost:5000/predict', json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Response JSON keys:", response.json().keys())
        print("Prediction:", response.json()['digit'])
        activations = response.json()['activations']
        print(f"L1 stats: min={min(activations[0])}, max={max(activations[0])}")
        print(f"L2 stats: min={min(activations[1])}, max={max(activations[1])}")
    else:
        print("Error:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
