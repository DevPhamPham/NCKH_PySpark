from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})
    
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        # Process the image using your model here
        # For example, if using TensorFlow/Keras:
        # result = model.predict(processed_image)
        # Then return the result as JSON
        return jsonify({'result': 'Your processed result here'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
