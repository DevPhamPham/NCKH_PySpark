import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


class FaceDetector:
    def __init__(self, face_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_faces(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # cropped_faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped_face = img[y:y+h, x:x+w]
            # cropped_faces.append(cropped_face)

        # cv2.imshow('face',np.array(cropped_face))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return cropped_face

class ExtractFeature:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.intermediate_layer_model = Model(inputs=self.model.input, outputs=self.model.layers[-6].output)

    def load_and_preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(128, 128))  # Đảm bảo kích thước hình ảnh là 128x128
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def vectorize_and_save(self, image_path, output_folder):
        # Load và tiền xử lý ảnh
        img = self.load_and_preprocess_image(image_path)

        # Đưa ảnh qua mô hình để lấy vector
        vector = self.intermediate_layer_model.predict(img)

        # Lấy tên file từ đường dẫn ảnh
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Lưu vector thành file .npy
        vector_file = os.path.join(output_folder, filename + '.npy')
        np.save(vector_file, vector)

        print(f"Saved vector for {filename} in {vector_file}")

# Sử dụng class FaceDetector
if __name__ == "__main__":
    face_cascade_path = './haarcascade_frontalface_default.xml'
    image_path = './img.jpeg'
    face_detector = FaceDetector(face_cascade_path)
    ime_cut = face_detector.detect_faces(image_path)
    cv2.imwrite("image.jpg",ime_cut)
    
    model_path = './model.h5'

    feature = ExtractFeature(model_path)

    input_image = 'image.jpg'

    output_folder = './features'

    os.makedirs(output_folder, exist_ok=True)

    feature.vectorize_and_save(input_image, output_folder)

    print("Conversion complete.")

