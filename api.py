from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import findspark

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.1.1-bin-hadoop3.2"
findspark.init()

import pyspark as spark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import BucketedRandomProjectionLSHModel
# print("version pyspark: ",spark.__version__)


class FaceDetector:
    def __init__(self, face_cascade_path):
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

    def detect_faces(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cropped_face = img[y:y+h, x:x+w]

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

        # print(f"Saved vector for {filename} in {vector_file}")

def read_npy_files_from_parent_dir(spark,parent_dir):
    schema = StructType([
        StructField("id", StringType(), nullable=False),
        StructField("features", VectorUDT(), nullable=False)
    ])

    def process_file(file, idx):
        file_name = os.path.basename(file)
        vector = np.load(file)
        vector = Vectors.dense(*vector.tolist())  # Chuyển đổi sang Vector của PySpark
        return (file_name, vector)

    all_files = []
    for root, dirs, files in os.walk(parent_dir):
        for idx, file in enumerate(files):
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                all_files.append((file_path, idx))

    rdd = spark.sparkContext.parallelize(all_files)
    df = rdd.map(lambda x: process_file(x[0], x[1])).toDF(schema)
    return df



app = Flask(__name__)

@app.route('/v1/nckh', methods=['POST'])
def process_image():
    print(request.files)  # Print request files
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})
    
    try:
        # Lưu ảnh vào hệ thống và nhận đường dẫn
        image_path = 'user.jpg'  # Đường dẫn để lưu ảnh
        image_file.save(image_path)

        face_cascade_path = './haarcascade_frontalface_default.xml'
        # image_path = './img.jpeg'
        face_detector = FaceDetector(face_cascade_path)
        ime_cut = face_detector.detect_faces(image_path)
        cv2.imwrite("image.jpg",ime_cut)

        model_path = './model.h5'

        feature = ExtractFeature(model_path)

        input_image = 'image.jpg'

        output_folder = './features'

        os.makedirs(output_folder, exist_ok=True)

        feature.vectorize_and_save(input_image, output_folder)

        # print("Conversion complete.")

        # Khởi tạo SparkSession
        spark = SparkSession.builder \
            .appName("Load LSH Model") \
            .getOrCreate()

        # Đường dẫn tới thư mục chứa mô hình đã lưu
        model_path = "./model_lsh"
        directory = "./vectorEmbeddingToTrainLSH"
        df = read_npy_files_from_parent_dir(spark, directory)

        # Tải mô hình đã lưu
        loaded_model = BucketedRandomProjectionLSHModel.load(model_path)

        # Hiển thị thông tin của mô hình
        # print("Model Parameters:")
        # print("InputCol:", loaded_model.getInputCol())
        # print("OutputCol:", loaded_model.getOutputCol())
        # print("NumHashTables:", loaded_model.getNumHashTables())

        # Tạo DataFrame chứa vector test
        # Trích xuất đặc trưng từ ảnh
        img_features = feature.intermediate_layer_model.predict(feature.load_and_preprocess_image(input_image))
        # Chuyển đổi đặc trưng thành Vector
        img_vector = Vectors.dense(img_features.flatten())
        # print("Image Vector:")
        # print(img_vector.shape)

        # Tìm dữ liệu gần nhất với vector test bằng mô hình đã tải
        nearest_df = loaded_model.approxNearestNeighbors(df, img_vector, 5)
        nearest_df.show()
    
        result = nearest_df.select("id").collect()
        print(result)
        print(type(result))

        # Return the result as JSON
        # Dừng SparkSession
        spark.stop()
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
