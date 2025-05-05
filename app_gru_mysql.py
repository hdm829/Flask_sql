import os
import io
import numpy as np
import pymysql
from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine, text

app = Flask(__name__)

# Load model
model = load_model('model/GRU_model.h5')

# MySQL connection config
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "npy_db")

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}")

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 데이터베이스에서 npy 데이터를 읽어옴
        with engine.connect() as conn:
            result = conn.execute(text("SELECT data FROM npy_table WHERE id = 1")).fetchone()
            if result is None:
                return jsonify({"error": "No data found in database"}), 404

            npy_bytes = result[0]
            input_data = np.load(io.BytesIO(npy_bytes))

        # 모델에 입력 형태가 맞는지 확인
        if len(input_data.shape) != 3:
            return jsonify({"error": f"Input shape must be 3D, got {input_data.shape}"}), 400

        # 예측
        prediction = model.predict(input_data)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
