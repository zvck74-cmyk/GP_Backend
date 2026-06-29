import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter
import pandas as pd
import gdown
import gc
# ==========================================
# 1. إعدادات الثوابت والخرائط (Configurations)
# ==========================================



MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_LINKS = {
    "VGG19.h5": "1WJcclKhXYGA5AFdgO-1yLP--yn5ofrb_",
    "VGG16.h5": "1H5IhdbE9VQuDo4xGI7Bp_Ho5RjNfG7GD",
    "resnet101v2.h5": "1ShViyv5O9P3xfC8Aoe_2mMxOdkwawnq7",
    "InceptionV3.h5": "13H3e57NDTr4tnndBi1sW3j-59_Auf2OI",
    "crop_model.pkl": "1RabvGwF-CqOo3wmcuZVeSmMLML080QR0",
    "crop_label_encoder.pkl": "1QpwSKHlKJu5zkeXWcKaKJnI3nzAa2Dwl",
    "fertilizer_model_Xg.pkl": "1Ptosyi84tKMzC8vpQAKY_yi2m7vblhCZ",
    "fertilizer_label_encoder.pkl": "1sQTu2kvfpHMKDTLJc-bydOFoVx_mk-HC"
}

def download_models():
    for name, file_id in MODEL_LINKS.items():
        path = os.path.join(MODEL_DIR, name)
        
        if os.path.exists(path):
            print(f"{name} already exists ")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {name}...")
        gdown.download(url, path, quiet=False)

CLASS_INDICES = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37
}
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}


class AgriculturalAI:
    def __init__(self, models_dir="models"):
        download_models()
        self.models_dir = models_dir
        self.models = {}
        self.load_ml_models()

    def load_ml_models(self):
        try:
            self.models['crop_model'] = joblib.load(os.path.join(self.models_dir, "crop_model.pkl"))
            self.models['le_crop'] = joblib.load(os.path.join(self.models_dir, "crop_label_encoder.pkl"))
            self.models['fert_model'] = joblib.load(os.path.join(self.models_dir, "fertilizer_model_Xg.pkl"))
            self.models['le_fert'] = joblib.load(os.path.join(self.models_dir, "fertilizer_label_encoder.pkl"))
            print("Lightweight ML Models loaded.")
        except Exception as e:
            print(f"ML Models loading error: {e}")

    def predict_disease(self, img):
        results = []
        confidences = []
        
        configs = {
            "VGG19.h5": 224, "VGG16.h5": 224, "resnet101v2.h5": 224, "InceptionV3.h5": 299
        }

        for m_name, size in configs.items():
            path = os.path.join(self.models_dir, m_name)
            try:
                print(f"Lazy loading {m_name}...")
                current_model = load_model(path, compile=False)
                
                img_resized = img.resize((size, size))
                x = image.img_to_array(img_resized) / 255.0
                x = np.expand_dims(x, axis=0)
                
                preds = current_model.predict(x, verbose=0)[0]
                idx = np.argmax(preds)
                results.append(idx)
                confidences.append(preds[idx])

                del current_model
                tf.keras.backend.clear_session()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing {m_name}: {e}")

        if not results:
            return {"error": "All models failed to load"}

        occurence = Counter(results)
        most_common, count = occurence.most_common(1)[0]
        final_idx = most_common if count >= 2 else results[np.argmax(confidences)]
        
        return {
            "disease": IDX_TO_CLASS[final_idx],
            "confidence": float(confidences[results.index(final_idx)]),
            "agreement": f"{count}/4"
        }


    def recommend_crop_and_fert(self, env_data):
        """Fertilizer Recommendtion"""
        df = pd.DataFrame([env_data])
        
        crop_pred = self.models['crop_model'].predict(df)
        crop_name = self.models['le_crop'].inverse_transform(crop_pred)[0]
        
        df['Crop'] = crop_name
        fert_pred = self.models['fert_model'].predict(df)
        fert_name = self.models['le_fert'].inverse_transform(fert_pred)[0]
        
        return {
            "recommended_crop": crop_name,
            "recommended_fertilizer": fert_name
        }


# ==========================================
# 4. Application Entry (Interactive CLI)
# ==========================================
if __name__ == "__main__":

    ai_system = AgriculturalAI(models_dir="models")

    print("\n==============================")
    print(" Intelligent Agricultural AI ")
    print("==============================")

    print("\nSelect Service:")
    print("1 - Plant Disease Detection")
    print("2 - Recommend Crop & Fertilizer from Soil Data")
    print("3 - Recommend Fertilizer for Specific Crop")

    choice = input("\nEnter your choice (1/2/3): ")

    # =====================================
    # 1️⃣ Plant Disease Detection
    # =====================================
    if choice == "1":

        img_path = os.path.join(
            "..",
            "data",
            "val_imgs",
            "PlantVillage",
            "val",
            "Apple___Cedar_apple_rust",
            "4e6676b6-154c-4f7d-a355-bcc00a397c3d___FREC_C.Rust 9853.jpg"
        )        # img = image.load_img(img_path, target_size=(size, size))
            
        if os.path.exists(img_path):
            img = image.load_img(img_path)
            result = ai_system.predict_disease(img)

            print("\n--- Diagnosis Result ---")
            print("Disease    :", result["disease"])
            print("Confidence :", result["confidence"])
            print("Agreement  :", result["agreement"])

        else:
            print("Image not found.")

    # =====================================
    # 2️⃣ Soil → Crop + Fertilizer
    # =====================================
    elif choice == "2":

        print("\nEnter Soil & Environment Data:")

        Nitrogen = float(input("Nitrogen: "))
        Phosphorus = float(input("Phosphorus: "))
        Potassium = float(input("Potassium: "))
        pH = float(input("pH: "))
        Rainfall = float(input("Rainfall: "))
        Temperature = float(input("Temperature: "))
        Soil_color = input("Soil Color: ")

        env_data = {
            "Nitrogen": Nitrogen,
            "Phosphorus": Phosphorus,
            "Potassium": Potassium,
            "pH": pH,
            "Rainfall": Rainfall,
            "Temperature": Temperature,
            "Soil_color": Soil_color
        }

        result = ai_system.recommend_crop_and_fert(env_data)

        print("\n--- Recommendation ---")
        print("Recommended Crop       :", result["recommended_crop"])
        print("Recommended Fertilizer :", result["recommended_fertilizer"])

    # =====================================
    # 3️⃣ Crop → Fertilizer
    # =====================================
    elif choice == "3":

        print("\nEnter Soil Data + Crop")

        Nitrogen = float(input("Nitrogen: "))
        Phosphorus = float(input("Phosphorus: "))
        Potassium = float(input("Potassium: "))
        pH = float(input("pH: "))
        Rainfall = float(input("Rainfall: "))
        Temperature = float(input("Temperature: "))
        Soil_color = input("Soil Color: ")
        Crop = input("Crop Name: ")

        df = pd.DataFrame([{
            "Nitrogen": Nitrogen,
            "Phosphorus": Phosphorus,
            "Potassium": Potassium,
            "pH": pH,
            "Rainfall": Rainfall,
            "Temperature": Temperature,
            "Soil_color": Soil_color,
            "Crop": Crop
        }])

        fert_pred = ai_system.models['fert_model'].predict(df)
        fert_name = ai_system.models['le_fert'].inverse_transform(fert_pred)[0]

        print("\n--- Fertilizer Recommendation ---")
        print("Recommended Fertilizer :", fert_name)

    else:
        print("Invalid choice.")









