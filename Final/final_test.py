# predict_final.py - Final working version
import joblib
import numpy as np
import os
import tensorflow as tf

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MODEL_PATH = r'C:\Users\suraj\Desktop\INTERNSHIP\Yenepoya_Intern\5G_IDS_Results-20260410T060035Z-3-001\5G_IDS_Results\\'

print("="*60)
print("AEGIS 5G")
print("="*60)

# Load scaler and features
print("\n📂 Loading files...")
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))
print(f"✅ Loaded {len(feature_names)} features")

# Load model (prefer .keras format)
keras_path = os.path.join(MODEL_PATH, 'best_5g_ids_model.keras')
h5_path = os.path.join(MODEL_PATH, 'best_5g_ids_model.h5')

if os.path.exists(keras_path):
    model = tf.keras.models.load_model(keras_path)
    print("✅ Loaded .keras model")
elif os.path.exists(h5_path):
    model = tf.keras.models.load_model(h5_path)
    print("✅ Loaded .h5 model")
else:
    raise FileNotFoundError("No model file found!")

print("✅ Model ready for predictions!")

def predict(values):
    """Predict if traffic is normal or attack"""
    input_array = np.array(values, dtype=np.float32).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return probability

print("\n" + "="*60)
print("PREDICTION MODE")
print("="*60)

print("\n📝 COPY-PASTE THIS for NORMAL traffic:")
print("0.5,500,64,20,50,5,100,50,1,50000,100000,1000,1500,0.5,10,0.5,1000,10,0,64")

print("\n📝 COPY-PASTE THIS for ATTACK traffic:")
print("2.5,100,32,50,200,8,10,5,2,10000000,20000000,50000,1500,0.9,2,0.9,50000,2,2,64")

print("\n📝 COPY-PASTE THIS for MIXED traffic (should be NORMAL):")
print("1.0,300,128,30,80,4,200,100,1,25000,50000,2000,1400,0.3,20,0.3,2000,20,1,128")

while True:
    print("\n" + "-"*50)
    user_input = input("Enter 20 values (or 'quit'): ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye! 👋")
        break
    
    if not user_input:
        continue
    
    try:
        values = [float(x.strip()) for x in user_input.split(',')]
        
        if len(values) != 20:
            print(f"❌ Need exactly 20 values. You entered {len(values)}")
            continue
        
        probability = predict(values)
        
        print("\n" + "="*50)
        if probability > 0.5:
            print(f"⚠️  RESULT: ATTACK DETECTED!")
            print(f"   Confidence: {probability*100:.2f}%")
            if probability > 0.8:
                print(f"   Severity: HIGH")
            elif probability > 0.6:
                print(f"   Severity: MEDIUM")
            else:
                print(f"   Severity: LOW")
        else:
            print(f"✅ RESULT: NORMAL TRAFFIC")
            print(f"   Confidence: {(1-probability)*100:.2f}%")
        print(f"   Raw Score: {probability:.6f}")
        print("="*50)
        
    except ValueError:
        print("❌ Invalid input. Enter numbers separated by commas")
    except Exception as e:
        print(f"❌ Error: {e}")