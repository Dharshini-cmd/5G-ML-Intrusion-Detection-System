# predict_manual.py - User enters values manually
import joblib
import numpy as np
import tensorflow as tf
import os
import pandas as pd

SAVE_DIR = r'C:\Users\suraj\Desktop\INTERNSHIP\Yenepoya_Intern\5G_IDS_Results-20260410T060035Z-3-001\5G_IDS_Results\saved_model\\'
DATA_PATH = r'C:\Users\suraj\Desktop\INTERNSHIP\Yenepoya_Intern\5G_IDS_Results-20260410T060035Z-3-001\5G_IDS_Results\final_balanced_5g_dataset.csv'

print("="*60)
print("5G DISTRIBUTED IDS - MANUAL PREDICTION")
print("="*60)

# Load model and files
model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'best_5g_ids_model.h5'))
scaler = joblib.load(os.path.join(SAVE_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(SAVE_DIR, 'feature_names.pkl'))
df = pd.read_csv(DATA_PATH)

print(f"✅ Model loaded successfully!")
print(f"✅ Model performance: 97.5% accuracy, 100% attack detection")

def predict(values):
    """Predict if traffic is normal or attack"""
    input_array = np.array(values, dtype=np.float32).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict(input_scaled, verbose=0)[0][0]
    return probability

# Get real examples from dataset (for reference)
normal_example = df[df['is_attack'] == 0][feature_names].iloc[0].values
attack_example = df[df['is_attack'] == 1][feature_names].iloc[0].values

print("\n" + "="*60)
print("REFERENCE VALUES (for copy-paste)")
print("="*60)

print("\n📝 NORMAL Traffic Values:")
normal_str = ','.join([f"{x:.6f}" for x in normal_example])
print(normal_str)

print("\n📝 ATTACK Traffic Values:")
attack_str = ','.join([f"{x:.6f}" for x in attack_example])
print(attack_str)

print("\n" + "="*60)
print("FEATURE NAMES (Enter values in this order)")
print("="*60)

for i, name in enumerate(feature_names, 1):
    print(f"{i:2}. {name}")

print("\n" + "="*60)
print("MANUAL ENTRY MODE")
print("="*60)

while True:
    print("\n" + "-"*50)
    print("OPTIONS:")
    print("  1. Enter values manually one by one")
    print("  2. Paste all 20 values at once (comma-separated)")
    print("  3. Use NORMAL example")
    print("  4. Use ATTACK example")
    print("  5. Exit")
    
    choice = input("\nChoose option (1-5): ").strip()
    
    if choice == '5' or choice.lower() == 'exit':
        print("Goodbye! 👋")
        break
    
    values = []
    
    if choice == '1':
        # Option 1: Enter one by one
        print("\n📝 Enter each value one by one:")
        print("-"*40)
        for i, name in enumerate(feature_names, 1):
            while True:
                try:
                    val = float(input(f"Enter {i}. {name}: "))
                    values.append(val)
                    break
                except ValueError:
                    print("❌ Invalid number. Please enter a numeric value.")
        print("\n✅ All values entered!")
        
    elif choice == '2':
        # Option 2: Paste all at once
        print("\n📝 Paste 20 comma-separated values:")
        user_input = input("Values: ").strip()
        try:
            values = [float(x.strip()) for x in user_input.split(',')]
            if len(values) != 20:
                print(f"❌ Need exactly 20 values. You entered {len(values)}")
                continue
        except ValueError:
            print("❌ Invalid input. Use numbers separated by commas")
            continue
            
    elif choice == '3':
        # Option 3: Use normal example
        values = normal_example
        print("\n✅ Using NORMAL example")
        
    elif choice == '4':
        # Option 4: Use attack example
        values = attack_example
        print("\n✅ Using ATTACK example")
        
    else:
        print("❌ Invalid choice. Please select 1-5")
        continue
    
    # Make prediction
    probability = predict(values)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    
    if probability > 0.5:
        print(f"⚠️  RESULT: ATTACK DETECTED!")
        print(f"   Confidence: {probability*100:.2f}%")
        if probability > 0.8:
            print(f"   Threat Level: HIGH - Immediate action required!")
        elif probability > 0.6:
            print(f"   Threat Level: MEDIUM - Investigate soon")
        else:
            print(f"   Threat Level: LOW - Monitor")
    else:
        print(f"✅ RESULT: NORMAL TRAFFIC")
        print(f"   Confidence: {(1-probability)*100:.2f}%")
        if (1-probability) > 0.9:
            print(f"   Safety Level: HIGH - Traffic appears safe")
        elif (1-probability) > 0.7:
            print(f"   Safety Level: MEDIUM - Likely normal")
        else:
            print(f"   Safety Level: LOW - Borderline, monitor")
    
    print(f"\n   Raw Score: {probability:.6f}")
    print(f"   (Score > 0.5 = Attack, Score < 0.5 = Normal)")
    print("="*50)