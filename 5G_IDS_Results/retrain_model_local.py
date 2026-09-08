# train_local.py - Train 5G IDS model locally
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("5G DISTRIBUTED IDS - LOCAL TRAINING")
print("="*60)

# ============================================
# CONFIGURATION
# ============================================

# Update this path to where your CSV file is located
DATA_PATH = r'C:\Users\suraj\Desktop\INTERNSHIP\Yenepoya_Intern\5G_IDS_Results-20260410T060035Z-3-001\Final\final_balanced_5g_dataset.csv'

# Where to save the trained models
SAVE_DIR = r'C:\Users\suraj\Desktop\INTERNSHIP\Yenepoya_Intern\5G_IDS_Results-20260410T060035Z-3-001\Final\saved_model\\'

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================
# STEP 1: LOAD DATA
# ============================================

print("\n📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")

# Identify features (exclude target columns)
target_cols = ['is_attack', 'Label', 'split']
feature_cols = [col for col in df.columns if col not in target_cols]
X = df[feature_cols].values
y = df['is_attack'].values

print(f"✅ Features: {len(feature_cols)}")
print(f"   Class distribution - Normal: {sum(y==0)}, Attack: {sum(y==1)}")

# ============================================
# STEP 2: TRAIN/VAL/TEST SPLIT
# ============================================

print("\n📊 Creating train/val/test split...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"   Training: {len(X_train)} samples")
print(f"   Validation: {len(X_val)} samples")
print(f"   Test: {len(X_test)} samples")

# ============================================
# STEP 3: STANDARDIZE FEATURES
# ============================================

print("\n📊 Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✅ Features standardized (mean=0, std=1)")

# ============================================
# STEP 4: BUILD DNN MODEL
# ============================================

print("\n🤖 Building DNN model...")
model = keras.Sequential([
    layers.Input(shape=(len(feature_cols),)),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

model.summary()

# ============================================
# STEP 5: TRAIN MODEL
# ============================================

print("\n🏋️ Training model...")
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ============================================
# STEP 6: EVALUATE MODEL
# ============================================

print("\n📊 Evaluating model on test set...")
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Test Results:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f} ({f1*100:.2f}%)")

# ============================================
# STEP 7: SAVE MODEL AND FILES
# ============================================

print("\n💾 Saving model and files...")

# Save in different formats
# Format 1: H5 (most compatible)
model.save(os.path.join(SAVE_DIR, 'best_5g_ids_model.h5'))
print(f"   ✅ Saved: best_5g_ids_model.h5")

# Format 2: Keras format
model.save(os.path.join(SAVE_DIR, 'best_5g_ids_model.keras'))
print(f"   ✅ Saved: best_5g_ids_model.keras")

# Format 3: SavedModel
tf.saved_model.save(model, os.path.join(SAVE_DIR, 'saved_model'))
print(f"   ✅ Saved: saved_model/")

# Save scaler
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
print(f"   ✅ Saved: scaler.pkl")

# Save feature names
joblib.dump(feature_cols, os.path.join(SAVE_DIR, 'feature_names.pkl'))
print(f"   ✅ Saved: feature_names.pkl")

# Save results
results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'features': feature_cols,
    'n_features': len(feature_cols)
}

import json
with open(os.path.join(SAVE_DIR, 'training_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✅ Saved: training_results.json")

# ============================================
# STEP 8: GENERATE CONFUSION MATRIX
# ============================================

print("\n📊 Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal (0)', 'Attack (1)'],
            yticklabels=['Normal (0)', 'Attack (1)'])
plt.title(f'Confusion Matrix (Accuracy: {accuracy*100:.2f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()
print(f"   ✅ Saved: confusion_matrix.png")

# ============================================
# STEP 9: SAVE TRAINING HISTORY PLOT
# ============================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'), dpi=150)
plt.show()
print(f"   ✅ Saved: training_history.png")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*60)
print("TRAINING COMPLETE! 🎉")
print("="*60)
print(f"\n📁 All files saved to: {SAVE_DIR}")
print("\n📋 Saved files:")
print("   1. best_5g_ids_model.h5       - H5 model format")
print("   2. best_5g_ids_model.keras    - Keras format")
print("   3. saved_model/               - TensorFlow SavedModel")
print("   4. scaler.pkl                 - StandardScaler")
print("   5. feature_names.pkl          - Feature names")
print("   6. training_results.json      - Performance metrics")
print("   7. confusion_matrix.png       - Confusion matrix plot")
print("   8. training_history.png       - Training curves")
print("\n🏆 Final Performance:")
print(f"   Accuracy:  {accuracy*100:.2f}%")
print(f"   Precision: {precision*100:.2f}%")
print(f"   Recall:    {recall*100:.2f}%")
print(f"   F1-Score:  {f1*100:.2f}%")