
# ================================
# A. Imports
# ================================
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.applications import DenseNet121, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ================================
# B. Dataset Setup
# ================================
DATA_DIR = r"C:\Users\Nandhitha K\Downloads\COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical"
)

# ================================
# C. Model Setup
# ================================
base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
# Alternative: EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation="softmax")(x)  # 3 classes: COVID19, Pneumonia, Normal

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base CNN layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_gen, validation_data=val_gen, epochs=10)

# ================================
# D. Grad-CAM Function
# ================================
def get_gradcam(img_array, model, last_conv_layer_name="conv5_block16_concat"):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-7

    return heatmap, class_idx.numpy()

# ================================
# E. Prediction Function
# ================================
class_labels = list(train_gen.class_indices.keys())
print("Class labels:", class_labels)

def predict_image(img_path, model, img_size=224):
    img = image.load_img(img_path, target_size=(img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    plt.imshow(img)
    plt.title(f"Predicted: {class_labels[pred_class]} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()

    return class_labels[pred_class], confidence

# Example prediction
test_img = os.path.join(TEST_DIR, "COVID19", "some_covid_image.jpeg")  # Replace with actual test file
predict_image(test_img, model)

# ================================
# F. Visualization: Grad-CAM
# ================================
x = image.img_to_array(image.load_img(test_img, target_size=(img_size, img_size)))
x = np.expand_dims(x, axis=0) / 255.0

heatmap, pred_class = get_gradcam(x, model)

img_orig = cv2.imread(test_img)
heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img_orig

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM Heatmap")
plt.imshow(cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# ================================
# G. Accuracy & Loss Graphs
# ================================
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss vs Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ================================
# H. Confusion Matrix & Report
# ================================
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# ================================
# I. ROC-AUC Curve
# ================================
y_true_bin = label_binarize(y_true, classes=list(range(len(class_labels))))

plt.figure(figsize=(8, 6))
for i, class_name in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)
plt.plot(fpr_micro, tpr_micro, linestyle="--", color="black", label=f"Micro-average (AUC = {roc_auc_micro:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Chance", alpha=0.7)
plt.title("ROC-AUC Curve (Validation Set)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

# ================================
# J. Severity Estimation
# ================================
def calculate_severity(heatmap, threshold=0.5):
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) + 1e-7)
    activated_pixels = np.sum(heatmap > threshold)
    total_pixels = heatmap.size
    severity_score = (activated_pixels / total_pixels) * 100

    if severity_score < 20:
        return "Mild", severity_score, "green"
    elif severity_score < 50:
        return "Moderate", severity_score, "orange"
    else:
        return "Severe", severity_score, "red"

def predict_with_severity(img_path, model, class_labels, last_conv_layer="conv5_block16_concat"):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    heatmap, _ = get_gradcam(x, model, last_conv_layer)

    img_orig = cv2.imread(img_path)
    img_orig = cv2.resize(img_orig, (224, 224))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap_colored, 0.4, 0)

    severity_level, severity_score, color = None, None, None
    if class_labels[pred_class] in ["COVID19", "Pneumonia"]:
        severity_level, severity_score, color = calculate_severity(heatmap)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    title = f"Predicted: {class_labels[pred_class]} ({confidence*100:.2f}%)"
    if severity_level:
        title += f"\nSeverity: {severity_level} ({severity_score:.1f}%)"
    plt.title(title, color=color)
    plt.axis("off")
    plt.show()

    return {
        "prediction": class_labels[pred_class],
        "confidence": confidence,
        "severity_level": severity_level,
        "severity_score": severity_score
    }

# Example with severity
result = predict_with_severity(test_img, model, class_labels)
print(result)

# ================================
# K. Save Model
# ================================
model.save("covid_pneumonia_normal_model.h5")
print("✅ Model saved successfully!")
