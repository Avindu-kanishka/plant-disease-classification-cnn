
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# =====================================================
# CONFIGURATION
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "processed" / "plantdoc_more_data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_CSV = DATA_DIR / "combined_train.csv"
VAL_CSV = DATA_DIR / "combined_val.csv"

MODEL_PATH = MODEL_DIR / "plant_disease_mobilenetv2_7class.keras"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

EPOCHS_HEAD = 10
EPOCHS_FINE = 15

CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

NUM_CLASSES = len(CLASS_NAMES)

# =====================================================
# DIRECTORY CREATION
# =====================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

# =====================================================
# LOAD CSV DATA
# =====================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

class_to_id = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

train_df["label_id"] = train_df["label"].map(class_to_id)
val_df["label_id"] = val_df["label"].map(class_to_id)

if train_df["label_id"].isna().any() or val_df["label_id"].isna().any():
    raise ValueError("Unknown class labels found in CSV files.")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Number of classes:", NUM_CLASSES)

# =====================================================
# CLASS WEIGHTS
# =====================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["label_id"].values
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nClass weights:")
print(class_weights)

# =====================================================
# IMAGE LOADING
# =====================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMG_SIZE)
    image = preprocess_input(image)

    label = tf.cast(label, tf.int32)

    return image, label


def create_dataset(dataframe, training=False):
    paths = dataframe["image_path"].astype(str).values
    labels = dataframe["label_id"].astype(int).values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df, training=True)
val_ds = create_dataset(val_df)

# =====================================================
# DATA AUGMENTATION
# =====================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")

# =====================================================
# BUILD MOBILE NET V2 MODEL
# =====================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =====================================================
# PHASE 1: TRAIN CLASSIFICATION HEAD
# =====================================================

print("\n========== PHASE 1: FEATURE EXTRACTION ==========")

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# PHASE 2: FINE-TUNING
# =====================================================

print("\n========== PHASE 2: FINE-TUNING ==========")

base_model.trainable = True

# Freeze all layers except last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep BatchNormalization layers frozen
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# EVALUATE BEST VALIDATION CHECKPOINT
# =====================================================

print("\n========== VALIDATION EVALUATION ==========")

best_model = tf.keras.models.load_model(MODEL_PATH)

val_loss, val_accuracy = best_model.evaluate(val_ds, verbose=1)

print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")

# =====================================================
# SAVE RESULTS
# =====================================================

results = {
    "model": "MobileNetV2",
    "dataset": "PlantVillage + PlantDoc",
    "num_classes": NUM_CLASSES,
    "training_images": int(len(train_df)),
    "validation_images": int(len(val_df)),
    "class_weights": class_weights,
    "validation_loss": float(val_loss),
    "validation_accuracy": float(val_accuracy),
    "epochs_head_completed": len(history_head.history["loss"]),
    "epochs_finetuning_completed": len(history_fine.history["loss"]),
    "class_names": CLASS_NAMES
}

results_path = REPORT_DIR / "plantdoc_combined_training_results.json"

with open(results_path, "w") as file:
    json.dump(results, file, indent=4)

print("\nTraining completed successfully!")
print("Best model:", MODEL_PATH)
print("Results:", results_path)
