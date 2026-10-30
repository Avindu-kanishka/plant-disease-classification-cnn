import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

SEED = 42

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32

EPOCHS = 12
LEARNING_RATE = 5e-6

PLANTDOC_WEIGHT = 2.0
PLANTVILLAGE_WEIGHT = 1.0

OVERSAMPLE_FACTOR = 3

TARGET_CLASSES = [
    "Tomato___Leaf_Mold",
    "Tomato___Tomato_mosaic_virus"
]

tf.keras.utils.set_random_seed(SEED)


# =====================================================
# 2. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed" / "plantdoc_more_data"

TRAIN_CSV = DATA_DIR / "combined_train.csv"
VAL_CSV = DATA_DIR / "combined_val.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

INITIAL_MODEL_PATH = MODEL_DIR / "plantdoc_aware_mobilenetv2.keras"

MODEL_PATH = MODEL_DIR / "plantdoc_targeted_mobilenetv2.keras"

RESULT_PATH = REPORT_DIR / "plantdoc_targeted_training_results.json"


# =====================================================
# 3. CLASS DEFINITIONS
# =====================================================

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

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# =====================================================
# 4. LOAD DATA
# =====================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

required_columns = {"image_path", "label", "source"}

for name, df in [
    ("Training", train_df),
    ("Validation", val_df)
]:

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"{name} CSV missing columns: {missing}")

    if not df["label"].isin(CLASS_NAMES).all():
        raise ValueError(f"Unknown class in {name} dataset.")

    if not df["source"].isin(["PlantDoc", "PlantVillage"]).all():
        raise ValueError(f"Unknown source in {name} dataset.")


print("Original training images:", len(train_df))
print("Validation images:", len(val_df))


# =====================================================
# 5. CALCULATE CLASS WEIGHTS
# =====================================================

print("\nCalculating class weights...")

# Calculate using original data BEFORE oversampling.
y_original = train_df["label"].map(CLASS_TO_ID).values

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_original
)

# Reduce extreme class weighting.
class_weights = np.sqrt(class_weights)

# Normalize average weight.
class_weights = class_weights / np.mean(class_weights)

class_weight_dict = {
    i: float(weight)
    for i, weight in enumerate(class_weights)
}

print("\nClass weights:")

for name, weight in zip(CLASS_NAMES, class_weights):
    print(f"{name}: {weight:.3f}")


# =====================================================
# 6. TARGETED OVERSAMPLING
# =====================================================

print("\n========== TARGETED OVERSAMPLING ==========")

target_mask = (
    (train_df["source"] == "PlantDoc")
    &
    (train_df["label"].isin(TARGET_CLASSES))
)

target_df = train_df[target_mask].copy()

print("\nTarget images before oversampling:")

print(
    target_df.groupby(["label", "source"]).size()
)

# Add two extra copies, making 3 copies in total.
extra_copies = pd.concat(
    [target_df] * (OVERSAMPLE_FACTOR - 1),
    ignore_index=True
)

train_df = pd.concat(
    [train_df, extra_copies],
    ignore_index=True
)

train_df = train_df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("\nTraining images after oversampling:", len(train_df))

print("\nNew training distribution:")

print(
    train_df.groupby(["label", "source"]).size()
)


# =====================================================
# 7. SAMPLE WEIGHTS
# =====================================================

def calculate_sample_weights(df):

    labels = df["label"].map(CLASS_TO_ID).values

    source_weights = np.where(
        df["source"].values == "PlantDoc",
        PLANTDOC_WEIGHT,
        PLANTVILLAGE_WEIGHT
    )

    weights = np.array([
        source_weights[i] * class_weights[labels[i]]
        for i in range(len(df))
    ])

    weights = weights / np.mean(weights)

    return weights.astype(np.float32)


train_weights = calculate_sample_weights(train_df)
val_weights = calculate_sample_weights(val_df)

print("\nAverage training weight:", train_weights.mean())
print("Average validation weight:", val_weights.mean())


# =====================================================
# 8. IMAGE PREPROCESSING
# =====================================================

def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label


def create_dataset(df, sample_weights, training=False):

    paths = df["image_path"].astype(str).values

    labels = df["label"].map(CLASS_TO_ID).values.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels, sample_weights)
    )

    def load_weighted_image(path, label, weight):

        image, label = load_image(path, label)

        return image, label, weight

    dataset = dataset.map(
        load_weighted_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(
    train_df,
    train_weights,
    training=True
)

val_dataset = create_dataset(
    val_df,
    val_weights,
    training=False
)


# =====================================================
# 9. LOAD EXPERIMENT 02 MODEL
# =====================================================

print("\nLoading Experiment 02 model...")

if not INITIAL_MODEL_PATH.exists():
    raise FileNotFoundError(INITIAL_MODEL_PATH)

model = tf.keras.models.load_model(
    INITIAL_MODEL_PATH
)

print("Model loaded successfully.")

print("\nModel layers:")

for i, layer in enumerate(model.layers):
    print(i, layer.name, type(layer).__name__)


# =====================================================
# 10. FINE-TUNING
# =====================================================

print("\n========== FINE-TUNING ==========")

base_model = model.get_layer("mobilenetv2_1.00_160")

base_model.trainable = True

# Freeze all layers except the last 30.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:

    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


print("\nTrainable backbone layers:")

for layer in base_model.layers[-30:]:

    if layer.trainable:
        print(layer.name)


# =====================================================
# 11. COMPILE MODEL
# =====================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"],

    weighted_metrics=["accuracy"]

)


# =====================================================
# 12. CALLBACKS
# =====================================================

callbacks = [

    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_weighted_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_weighted_accuracy",
        mode="max",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )

]


# =====================================================
# 13. TRAIN MODEL
# =====================================================

print("\n========== EXPERIMENT 03 TRAINING ==========")

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=EPOCHS,

    callbacks=callbacks

)


# =====================================================
# 14. LOAD BEST MODEL
# =====================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# =====================================================
# 15. FINAL VALIDATION
# =====================================================

print("\n========== FINAL VALIDATION ==========")

results = best_model.evaluate(
    val_dataset,
    verbose=1,
    return_dict=True
)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")


# =====================================================
# 16. SAVE RESULTS
# =====================================================

training_results = {

    "experiment": "Experiment 03 - Targeted Class Improvement",

    "initial_model": str(INITIAL_MODEL_PATH.name),

    "target_classes": TARGET_CLASSES,

    "oversample_factor": OVERSAMPLE_FACTOR,

    "plantdoc_source_weight": PLANTDOC_WEIGHT,

    "plantvillage_source_weight": PLANTVILLAGE_WEIGHT,

    "original_training_images": int(len(train_df) - len(extra_copies)),

    "training_images_after_oversampling": int(len(train_df)),

    "validation_images": int(len(val_df)),

    "best_validation_loss": float(results["loss"]),

    "validation_accuracy": float(results["accuracy"]),

    "validation_weighted_accuracy": float(
        results["weighted_accuracy"]
    ),

    "epochs_completed": len(history.history["loss"])

}


with open(RESULT_PATH, "w") as f:

    json.dump(
        training_results,
        f,
        indent=4
    )


print("\n======================================")

print("Experiment 03 training completed!")

print("Model saved:", MODEL_PATH)

print("Results saved:", RESULT_PATH)

print("======================================")