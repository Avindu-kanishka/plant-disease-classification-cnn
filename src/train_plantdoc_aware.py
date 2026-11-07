import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
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

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 15

PLANTDOC_WEIGHT = 3.0
PLANTVILLAGE_WEIGHT = 1.0

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

MODEL_PATH = MODEL_DIR / "plantdoc_aware_mobilenetv2.keras"

RESULT_PATH = REPORT_DIR / "plantdoc_aware_training_results.json"


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

for df_name, df in [
    ("Training", train_df),
    ("Validation", val_df)
]:
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{df_name} CSV missing columns: {missing}"
        )

    if not df["label"].isin(CLASS_NAMES).all():
        raise ValueError(
            f"Unknown class found in {df_name} dataset."
        )

    if not df["source"].isin(
        ["PlantDoc", "PlantVillage"]
    ).all():
        raise ValueError(
            f"Unknown source found in {df_name} dataset."
        )


print("Training images:", len(train_df))
print("Validation images:", len(val_df))

print("\nTraining source distribution:")
print(train_df["source"].value_counts())

print("\nValidation source distribution:")
print(val_df["source"].value_counts())


# =====================================================
# 5. CALCULATE CLASS WEIGHTS
# =====================================================

print("\nCalculating class weights...")

y_train = train_df["label"].map(CLASS_TO_ID).values

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=y_train
)

# Square root reduces extreme class weighting.
class_weights = np.sqrt(class_weights)

# Normalize average class weight to 1.
class_weights = class_weights / np.mean(class_weights)

class_weight_dict = {
    i: float(weight)
    for i, weight in enumerate(class_weights)
}

print("\nClass weights:")

for class_name, weight in zip(
    CLASS_NAMES, class_weights
):
    print(f"{class_name}: {weight:.3f}")


# =====================================================
# 6. CALCULATE SOURCE + CLASS SAMPLE WEIGHTS
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

    # Normalize average sample weight to 1.
    weights = weights / np.mean(weights)

    return weights.astype(np.float32)


train_weights = calculate_sample_weights(train_df)
val_weights = calculate_sample_weights(val_df)

print("\nAverage training sample weight:", train_weights.mean())
print("Average validation sample weight:", val_weights.mean())


# =====================================================
# 7. IMAGE PREPROCESSING
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

    image = preprocess_input(image)

    return image, label


def create_dataset(df, sample_weights=None, training=False):

    paths = df["image_path"].astype(str).values

    labels = df["label"].map(CLASS_TO_ID).values.astype(
        np.int32
    )

    if sample_weights is None:

        dataset = tf.data.Dataset.from_tensor_slices(
            (paths, labels)
        )

        dataset = dataset.map(
            load_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    else:

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

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

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
# 8. DATA AUGMENTATION
# =====================================================

augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1),

], name="data_augmentation")


# =====================================================
# 9. BUILD MODEL FROM IMAGENET WEIGHTS
# =====================================================

print("\nBuilding MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze backbone initially.
base_model.trainable = False

inputs = layers.Input(
    shape=(160, 160, 3)
)

x = augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

print("\nModel summary:")

model.summary()


# =====================================================
# 10. CALLBACKS
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
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

]


# =====================================================
# 11. PHASE 1 - TRAIN CLASSIFICATION HEAD
# =====================================================

print("\n========== PHASE 1: HEAD TRAINING ==========")

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"],

    weighted_metrics=["accuracy"]

)


history1 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=HEAD_EPOCHS,

    callbacks=callbacks

)


# =====================================================
# 12. PHASE 2 - FINE TUNING
# =====================================================

print("\n========== PHASE 2: FINE TUNING ==========")

base_model.trainable = True

# Freeze all layers except last 30.
for layer in base_model.layers[:-30]:

    layer.trainable = False

# Keep Batch Normalization layers frozen.
for layer in base_model.layers:

    if isinstance(layer, layers.BatchNormalization):

        layer.trainable = False


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"],

    weighted_metrics=["accuracy"]

)


history2 = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=callbacks

)


# =====================================================
# 13. LOAD BEST CHECKPOINT
# =====================================================

print("\nLoading best saved model...")

best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# =====================================================
# 14. FINAL VALIDATION EVALUATION
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
# 15. SAVE RESULTS
# =====================================================

training_results = {

    "experiment": "Experiment 02 - PlantDoc-Aware MobileNetV2",

    "initial_weights": "ImageNet",

    "classes": CLASS_NAMES,

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "plantdoc_source_weight": PLANTDOC_WEIGHT,

    "plantvillage_source_weight": PLANTVILLAGE_WEIGHT,

    "training_images": int(len(train_df)),

    "validation_images": int(len(val_df)),

    "training_source_distribution": {
        str(k): int(v)
        for k, v in train_df["source"].value_counts().items()
    },

    "validation_source_distribution": {
        str(k): int(v)
        for k, v in val_df["source"].value_counts().items()
    },

    "class_weights": class_weight_dict,

    "best_validation_loss": float(results["loss"]),

    "validation_accuracy": float(results["accuracy"]),

    "validation_weighted_accuracy": float(
        results["weighted_accuracy"]
    )

}


with open(RESULT_PATH, "w") as f:

    json.dump(
        training_results,
        f,
        indent=4
    )


print("\n======================================")

print("Experiment 02 training completed!")

print("Model saved:", MODEL_PATH)

print("Results saved:", RESULT_PATH)

print("======================================")