
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PLANTDOC_TRAIN = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train"
)

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "plant_disease_mobilenetv2_7class.keras"
)

RESULTS_PATH = (
    PROJECT_ROOT / "reports" /
    "plantdoc_7class_training_results.json"
)

SPLIT_PATH = (
    PROJECT_ROOT / "reports" /
    "plantdoc_7class_split.json"
)

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 15
SEED = 42

CLASS_MAPPING = {
    "Tomato Early blight leaf": "Early Blight",
    "Tomato leaf bacterial spot": "Bacterial Spot",
    "Tomato leaf late blight": "Late Blight",
    "Tomato leaf mosaic virus": "Mosaic Virus",
    "Tomato leaf yellow virus": "Yellow Leaf Curl Virus",
    "Tomato mold leaf": "Leaf Mold",
    "Tomato Septoria leaf spot": "Septoria Leaf Spot",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".jfif", ".webp"
}


def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    return image, label


def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    return dataset.prefetch(tf.data.AUTOTUNE)


def main():

    tf.keras.utils.set_random_seed(SEED)

    paths = []
    labels = []

    class_names = list(CLASS_MAPPING.values())

    # Load all 7 classes
    for class_index, folder_name in enumerate(CLASS_MAPPING):

        folder = PLANTDOC_TRAIN / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        for image_path in folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                paths.append(str(image_path))
                labels.append(class_index)

    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)

    if len(paths) == 0:
        raise RuntimeError("No training images found.")

    # 70% train, 15% validation, 15% test
    train_val_paths, test_paths, train_val_labels, test_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.15,
            random_state=SEED,
            stratify=labels
        )
    )

    train_paths, val_paths, train_labels, val_labels = (
        train_test_split(
            train_val_paths,
            train_val_labels,
            test_size=(0.15 / 0.85),
            random_state=SEED,
            stratify=train_val_labels
        )
    )

    print("\nDataset Summary")
    print("----------------")
    print("Total images:", len(paths))
    print("Training:", len(train_paths))
    print("Validation:", len(val_paths))
    print("Testing:", len(test_paths))
    print("Classes:", len(class_names))

    # Save split information for later evaluation
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "class_names": class_names,
            "train_paths": train_paths.tolist(),
            "validation_paths": val_paths.tolist(),
            "test_paths": test_paths.tolist(),
            "test_labels": test_labels.tolist()
        }, f, indent=2)

    train_ds = make_dataset(
        train_paths, train_labels, training=True
    )

    val_ds = make_dataset(val_paths, val_labels)

    # Data augmentation
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # Start from ImageNet weights, not previous project models
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = False

    inputs = tf.keras.Input(shape=(160, 160, 3))

    x = augmentation(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = backbone(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    # Class weights
    class_ids = np.unique(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=class_ids,
        y=train_labels
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(class_ids, weights)
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7
    )

    callbacks = [
        checkpoint,
        early_stopping,
        reduce_lr
    ]

    # Phase 1: Train classification head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nPhase 1: Training classification head")

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Phase 2: Fine-tune last 30 backbone layers
    print("\nPhase 2: Fine-tuning MobileNetV2")

    backbone.trainable = True

    fine_tune_from = len(backbone.layers) - 30

    for index, layer in enumerate(backbone.layers):
        layer.trainable = (
            index >= fine_tune_from
            and not isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            )
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # Load best saved model
    best_model = tf.keras.models.load_model(MODEL_PATH)

    val_loss, val_accuracy = best_model.evaluate(
        val_ds,
        verbose=0
    )

    results = {
        "total_images": int(len(paths)),
        "training_images": int(len(train_paths)),
        "validation_images": int(len(val_paths)),
        "test_images": int(len(test_paths)),
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "head_epochs_completed": len(history_head.history["loss"]),
        "fine_tuning_epochs_completed": len(history_fine.history["loss"]),
        "best_validation_accuracy": float(val_accuracy),
        "best_validation_loss": float(val_loss),
        "model_path": str(MODEL_PATH)
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTraining completed!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
