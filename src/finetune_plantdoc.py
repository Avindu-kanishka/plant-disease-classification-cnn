from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "reports" / "class_names.json"

PLANTDOC_TRAIN = Path(
    r"D:\projects\Plant disease data set\PlantDoc-Dataset-master\train"
)

OUTPUT_PATH = PROJECT_ROOT / "models" / "plant_disease_mobilenetv2_finetuned.keras"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

CLASS_MAPPING = {
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".webp"}


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
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
            len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    if isinstance(class_names, dict):
        class_names = class_names["classes"]

    paths = []
    labels = []

    for folder_name, model_class in CLASS_MAPPING.items():
        folder = PLANTDOC_TRAIN / folder_name

        if not folder.is_dir():
            raise FileNotFoundError(folder)

        class_index = class_names.index(model_class)

        for image_path in folder.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(image_path))
                labels.append(class_index)

    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)

    if len(paths) == 0:
        raise RuntimeError("No PlantDoc training images found.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print("Training images:", len(train_paths))
    print("Validation images:", len(val_paths))
    print("Classes:", len(np.unique(labels)))

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels)

    model = tf.keras.models.load_model(MODEL_PATH)

    backbone = model.get_layer("mobilenetv2_1.00_160")
    backbone.trainable = True

    fine_tune_from = len(backbone.layers) - 30

    for index, layer in enumerate(backbone.layers):
        layer.trainable = (
            index >= fine_tune_from
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )

    trainable_layers = sum(
        1 for layer in backbone.layers if layer.trainable
    )

    print("Trainable backbone layers:", trainable_layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(OUTPUT_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    if not OUTPUT_PATH.exists():
        model.save(OUTPUT_PATH)

    results = {
        "training_images": len(train_paths),
        "validation_images": len(val_paths),
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "model_path": str(OUTPUT_PATH)
    }

    results_path = PROJECT_ROOT / "reports" / "plantdoc_finetuning_results.json"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFine-tuning completed.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
