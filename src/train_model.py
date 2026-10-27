from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "split"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("Loading image datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        seed=SEED,
    )

    class_names = train_ds.class_names
    print("Classes:", class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        class_names=class_names,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    print("Calculating class weights...")
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=labels,
    )
    class_weights = {
        int(i): float(weight) for i, weight in enumerate(weights)
    }

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ], name="data_augmentation")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(
        len(class_names), activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs, name="tomato_disease_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    model_path = MODEL_DIR / "plant_disease_mobilenetv2.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting transfer-learning training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)

    results = {
        "classes": class_names,
        "image_size": list(IMG_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }

    with open(REPORT_DIR / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(REPORT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print("\nTraining complete!")
    print(json.dumps(results, indent=2))
    print("Saved model:", model_path)

if __name__ == "__main__":
    main()
