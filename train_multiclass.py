"""
RetinaScope — Multi-Disease Retinal Classification Training Pipeline
=====================================================================
Supports multi-class classification for:
  - Normal
  - Diabetic Retinopathy
  - Glaucoma
  - Cataract / AMD

Usage:
  1. Organize your dataset directory as:
     dataset/
       ├── train/
       │     ├── Normal/
       │     ├── Diabetic_Retinopathy/
       │     ├── Glaucoma/
       │     └── AMD/
       └── val/
             ├── Normal/
             ├── Diabetic_Retinopathy/
             ├── Glaucoma/
             └── AMD/

  2. Run training:
     python train_multiclass.py --data_dir ./dataset --epochs 25 --batch_size 32
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_model(num_classes, img_size=(224, 224)):
    """Builds a transfer-learning classifier using MobileNetV2."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze the base model initially
    base_model.trainable = False

    inputs = layers.Input(shape=(*img_size, 3))
    
    # Preprocessing / Augmentation layers inside the model
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)
    
    # Preprocessing expected by MobileNetV2 (scales from [0, 255] to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="RetinaScope_MultiClass")
    return model, base_model

def train(data_dir, epochs=25, batch_size=32, img_size=(224, 224), output_model="retinal_cnn_model_v2.keras"):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        # Allow flat directory with auto-split
        print(f"[INFO] '{train_dir}' not found. Loading directly from '{data_dir}' with 80/20 train-val split...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=42,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=42,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        class_names = train_ds.class_names
    else:
        print(f"[INFO] Loading from train/ and val/ subdirectories...")
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical"
        )
        class_names = train_ds.class_names

    num_classes = len(class_names)
    print(f"\n[INFO] Detected {num_classes} classes: {class_names}")

    # Optimize data loading pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 1. Build Model
    model, base_model = build_model(num_classes, img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    model.summary()

    # Callbacks
    cb_list = [
        callbacks.ModelCheckpoint(
            output_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n--- Phase 1: Training Classification Head ---")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2 or 10,
        callbacks=cb_list
    )

    print("\n--- Phase 2: Fine-Tuning Top Layers of MobileNetV2 ---")
    base_model.trainable = True
    # Freeze earlier layers, fine-tune the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=cb_list
    )

    # Save classes.json for app.py to dynamically read
    import json
    classes_path = os.path.join(os.path.dirname(output_model) or ".", "classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[INFO] Saved class labels mapping to {classes_path}")

    print(f"\n[SUCCESS] Model saved to {output_model} with classes: {class_names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-disease retinal classification model")
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to dataset root folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_model", type=str, default="retinal_cnn_model_v2.keras", help="Output .keras filename")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Directory '{args.data_dir}' does not exist.")
        print("Please download a dataset and arrange it in class folders, e.g.:")
        print("  dataset/Normal/")
        print("  dataset/Diabetic_Retinopathy/")
        print("  dataset/Glaucoma/")
        print("  dataset/AMD/")
        exit(1)

    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, output_model=args.output_model)
