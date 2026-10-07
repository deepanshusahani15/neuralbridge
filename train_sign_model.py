"""
Train a sign language classifier from the images in dataset/<sign>/

Usage:
    python train_sign_model.py
    python train_sign_model.py --epochs 25 --data dataset
    python train_sign_model.py --scratch        # small CNN, no pretrained weights download

Outputs:
    sign_model.keras   the trained model
    labels.json        class names, in the order the model uses
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TensorFlow noise

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

IMG_EXT = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser(description="Train sign language model")
    p.add_argument("--data", default="dataset", help="Folder with one subfolder per sign")
    p.add_argument("--out", default="sign_model.keras", help="Where to save the model")
    p.add_argument("--labels", default="labels.json", help="Where to save class names")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--finetune-epochs", type=int, default=5,
                   help="Extra epochs that fine-tune the pretrained model (0 = skip)")
    p.add_argument("--scratch", action="store_true",
                   help="Use a small CNN trained from scratch (no internet needed)")
    return p.parse_args()


def find_classes(data_dir):
    """Every subfolder with at least one image becomes a class."""
    classes = []
    for d in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, d)
        if not os.path.isdir(path) or d.startswith("."):
            continue
        n = len([f for f in os.listdir(path) if f.lower().endswith(IMG_EXT)])
        if n == 0:
            continue
        note = "" if n >= 100 else "   <-- fewer than 100, capture more"
        print(f"{d:>14}: {n:4d} images{note}")
        classes.append(d)
    return classes


def build_model(num_classes, img_size, scratch):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    # augmentation: only active during training. No horizontal flip, because
    # left/right hand can change the meaning of a sign.
    x = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augment")(inputs)

    base = None
    if scratch:
        x = layers.Rescaling(1 / 255.0)(x)
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
    else:
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)       # MobileNetV2 preprocessing
        base = keras.applications.MobileNetV2(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
        base.trainable = False
        x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])


def main():
    args = parse_args()

    print("Signs found:")
    classes = find_classes(args.data)
    if len(classes) < 2:
        raise SystemExit("Need at least 2 signs (folders with images) to train.")

    size = (args.img_size, args.img_size)
    common = dict(directory=args.data, class_names=classes, image_size=size,
                  batch_size=args.batch, validation_split=0.2, seed=42,
                  label_mode="int")
    train_ds = keras.utils.image_dataset_from_directory(subset="training", **common)
    val_ds = keras.utils.image_dataset_from_directory(subset="validation", **common)
    train_ds = train_ds.cache().shuffle(1000).prefetch(-1)
    val_ds = val_ds.cache().prefetch(-1)

    model, base = build_model(len(classes), args.img_size, args.scratch)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True)

    print("\n--- Training ---")
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop])

    if base is not None and args.finetune_epochs > 0:
        print("\n--- Fine-tuning ---")
        base.trainable = True
        for layer in base.layers[:-30]:          # keep early layers frozen
            layer.trainable = False
        compile_model(model, 1e-5)
        model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs,
                  callbacks=[stop])

    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc * 100:.1f}%")

    # accuracy per sign, to spot weak ones
    y_true, y_pred = [], []
    for x, y in val_ds:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print("Per sign:")
    for i, name in enumerate(classes):
        m = y_true == i
        if m.any():
            print(f"{name:>14}: {(y_pred[m] == i).mean() * 100:5.1f}%  ({m.sum()} val images)")

    model.save(args.out)
    with open(args.labels, "w") as f:
        json.dump(classes, f)
    print(f"\nSaved model -> {args.out}")
    print(f"Saved labels -> {args.labels}")


if __name__ == "__main__":
    main()