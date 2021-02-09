# -*- coding:utf-8 -*-

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, transform
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model(num_class) -> Sequential:
    # モデルの構築
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activation="softmax"))

    # オプティマイザーと評価指標の設定
    adam = Adam(learning_rate=1e-3)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=adam,
        metrics=["accuracy"],
    )

    return model


def train(args):
    # 乱数の固定
    tf.random.set_seed(0)

    # データ読み込み
    with open(glob.glob(f"{args.input_path}/**/train_test_datas.pkl")[0], "br") as f:
        (X_train, y_train), (X_test, y_test) = pickle.load(f)
    with open(glob.glob(f"{args.input_path}/**/labels_idx.pkl")[0], "br") as f:
        label_index = pickle.load(f)

    # data generator の宣言
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        data_format="channels_last",
    )

    # コールバックの設定
    es = EarlyStopping(monitor="val_loss", patience=10)
    log_dir = f"{args.log_path}"
    tb = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True)
    cp = ModelCheckpoint(
        f"{args.output_path}/params.hdf5",
        monitor="val_loss",
        save_best_only=True,
    )

    # データ
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    # model をビルド
    model = build_model(len(label_index))

    # 学習実行
    model.fit_generator(
        train_generator,
        steps_per_epoch=X_train.shape[0] // 32,
        verbose=2,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[es, tb, cp],
    )

    # confusion matrixの作成
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    cm = confusion_matrix(y_test, y_pred, labels=list(label_index.values()))
    cm = pd.DataFrame(cm, columns=label_index.keys(), index=label_index.keys())
    cm.to_csv(f"{args.output_path}/confusion_matrix.csv")


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="aqualium demo")
    parser.add_argument("--input_path", default="/kqi/input")
    parser.add_argument("--output_path", default="/kqi/output/demo")
    parser.add_argument("--log_path", default="/kqi/output/logs")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    train(args)
