# -*- coding:utf-8 -*-

import argparse
import glob
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
from skimage import io, transform
from sklearn.model_selection import train_test_split


def preprocessing(args):
    # データ読み込み
    path = pathlib.Path(f"{args.input_path}")
    all_image_paths = [
        glob.glob(f"{args.input_path}/**/{item.parent.name}/{item.name}")[0]
        for item in path.glob("**/*")
        if item.is_file()
    ]

    # テストデータと学習データの分割
    train_image_paths, test_image_paths = train_test_split(all_image_paths)
    img_id = 0

    # 画像データの読み込み
    X_train = np.array(
        [
            transform.resize(io.imread(path, as_gray=True), (28, 28))
            for path in train_image_paths
        ]
    )
    X_train = np.reshape(X_train, (-1, 28, 28, 1))
    y_train = [pathlib.Path(path).parent.name for path in train_image_paths]

    X_test = np.array(
        [
            transform.resize(io.imread(path, as_gray=True), (28, 28))
            for path in test_image_paths
        ]
    )
    X_test = np.reshape(X_test, (-1, 28, 28, 1))
    y_test = [pathlib.Path(path).parent.name for path in test_image_paths]

    # ラベルの変換
    all_labels = list(set(y_train) & set(y_test))
    label_index = {label: idx for idx, label in enumerate(all_labels)}
    y_train = np.array([label_index[label] for label in y_train])
    y_test = np.array([label_index[label] for label in y_test])

    # pickle として保存
    with open(f"{args.output_path}/train_test_datas.pkl", "bw") as f:
        pickle.dump(((X_train, y_train), (X_test, y_test)), f)

    with open(f"{args.output_path}/labels_idx.pkl", "bw") as f:
        pickle.dump(label_index, f)


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="aqualium demo")
    parser.add_argument("--input_path", default="/kqi/input/images")
    parser.add_argument("--output_path", default="/kqi/output/preprocess")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    preprocessing(args)
