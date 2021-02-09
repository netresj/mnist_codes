# -*- coding:utf-8 -*-

import argparse
import datetime
import glob
import os
import pathlib
import zipfile
import pandas as pd

from tqdm import tqdm
import numpy as np
from skimage import io, transform

from sklearn.model_selection import train_test_split


def preprocessing(args):
    # データ読み込み
    path = pathlib.Path(f"{args.input_path}")
    all_labels = [item.name for item in path.glob("*") if item.is_dir()]
    label_index = {label: idx for idx, label in enumerate(all_labels)}
    all_image_paths = [
        glob.glob(f"{args.input_path}/**/{item.parent.name}/{item.name}")[0]
        for item in path.glob("**/*")
        if item.is_file()
    ]

    # テストデータと学習データの分割
    test_image_paths, train_image_paths = train_test_split(all_image_paths)
    img_id = 0

    label_filename_lst = []
    for path in tqdm(train_image_paths):
        img = io.imread(path, as_gray=True)
        img = transform.resize(img, (28, 28))
        io.imsave(f"{args.output_path}/{img_id}.png", img * 255)
        label = pathlib.Path(path).parent.name
        label_filename_lst.append([f"{img_id}.png", label, label_index[label]])
        img_id += 1
    df = pd.DataFrame(label_filename_lst, columns=["file_name", "label", "label_id"])
    df.to_csv(f"{args.output_path}/train.csv", index=False)

    label_filename_lst = []
    for path in tqdm(test_image_paths):
        img = io.imread(path, as_gray=True)
        img = transform.resize(img, (28, 28))
        io.imsave(f"{args.output_path}/{img_id}.png", img * 255)
        label = pathlib.Path(path).parent.name
        label_filename_lst.append([f"{img_id}.png", label, label_index[label]])
        img_id += 1
    df = pd.DataFrame(label_filename_lst, columns=["file_name", "label", "label_id"])
    df.to_csv(f"{args.output_path}/test.csv", index=False)


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="aqualium demo")
    parser.add_argument("--input_path", default="/kqi/input/images")
    parser.add_argument("--output_path", default="/kqi/output/preprocess")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    preprocessing(args)
