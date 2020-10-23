import os
import argparse

import fitz as fitz
from lxml import etree
import numpy as np
import cv2
import pdf2image
import pandas as pd
from utils import draw_boxes
from pathlib import Path
from utils import *
<<<<<<< HEAD

if __name__ == '__main__':
    img_dir = "/home/mahad/tmp/01-protected-retirement-plan-customer-key-features/png"
    npy_dir = "/home/mahad/tmp/01-protected-retirement-plan-customer-key-features/npy"
    save_dir = "/tmp"
    img_files = os.listdir(img_dir)
    npy_files = os.listdir(npy_dir)
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        npy_path = os.path.join(npy_dir, Path(img_file).stem + ".npy")
        img = cv2.imread(img_path)
        all_boxes = np.load(npy_path, allow_pickle=True).tolist()
        print()
        # all_boxes = para_boxes + table_boxes
        # all_texts = para_texts + table_texts
        column_blocks = get_blocks((img.shape[0], img.shape[1]), all_boxes)
        column_blocks_merged = merge_blocks(column_blocks, all_boxes)
        column_blocks_merged_3 = merge_blocks_3(column_blocks_merged, all_boxes)
        ordered_boxes = create_order(column_blocks_merged_3, all_boxes)
        for ordered_box in ordered_boxes:
            img_draw = draw_boxes(img, [ordered_box])
=======
import glob
from fitz import frontend

if __name__ == '__main__':
    img_files = "test/images/*.png"
    pdf_file = "test/pdf/Ascentric_Pension_Account_SIPP_key_features_document.pdf"
    df_dir = "test/dfs"
    if __name__ == '__main__':
        imgs = glob.glob(img_files)
        for i, img_file in enumerate(imgs):
            img = cv2.imread(img_file)
            # get pdf width and height
            doc = fitz.open(pdf_file)
            page = doc.loadPage(int(1) - 1)
            _, _, pdf_width, pdf_height = page.MediaBox
            df_name = Path(img_file).stem + ".csv"
            df = pd.read_csv(os.path.join(df_dir, df_name))
            xmin = df["x0"].to_numpy()
            xmax = df["x1"].to_numpy()
            ymin = df["y0"].to_numpy()
            ymax = df["y1"].to_numpy()
            boxes = np.vstack((xmin, ymin, xmax, ymax)).transpose()
            img_draw = draw_boxes(img, boxes.astype(np.int).tolist())
>>>>>>> tmp
            cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
            cv2.waitKey()
            print(df)
            print()

