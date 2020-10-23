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
            cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
            cv2.waitKey()
            print(df)
            print()

