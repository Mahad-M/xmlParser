import os
import argparse
from lxml import etree
import numpy as np
import cv2
import pdf2image
import pandas as pd
from utils import draw_boxes
from pathlib import Path
from utils import *
import glob

if __name__ == '__main__':
    img_files = "test/images/pension-annuity-key-features-document/*.png"
    # pdf_file = "test/pdf/Ascentric_Pension_Account_SIPP_key_features_document.pdf"
    df_dir = "test/dfs"
    df_p_dir = "test/dfs_2/pension-annuity-key-features-document"
    if __name__ == '__main__':
        imgs = glob.glob(img_files)
        imgs.sort()
        for i, img_file in enumerate(imgs):
            # if i == 2:
                para_boxes = []
                is_bold = []
                img = cv2.imread(img_file)
                df_p_file = Path(img_file).stem + ".csv"
                df_p = pd.read_csv(os.path.join(df_p_dir, df_p_file))
                binValues = np.unique(df_p["bin"].to_numpy())
                for binValue in binValues:
                    _bin = df_p["bin"] == binValue
                    bin_df = df_p[_bin]
                    subBinValues = np.unique(bin_df["sub-bin"].to_numpy())
                    for subBinValue in subBinValues:
                        sub_bin = bin_df["sub-bin"] == subBinValue
                        sub_bin_df = bin_df[sub_bin]
                        pValues = np.unique(sub_bin_df["pTag"].to_numpy())
                        for pValue in pValues:
                            para = sub_bin_df["pTag"] == pValue
                            para_df = sub_bin_df[para]
                            n_bold = sum(para_df["bold"].tolist())
                            if n_bold >= 0.75*para_df.shape[0]:
                                is_bold.append(1)
                            else:
                                is_bold.append(0)
                            xmin_para = round(para_df["x0"].min()*300/72)
                            xmax_para = round(para_df["x1"].max()*300/72)
                            ymax_para = round(img.shape[0] - para_df["y0"].min()*300/72)
                            ymin_para = round(img.shape[0] - para_df["y1"].max()*300/72)
                            para_boxes.append([xmin_para, ymin_para, xmax_para, ymax_para])
                column_blocks = get_blocks((img.shape[0], img.shape[1]), para_boxes)
                column_blocks_merged = merge_blocks(column_blocks, para_boxes, is_bold)
                ordered_boxes = create_order2(column_blocks_merged, para_boxes, img)
                for ordered_box in ordered_boxes:
                    img_draw = draw_boxes(img, [ordered_box])
                    cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
                    cv2.waitKey()

