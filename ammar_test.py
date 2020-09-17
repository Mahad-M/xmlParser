import os
import argparse
from lxml import etree
import numpy as np
import cv2
import pdf2image
from utils import draw_boxes
from pathlib import Path
from utils import find_runs, get_raw_data, merge_blocks, create_order, get_blocks

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
        all_boxes = np.load(npy_path, allow_pickle=True)
        print()
        # all_boxes = para_boxes + table_boxes
        # all_texts = para_texts + table_texts
        column_blocks = get_blocks((img.shape[0], img.shape[1]), all_boxes)
        column_blocks_merged = merge_blocks(column_blocks, all_boxes)
        ordered_boxes = create_order(column_blocks_merged, all_boxes)
        # ordered_texts = []
        # for i in range(0, len(ordered_boxes)):
        #     idx = all_boxes.index(ordered_boxes[i])
        #     ordered_texts.append(all_texts[idx])
        # if idx:
        #     del idx
        for ordered_box in ordered_boxes:
            img_draw = draw_boxes(img, [ordered_box])
            cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
            # cv2.imshow('', img_draw)
            cv2.waitKey()
        # # cv2.imwrite("/tmp/" + xml_file.replace("xml", "png"), img_draw)
