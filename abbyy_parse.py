import os
import argparse
import numpy as np
import cv2
import pdf2image
from utils import draw_boxes
from pathlib import Path
from utils import get_raw_data, merge_blocks, create_order, get_blocks, remove_empty

if __name__ == '__main__':
    pdf_dir = "/home/mahad/abbyy_dummy_dataset/pdf"
    xml_dir = "/home/mahad/abbyy_dummy_dataset/xml"
    save_dir = "/tmp"
    pdf_files = os.listdir(pdf_dir)
    xml_files = os.listdir(xml_dir)
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        pdf_path = os.path.join(pdf_dir, Path(xml_file).stem + ".pdf")
        xml_data = get_raw_data(xml_path)
        for page in xml_data:
            para_boxes = page["para_boxes"]
            para_texts = page["para_texts"]
            para_boxes, para_texts = remove_empty(para_boxes, para_texts)
            tables = page["tables"]
            table_boxes = [tt["bbox"] for tt in tables]
            table_texts = [tt["rows"] for tt in tables]
            img = pdf2image.convert_from_path(pdf_path, size=(page["width"], page["height"]),
                                              first_page=page["page_number"], last_page=page["page_number"])
            img = np.asarray(img[0])
            all_boxes = para_boxes + table_boxes
            all_texts = para_texts + table_texts
            column_blocks = get_blocks((page["height"], page["width"]), all_boxes)
            column_blocks_merged = merge_blocks(column_blocks, all_boxes)
            ordered_boxes = create_order(column_blocks_merged, all_boxes)
            ordered_texts = []
            for i in range(0, len(ordered_boxes)):
                idx = all_boxes.index(ordered_boxes[i])
                ordered_texts.append(all_texts[idx])
            if idx:
                del idx
            for ordered_box in ordered_boxes:
                img_draw = draw_boxes(img, [ordered_box])
                cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
                # cv2.imshow('', img_draw)
                cv2.waitKey()
