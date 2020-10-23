import os
import argparse
import numpy as np
import cv2
import pdf2image
from utils import draw_boxes
from pathlib import Path
from utils import *

if __name__ == '__main__':
    pdf_dir = "/home/mahad/abby_file/pdf"
    xml_dir = "/home/mahad/abby_file/xml"
    save_dir = "/tmp"
    pdf_files = os.listdir(pdf_dir)
    xml_files = os.listdir(xml_dir)
    for pdf_file in pdf_files:
        try:
            print(pdf_file)
            xml_path = os.path.join(xml_dir, Path(pdf_file).stem + ".xml")
            pdf_path = os.path.join(pdf_dir, pdf_file)
            xml_data = get_raw_data(xml_path)
            for page in xml_data:
                if page["page_number"] == 3:
                    print()
                para_boxes = page["para_boxes"]
                para_texts = page["para_texts"]
                is_para_heading = page["is_heading"]
                # para_boxes = [x for x in para_boxes_ if x != []]
                # para_texts = [x for x in para_texts_ if x != ""]
                assert len(para_boxes) == len(para_texts)
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
                # ################## MAHAD ################################ #
                # if page["page_number"] == 3:
                #     img_draw = draw_boxes(img, column_blocks)
                #     cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
                #     cv2.waitKey()
                # ######################################################### #
                column_blocks_merged = merge_blocks(column_blocks, all_boxes, all_texts)
                ordered_boxes = create_order(column_blocks_merged, all_boxes)
                ordered_texts = []
                for i in range(0, len(ordered_boxes)):
                    idx = all_boxes.index(ordered_boxes[i])
                    ordered_texts.append(all_texts[idx])
                if idx:
                    del idx
                for i in range(0, len(ordered_boxes)):
                    if not isinstance(ordered_texts[i], list):
                        img_draw = draw_boxes(img, [ordered_boxes[i]])
                        cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
                        cv2.waitKey()
                    else:
                        for row in ordered_texts[i]:
                            cells = row["boxes"]
                            if not cells:
                                continue
                            cells = np.array(row["boxes"])
                            cells = cells[np.argsort(cells[:, 0])]
                            for cell in cells:
                                img_draw = draw_boxes(img, [cell])
                                cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
                                cv2.waitKey()
        except OSError:
            print(pdf_file+" not found")
