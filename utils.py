import cv2
import numpy as np
from lxml import etree


def draw_boxes(curr_page, bounding_box, bb_color=(0, 0, 255)):
    """
    :param bb_color: color of bounding box
    :param curr_page: numpy array
    :param bounding_box:
    :return: current page with bounding boxes drawn
    """
    # curr_page_image = curr_page[:, :, ::-1].copy()  # converting to opencv image
    curr_page_image = curr_page
    # page_num, bbox, page_width, page_height = bounding_box
    for j in range(0, len(bounding_box)):
        bbox = bounding_box[j]
        top_left = (bbox[0], bbox[1])
        # bb_color = (255, 0, 0)
        bottom_right = (bbox[2], bbox[3])
        # bb_color = (0, 0, 255)
        cv2.rectangle(curr_page_image, top_left, bottom_right, bb_color, 3)
    return curr_page_image


def get_raw_data(xml_file):
    tree = etree.parse(xml_file)
    root = tree.getroot()
    page_num = 1
    pages = []
    for page in root:  # iterate over pages
        line_boxes = []
        line_texts = []
        para_boxes = []
        para_texts = []
        tables = []
        height = int(page.attrib.get("height"))
        width = int(page.attrib.get("width"))
        for block in page:  # iterate over blocks
            complete_table = {"bbox": [], "rows": []}
            if block.attrib.get("blockType") == "Text":
                for text in block:  # iterate over text blocks
                    for para in text:  # iterate over paragraphs
                        para_box = []
                        para_text = []
                        for line in para:  # iterate over lines
                            char_text = []
                            for formatting in line:
                                for charParams in formatting:  # iterating over characters
                                    if charParams.text is not None:
                                        char_text.append(charParams.text)
                            baseline = int(line.attrib.get("baseline"))
                            xmin = int(line.attrib.get("l"))
                            ymin = int(line.attrib.get("t"))
                            xmax = int(line.attrib.get("r"))
                            ymax = int(line.attrib.get("b"))
                            box = [xmin, ymin, xmax, ymax]
                            line_boxes.append(box)
                            if para_box:
                                if box[0] < para_box[0]:
                                    para_box[0] = box[0]
                                if box[1] < para_box[1]:
                                    para_box[1] = box[1]
                                if box[2] > para_box[2]:
                                    para_box[2] = box[2]
                                if box[3] > para_box[3]:
                                    para_box[3] = box[3]
                            else:
                                para_box.extend(box)
                            line_text = ["".join(char_text)]
                            line_texts.extend(line_text)
                            para_text.append(line_text[0])
                        para_text_str = " ".join(para_text)
                        para_texts.append(para_text_str)
                        para_boxes.append(para_box)
            elif block.attrib.get("blockType") == "Table":
                table_rows = []
                for row in block:
                    row_cells = {"boxes": [], "texts": []}
                    for cell in row:
                        cell_boxes = []
                        cell_text = ""
                        for elem in cell.iter():
                            if elem.tag.count("line") > 0:
                                cell_boxes.append([elem.attrib["l"], elem.attrib["t"], elem.attrib["r"],
                                                   elem.attrib["b"]])
                                for sub_elem in elem.iter():
                                    if sub_elem.tag.count("charParams") > 0:
                                        cell_text += sub_elem.text
                        if cell_boxes == [] and cell_text == "":
                            continue
                        cell_boxes = np.array(cell_boxes).astype(np.float).astype(np.int)
                        if cell_boxes.ndim == 1:
                            cell_boxes = np.expand_dims(cell_boxes, axis=1)
                        row_cells["boxes"].append([np.amin(cell_boxes[:, 0]), np.amin(cell_boxes[:, 1]),
                                                   np.amax(cell_boxes[:, 2]), np.amax(cell_boxes[:, 3])])
                        row_cells["texts"].append(cell_text)
                        # for text in cell:
                        #     for para in text:
                        #         for line in para:
                        #             char_text = []
                        #             for formatting in line:
                        #                 for charParams in formatting:
                        #                     if charParams.text is not None:
                        #                         char_text.append(charParams.text)
                        #                 baseline = int(line.attrib.get("baseline"))
                        #                 xmin = int(line.attrib.get("l"))  # - 35
                        #                 ymin = int(line.attrib.get("t"))  # - 35
                        #                 xmax = int(line.attrib.get("r"))  # + 35
                        #                 ymax = int(line.attrib.get("b"))  # + 35
                        #                 cell = [xmin, ymin, xmax, ymax]
                        #                 row_cells["boxes"].append(cell)
                        #                 row_cells["texts"].append("".join(char_text))
                    table_rows.append(row_cells)
                print()
                all_cells = []
                for table_row in table_rows:
                    all_cells.extend(table_row["boxes"])
                all_cells = np.array(all_cells)
                complete_table["bbox"] = [np.amin(all_cells[:, 0]), np.amin(all_cells[:, 1]), np.amax(all_cells[:, 2]),
                                          np.amax(all_cells[:, 3])]
                complete_table["rows"].extend(table_rows)
            if not complete_table["rows"] == []:
                tables.append(complete_table)
        page = {
            "page_number": page_num,
            "width": width,
            "height": height,
            "para_boxes": para_boxes,
            "para_texts": para_texts,
            "line_boxes": line_boxes,
            "line_texts": line_texts,
            "tables": tables
        }
        pages.append(page)
        page_num += 1

    # result = {
    #     "width": width,
    #     "height": height,
    #     "para_boxes": para_boxes,
    #     "para_texts": para_texts,
    #     "line_boxes": line_boxes,
    #     "line_texts": line_texts,
    #     "tables": tables
    # }
    return pages


def get_blocks(shape, boxes):
    """
    :param shape: tuple (height, width)
    :param boxes: list of boxes
    :return: list of the bounding boxes of all blocks
    """
    # boxes = np.array(boxes)
    width, height = shape
    img = np.zeros(shape).astype(np.uint8)
    boxes = [bb for bb in boxes if bb]
    for box in boxes:
        # if box:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
    # img = cv2.resize(img, fx=0.25, fy=0.25, dsize=None)  # downsizing the image to speed up the process

    kernel = np.ones((1, round(img.shape[0] * 0.6)), np.uint8)  # closing with 60 percent of the width
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, img2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        patch = img[y:y + h - 1, x:x + w - 1]
        m = np.mean(patch)
        if m > 128:  # remove black contours
            blocks.append([x, y, x + w, y + h])
    cv2.imwrite("/tmp/blocks.png", img)
    return blocks


def get_col_bounds(boxes, page_width, eps=150):
    # boxes = np.array(boxes)
    boxes = boxes[np.argsort(boxes[:, 0])]
    col_bounds = []
    while len(boxes) > 0:
        xmin = np.amin(boxes[:, 0])
        idx = np.logical_and(boxes[:, 0] < xmin + 0.2*page_width, boxes[:, 0] > xmin - 0.2*page_width)
        col_boxes = boxes[idx]
        col_bounds.append([np.amin(col_boxes[:, 0]), np.amin(col_boxes[:, 1]), np.amax(col_boxes[:, 2]),
                           np.amax(col_boxes[:, 3])])
        boxes = np.delete(boxes, np.where(idx == True), axis=0)
    return col_bounds


def merge_blocks(blocks, para_boxes):
    blocks = np.array(blocks)
    blocks = blocks[np.argsort(blocks[:, 1])]
    para_boxes = np.array(para_boxes)
    eps = 15
    n_cols = []
    page_height = np.amax(blocks[:, 3]) - np.amin(blocks[:, 1])
    page_width = np.amax(blocks[:, 2]) - np.amin(blocks[:, 0])
    for block in blocks:
        block_boxes = para_boxes[
            np.logical_and(np.logical_and(np.logical_and(para_boxes[:, 0] >= block[0] - eps,
                                                         para_boxes[:, 1] >= block[1] - eps),
                                          para_boxes[:, 2] <= block[2] + eps),
                           para_boxes[:, 3] <= block[3] + eps)]
        n_cols.append(len(get_col_bounds(block_boxes, page_width)))

    working_bounds = []
    for j in range(0, len(n_cols) - 1):
        if n_cols[j] - n_cols[j + 1] == 1:
            curr_boxes = para_boxes[
                np.logical_and(np.logical_and(np.logical_and(para_boxes[:, 0] >= blocks[j, 0] - eps,
                                                             para_boxes[:, 1] >= blocks[j, 1] - eps),
                                              para_boxes[:, 2] <= blocks[j, 2] + eps),
                               para_boxes[:, 3] <= blocks[j, 3] + eps)]
            next_boxes = para_boxes[
                np.logical_and(np.logical_and(np.logical_and(para_boxes[:, 0] >= blocks[j + 1, 0] - eps,
                                                             para_boxes[:, 1] >= blocks[j + 1, 1] - eps),
                                              para_boxes[:, 2] <= blocks[j + 1, 2] + eps),
                               para_boxes[:, 3] <= blocks[j + 1, 3] + eps)]
            curr_col_bounds = np.array(get_col_bounds(curr_boxes, page_width))
            if not working_bounds:
                working_bounds = curr_col_bounds.tolist()
            else:
                curr_col_bounds = np.vstack((curr_col_bounds, working_bounds))
                working_bounds = get_col_bounds(curr_col_bounds, page_width)
            next_col_bounds = get_col_bounds(next_boxes, page_width)
            if not any(isinstance(el, list) for el in next_col_bounds):
                next_col_bounds = [next_col_bounds]
            if not any(isinstance(el, list) for el in working_bounds):
                working_bounds = [working_bounds]

            bottom_top = np.amax(np.array(working_bounds)[:, 3])
            top_bottom = np.amin(np.array(next_col_bounds)[:, 1])
            if top_bottom - bottom_top >= 0.2*page_height:
                working_bounds = []
                break

            und_cols = 0
            for k in range(0, len(working_bounds)):
                lb = np.amin(working_bounds[k][0])
                ub = np.amax(working_bounds[k][2])
                for l in range(0, len(next_col_bounds)):
                    if np.logical_and(next_col_bounds[l][0] >= lb - 0.07*page_width,
                                      next_col_bounds[l][2] <= ub + 0.07*page_height):

                        und_cols += 1
                        if next_col_bounds[l][0] < lb:
                            lb = next_col_bounds[l][0]
                        if next_col_bounds[l][2] > ub:
                            ub = next_col_bounds[l][2]
            if und_cols == n_cols[j + 1]:
                n_cols[j + 1] = n_cols[j]
            else:
                working_bounds = []
        else:
            working_bounds = []
    n, rs, rl = find_runs(n_cols)
    merged = []
    for j in range(0, len(n)):
        if n[j] == 1:
            for kkk in range(rs[j], rs[j] + rl[j]):
                merged.append(blocks[kkk])
        else:
            to_merge = blocks[rs[j]:rs[j] + rl[j]]
            merged.append([np.amin(to_merge[:, 0]), np.amin(to_merge[:, 1]), np.amax(to_merge[:, 2]),
                           np.amax(to_merge[:, 3])])
    if not merged:
        return blocks.tolist()
    else:
        return merged


def update_col_bounds(bounds1, bounds2):
    out = []
    bounds1 = np.array(bounds1)
    bounds2 = np.array(bounds2)
    for i in range(0, len(bounds1)):
        col = [min(bounds1[i, 0], bounds2[i, 0]), min(bounds1[i, 1], bounds2[i, 1]), max(bounds1[i, 2], bounds2[i, 2]),
               max(bounds1[i, 3], bounds2[i, 3])]
        out.append(col)
    return out


def lies_under(bounds1, bounds2, eps=85):
    bounds1 = np.array(bounds1)
    bounds2 = np.array(bounds2)
    out_cols = []
    for i in range(0, len(bounds1)):
        if (bounds2[0] >= bounds1[i, 0] - eps) and (bounds2[2] <= bounds1[i, 2] + eps):
            return i
    return -1


def create_order(blocks, boxes):
    blocks = np.array(blocks)
    boxes = np.array(boxes)
    eps = 3
    boxes_out = []
    for block in blocks:
        block_boxes = boxes[
            np.logical_and(np.logical_and(np.logical_and(boxes[:, 0] >= block[0] - eps, boxes[:, 1] >= block[1] - eps),
                                          boxes[:, 2] <= block[2] + eps), boxes[:, 3] <= block[3] + eps)]
        # block_boxes = block_boxes[np.argsort(block_boxes[:, 1])]
        block_width = block[2] - block[0]
        block_height = block[3] - block[1]
        while len(block_boxes) > 0:
            idx = np.logical_and(np.amin(block_boxes[:, 0]) - 0.1 * block_width <= block_boxes[:, 0],
                                 block_boxes[:, 0] <= np.amin(block_boxes[:, 0] + 0.1 * block_width))
            # block_boxes = block_boxes[np.argsort(block_boxes[:, 0])]
            col_boxes = block_boxes[idx]
            col_boxes = col_boxes[np.argsort(col_boxes[:, 1])]
            boxes_out.extend(col_boxes.tolist())
            block_boxes = np.delete(block_boxes, idx, axis=0)
    if not boxes_out:
        return boxes.tolist()
    else:
        return boxes_out


def find_runs(x):
    """Find runs of consecutive items in an array."""
    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]
    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]
        # find run values
        run_values = x[loc_run_start]
        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))
        return run_values, run_starts, run_lengths


def parse_HTML(html_file):
    parser = etree.HTMLParser()
    tree = etree.parse(html_file, parser)
    all_elements = list(tree.iter())
    return all_elements


def remove_empty(para_boxes, para_texts):
    to_remove = []
    for i in range(0, len(para_boxes)):
        if para_texts[i] == '':
            to_remove.append(i)
    for ele in sorted(to_remove, reverse=True):
        del para_texts[ele]
        del para_boxes[ele]
    return para_boxes, para_texts
