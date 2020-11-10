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


def draw_lines(curr_page, coords, vertical=0, bb_color=(0, 0, 255)):
    """
    :param vertical: boolean; 1 for vertical line 0 for horizontal
    :param bb_color: color of bounding box
    :param curr_page: numpy array
    :param coords:
    :return: current page with bounding boxes drawn
    """
    # curr_page_image = curr_page[:, :, ::-1].copy()  # converting to opencv image
    curr_page_image = curr_page
    # page_num, bbox, page_width, page_height = bounding_box
    for j in range(0, len(coords)):
        if vertical:
            start = (coords[j], 0)
            end = (coords[j], curr_page_image.shape[1])
        else:
            start = (0, coords[j])
            end = (curr_page_image.shape[0], coords[j])
        curr_page_image = cv2.line(curr_page_image, start, end, bb_color, 5)
    return curr_page_image


def get_raw_data(xml_file):
    tree = etree.parse(xml_file)
    for elem in tree.getiterator():
        elem.tag = etree.QName(elem).localname
    root = tree.getroot()
    page_num = 1
    pages = []
    for page in root:  # iterate over pages
        # paragraph level features
        line_boxes = []
        line_texts = []
        para_boxes = []
        para_texts = []
        tables = []
        n_lines = []
        is_heading = []
        ###############################
        height = int(page.attrib.get("height"))
        width = int(page.attrib.get("width"))
        for block in page:  # iterate over blocks
            complete_table = {"bbox": [], "rows": []}
            if block.attrib.get("blockType") == "Text":
                for text in block:  # iterate over text blocks
                    for para in text:  # iterate over paragraphs
                        para_box = []
                        para_text = []
                        n_lines_para = 0
                        char_bold = []
                        char_italic = []
                        char_underline = []
                        for line in para:  # iterate over lines
                            char_text = []
                            n_lines_para += 1
                            for formatting in line:
                                for charParams in formatting:  # iterating over characters
                                    if charParams.text is not None:
                                        char_text.append(charParams.text)
                                    if "bold" in charParams.attrib.keys():
                                        char_bold.append(1)
                                    else:
                                        char_bold.append(0)
                                    if "italic" in charParams.attrib.keys():
                                        char_italic.append(1)
                                    else:
                                        char_italic.append(0)
                                    if "underline" in charParams.attrib.keys():
                                        char_underline.append(1)
                                    else:
                                        char_underline.append(0)
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
                        if para_text_str != "" and para_box != []:
                            para_texts.append(para_text_str)
                            para_boxes.append(para_box)
                            n_lines.append(n_lines_para)
                            if char_bold and sum(char_bold) >= 0.75 * len(char_bold) and n_lines_para <= 2:
                                is_heading.append(1)
                            else:
                                is_heading.append(0)
            elif block.attrib.get("blockType") == "Table":
                is_heading.append(0)
                table_rows = []
                for row in block:
                    row_cells = {"boxes": [], "texts": []}
                    for cell in row:
                        cell_boxes = []
                        cell_text = ""
                        for elem in cell.iter():
                            if elem.tag == "line":
                                cell_boxes.append([elem.attrib["l"], elem.attrib["t"], elem.attrib["r"],
                                                   elem.attrib["b"]])
                                for sub_elem in elem.iter():
                                    if sub_elem.tag == "charParams":
                                        cell_text += sub_elem.text
                        if cell_boxes == [] and cell_text == "":
                            continue
                        cell_boxes = np.array(cell_boxes).astype(np.float).astype(np.int)
                        if cell_boxes.ndim == 1:
                            cell_boxes = np.expand_dims(cell_boxes, axis=1)
                        row_cells["boxes"].append([np.amin(cell_boxes[:, 0]), np.amin(cell_boxes[:, 1]),
                                                   np.amax(cell_boxes[:, 2]), np.amax(cell_boxes[:, 3])])
                        row_cells["texts"].append(cell_text)
                    if row_cells["boxes"]:
                        table_rows.append(row_cells)
                        n_lines.append(len(table_rows))
                all_cells = []
                for table_row in table_rows:
                    all_cells.extend(table_row["boxes"])
                if all_cells:
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
            "is_heading": is_heading,
            "line_boxes": line_boxes,
            "line_texts": line_texts,
            "tables": tables,
            "lines": n_lines,
        }
        pages.append(page)
        page_num += 1
    return pages


def get_blocks(shape, boxes):
    """
    :param shape: tuple (height, width)
    :param boxes: list of boxes
    :return: list of the bounding boxes of all blocks
    """
    # # boxes = np.array(boxes)
    # width, height = shape
    # img = np.zeros(shape).astype(np.uint8)
    # boxes = [bb for bb in boxes if bb]
    # for box in boxes:
    #     # if box:
    #     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1)
    # # img = cv2.resize(img, fx=0.25, fy=0.25, dsize=None)  # downsizing the image to speed up the process
    # # img = cv2.copyMakeBorder(img, round(img.shape[0] * 0.25), round(img.shape[1] * 0.25), round(img.shape[0] * 0.25),
    # #                          round(img.shape[1] * 0.25), cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # kernel = np.ones((1, round(img.shape[0] * 0.5)), np.uint8)  # closing with 50 percent of the width
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # # cv2.imshow('', cv2.resize(img, fx=0.25, fy=0.25, dsize=None))
    # # cv2.waitKey()
    # ret, thresh = cv2.threshold(img, 127, 255, 0)
    # contours, img2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # blocks = []
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #     patch = img[y:y + h - 1, x:x + w - 1]
    #     m = np.mean(patch)
    #     if m > 64:  # remove black contours
    #         blocks.append([x, y, x + w, y + h])
    #     else:
    #         print("contour dropped in get_blocks")
    # to_del = []
    # for i, block in enumerate(blocks):
    #     for j, block2 in enumerate(blocks):
    #         if not i == j:
    #             if block[0] <= block2[0] and block[1] <= block2[1] and block[2] >= block2[2] and block[3] >= block2[3]:
    #                 to_del.append(j)
    # for index in sorted(to_del, reverse=True):
    #     del blocks[index]
    # # cv2.imshow('', cv2.resize(img, fx=0.25, fy=0.25, dsize=None))
    # # cv2.waitKey()
    blocks = []
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    tops, bottoms = get_separator(boxes, vertical=1)
    for top, bottom in zip(tops, bottoms):
        block_boxes = boxes[np.logical_and(boxes[:, 1] >= top, boxes[:, 3] <= bottom)]
        if block_boxes.shape[0] != 0:
            blocks.append([np.amin(block_boxes[:, 0]), top, np.amax(block_boxes[:, 2]), bottom])
    return blocks


def get_col_bounds(boxes, page_width, eps=150):
    boxes = np.array(boxes)
    boxes = boxes[np.argsort(boxes[:, 0])]
    col_bounds = []
    while len(boxes) > 0:
        xmin = np.amin(boxes[:, 0])
        idx = np.logical_and(boxes[:, 0] < xmin + 0.2 * page_width, boxes[:, 0] > xmin - 0.2 * page_width)
        col_boxes = boxes[idx]
        col_bounds.append([np.amin(col_boxes[:, 0]), np.amin(col_boxes[:, 1]), np.amax(col_boxes[:, 2]),
                           np.amax(col_boxes[:, 3])])
        boxes = np.delete(boxes, np.where(idx == True), axis=0)
    to_del = []
    col_bounds = np.array(col_bounds)
    for i in range(0, len(col_bounds)):
        for j in range(0, len(col_bounds)):
            if i == j:
                continue
            if col_bounds[i, 0] >= col_bounds[j, 0] - 10 and col_bounds[i, 2] <= col_bounds[j, 2] + 10:
                to_del.append(i)
    col_bounds = col_bounds.tolist()
    to_del = list(set(to_del))
    for index in sorted(to_del, reverse=True):
        del col_bounds[index]
    # seps1, seps2 = get_separator(boxes, vertical=0)
    # boxes = np.array(boxes)
    # col_bounds = [[ele1, np.amin(boxes[:, 1]), ele2, np.amax(boxes[:, 3])] for ele1, ele2 in zip(seps1, seps2)]
    return col_bounds


def get_separator(boxes, vertical=1):
    seps1 = []  # separator starts
    seps2 = []  # separator ends
    boxes = np.array(boxes)
    assert vertical == 0 or vertical == 1
    # boxes = boxes[np.argsort(boxes[:, 1])]
    if vertical == 1:
        for i, box in enumerate(boxes):
            crossing_boxes_start = boxes[np.logical_and(boxes[:, 1] < box[1], boxes[:, 3] > box[1])].tolist()
            crossing_boxes_end = boxes[np.logical_and(boxes[:, 1] < box[3], boxes[:, 3] > box[3])].tolist()
            if not crossing_boxes_start and box[1] not in seps1:
                seps1.append(box[1])
            if not crossing_boxes_end and box[3] not in seps2:
                seps2.append(box[3])
    else:
        for i, box in enumerate(boxes):
            crossing_boxes_start = boxes[np.logical_and(boxes[:, 0] < box[0], boxes[:, 2] > box[0])].tolist()
            crossing_boxes_end = boxes[np.logical_and(boxes[:, 0] < box[2], boxes[:, 2] > box[2])].tolist()
            if not crossing_boxes_start and box[0] not in seps1:
                seps1.append(box[0])
            if not crossing_boxes_end and box[2] not in seps2:
                seps2.append(box[2])
    seps2.sort()
    seps1.sort()
    return seps1, seps2


def get_block_para(block, paras, eps):
    block_boxes = paras[
        np.logical_and(np.logical_and(np.logical_and(paras[:, 0] >= block[0] - eps,
                                                     paras[:, 1] >= block[1] - eps),
                                      paras[:, 2] <= block[2] + eps),
                       paras[:, 3] <= block[3] + eps)]
    return block_boxes


def merge_blocks(blocks, para_boxes, is_heading):
    blocks = np.array(blocks)
    blocks = blocks[np.argsort(blocks[:, 1])]
    para_boxes = np.array(para_boxes)
    is_heading = np.array(is_heading)
    is_heading = is_heading[np.argsort(para_boxes[:, 1])]
    para_boxes = para_boxes[np.argsort(para_boxes[:, 1])]
    eps = 15
    n_cols = []
    page_height = np.amax(blocks[:, 3]) - np.amin(blocks[:, 1])
    page_width = np.amax(blocks[:, 2]) - np.amin(blocks[:, 0])
    for block in blocks:
        block_boxes = get_block_para(block, para_boxes, eps)
        n_cols.append(len(get_col_bounds(block_boxes, page_width)))

    working_bounds = []
    for j in range(0, len(n_cols) - 1):
        if n_cols[j] - n_cols[j + 1] == 1:
            curr_boxes = get_block_para(blocks[j], para_boxes, eps)
            next_boxes = get_block_para(blocks[j + 1], para_boxes, eps)
            next_boxes = next_boxes[np.argsort(next_boxes[:, 1])]
            idx = np.where((para_boxes == next_boxes[0, :]).all(axis=1))
            next_head = is_heading[idx[0][0]]
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
            if top_bottom - bottom_top >= 0.2 * page_height:
                working_bounds = []
                break

            und_cols = under_cols(working_bounds, next_col_bounds, page_height, page_width)
            if und_cols == n_cols[j + 1]:
                if next_head == 1:
                    next_next_block = blocks[j+2]
                    next_next_boxes = get_block_para(next_next_block.tolist(), para_boxes, eps=15)
                    next_next_cols = get_col_bounds(next_next_boxes, page_width)
                    und_cols2 = under_cols(working_bounds, next_next_cols, page_height, page_width)
                    ss = sum(is_heading[idx[0][0]+1:])
                    if und_cols2 == n_cols[j+2]:
                    # if ss > 0:
                        n_cols[j + 1] = n_cols[j]
                    else:
                        break
                else:
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
                merged.append(blocks[kkk].tolist())
        else:
            to_merge = blocks[rs[j]:rs[j] + rl[j]]
            merged.append([np.amin(to_merge[:, 0]), np.amin(to_merge[:, 1]), np.amax(to_merge[:, 2]),
                           np.amax(to_merge[:, 3])])

    if not merged:
        return blocks.tolist()
    else:
        return merged


def merge_blocks_2(blocks, boxes, eps=15):
    blocks = np.array(blocks)
    blocks = blocks[np.argsort(blocks[:, 1])]
    boxes = np.array(boxes)
    n_cols = []
    # page_height = np.amax(blocks[:, 3]) - np.amin(blocks[:, 1])
    page_width = np.amax(blocks[:, 2]) - np.amin(blocks[:, 0])
    col_bounds = []
    for block in blocks:
        block_boxes = get_block_para(block, boxes, eps)
        n_cols.append(len(get_col_bounds(block_boxes, page_width)))
        col_bounds.append(get_col_bounds(block_boxes, page_width))
    col_blocks = []
    for col_bound in col_bounds:
        col_bound = np.array(col_bound)
        bb = [np.amin(col_bound[:, 0]), np.amin(col_bound[:, 1]), np.amax(col_bound[:, 2]), np.amax(col_bound[:, 3])]
        col_blocks.append(bb)
    rv, rs, rl = find_runs(n_cols)  # run value, run start, run length
    cont_blocks = []
    for i in range(0, len(rv)):
        cont_blocks.append(col_bounds[rs[i]:rs[i] + rl[i]])
    comb_column = []
    for cont_block in cont_blocks:
        cont_block = np.squeeze(cont_block)
        if len(cont_block.shape) > 2:  # multi-column
            for i in range(0, cont_block.shape[1]):
                aa = [np.amin(cont_block[:, i, 0]), np.amin(cont_block[:, i, 1]), np.amax(cont_block[:, i, 2]),
                      np.amax(cont_block[:, i, 3])]
                comb_column.append(aa)
        else:
            if cont_block.ndim == 1:
                cont_block = np.expand_dims(cont_block, axis=0)
            for column in cont_block:
                comb_column.append(column.tolist())
    #  Removing columns inside columns
    to_del = []
    for i in range(0, len(comb_column)):
        top = comb_column[i]
        for j in range(0, len(comb_column)):
            if i == j:
                continue
            curr = comb_column[j]
            if curr[0] >= top[0] and curr[1] >= top[1] and curr[2] <= top[2] and curr[3] <= top[3]:
                to_del.append(j)
    for index in sorted(to_del, reverse=True):
        del comb_column[index]

    # #  columns below
    # for i in range(0, len(comb_column)):
    #     col = comb_column[i]
    #     cols_under = []
    #     for j in range(0, len(comb_column)):
    #         if i == j:
    #             continue
    #         if comb_column[j, 0] >= [0] and curr[1] >= top[1] and curr[2] <= top[2] and curr[3] <= top[3]:
    #             to_del.append(j)
    return comb_column


def merge_blocks_3(blocks, boxes, eps=15):
    blocks = np.array(blocks)
    blocks = blocks[np.argsort(blocks[:, 1])[::-1]]
    para_boxes = np.array(boxes)
    eps = 15
    n_cols = []
    page_height = np.amax(blocks[:, 3]) - np.amin(blocks[:, 1])
    page_width = np.amax(blocks[:, 2]) - np.amin(blocks[:, 0])
    for block in blocks:
        block_boxes = get_block_para(block, para_boxes, eps)
        n_cols.append(len(get_col_bounds(block_boxes, page_width)))

    working_bounds = []
    for j in range(0, len(n_cols) - 1):
        if n_cols[j] - n_cols[j + 1] == 1:
            curr_boxes = get_block_para(blocks[j], para_boxes, eps)
            next_boxes = get_block_para(blocks[j + 1], para_boxes, eps)
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
            if top_bottom - bottom_top >= 0.2 * page_height:
                working_bounds = []
                break

            und_cols = 0
            for k in range(0, len(working_bounds)):
                lb = np.amin(working_bounds[k][0])
                ub = np.amax(working_bounds[k][2])
                for l in range(0, len(next_col_bounds)):
                    if np.logical_and(next_col_bounds[l][0] >= lb - 0.07 * page_width,
                                      next_col_bounds[l][2] <= ub + 0.07 * page_height):

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
                merged.append(blocks[kkk].tolist())
        else:
            to_merge = blocks[rs[j]:rs[j] + rl[j]]
            merged.append([np.amin(to_merge[:, 0]), np.amin(to_merge[:, 1]), np.amax(to_merge[:, 2]),
                           np.amax(to_merge[:, 3])])

    if not merged:
        return blocks.tolist()[::-1]
    else:
        return merged[::-1]


def update_col_bounds(bounds1, bounds2):
    out = []
    bounds1 = np.array(bounds1)
    bounds2 = np.array(bounds2)
    for i in range(0, len(bounds1)):
        col = [min(bounds1[i, 0], bounds2[i, 0]), min(bounds1[i, 1], bounds2[i, 1]), max(bounds1[i, 2], bounds2[i, 2]),
               max(bounds1[i, 3], bounds2[i, 3])]
        out.append(col)
    return out


def under_cols(working_bounds, next_col_bounds, page_height, page_width):
    """
    :param working_bounds: list
    :param next_col_bounds: list
    :param page_height: int
    :param page_width: int
    :return: und_cols (number of underlying columns)
    """
    und_cols = 0
    for k in range(0, len(working_bounds)):
        lb = np.amin(working_bounds[k][0])
        ub = np.amax(working_bounds[k][2])
        for l in range(0, len(next_col_bounds)):
            if np.logical_and(next_col_bounds[l][0] >= lb - 0.07 * page_width,
                              next_col_bounds[l][2] <= ub + 0.07 * page_height):

                und_cols += 1
                if next_col_bounds[l][0] < lb:
                    lb = next_col_bounds[l][0]
                if next_col_bounds[l][2] > ub:
                    ub = next_col_bounds[l][2]
    return und_cols


def create_order(blocks, boxes):
    blocks = np.array(blocks)
    boxes = np.array(boxes)
    eps = 3
    boxes_out = []
    for block in blocks:
        block_boxes = get_block_para(block, boxes, eps)
        block_width = block[2] - block[0]
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


def overlap(min1, max1, min2, max2):
    return max(0, min(max1, max2) - max(min1, min2))


def create_order2(blocks, boxes, img):
    blocks = np.array(blocks)
    boxes = np.array(boxes)
    eps = 3
    boxes_out = []
    blocks = blocks[np.argsort(blocks[:, 1])]
    for block in blocks:
        block_boxes = get_block_para(block, boxes, eps)
        block_boxes = block_boxes[np.argsort(block_boxes[:, 1])]
        seps_l, seps_r = get_separator(block_boxes, vertical=0)
        # img_draw = draw_boxes(img, [block], (0, 255, 0))
        for left, right in zip(seps_l, seps_r):
            col_block = [left, block[1], right, block[3]]
            # img_draw = draw_boxes(img, [col_block])
            # cv2.imshow('', cv2.resize(img_draw, fx=0.25, fy=0.25, dsize=None))
            # cv2.waitKey()
            col_boxes = get_block_para(col_block, block_boxes, eps=3)
            col_boxes = col_boxes[np.argsort(col_boxes[:, 1])]
            boxes_out.extend(col_boxes.tolist())
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
