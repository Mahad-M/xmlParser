import os
import numpy as np
import json
from lxml import etree
from pathlib import Path
import re
from utils2 import *
from lxml.etree import QName


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end], start
    except ValueError:
        return "", 0


if __name__ == '__main__':
    xml_file2 = "/home/mahad/docs_to_parse/xml/AR.xml"  # formatting xml
    xml_file1 = "/home/mahad/docs_to_parse/xml/AR_2.xml"  # plain text xml
    save_dir = "/home/mahad/docs_to_parse/xml_2"

    xml2_tree = etree.parse(xml_file2)
    for elem in xml2_tree.getiterator():
        elem.tag = etree.QName(elem).localname
    xml2_str = ""
    xml2_boxes = []
    total_chars = 0
    for page in xml2_tree.iter():
        if page.tag.count("page") > 0:
            page_arr = []
            page2_line_boxes = []
            for line in page.iter():
                if line.tag.count("line") > 0:
                    line_format = []
                    for formatting in line.iter():
                        if formatting.tag.count("formatting") > 0:
                            if formatting.text:
                                xml2_str += formatting.text
                                att = formatting.attrib
                                n_chars = len(formatting.text)
                                line_format.append((n_chars, att))
                    line_box = [int(line.attrib["l"]), int(line.attrib["t"]), int(line.attrib["r"]),
                                int(line.attrib["b"])]
                    page_arr.append([line_box, line_format])
            xml2_boxes.append(page_arr)

    xml1_tree = etree.parse(xml_file1)
    for elem in xml1_tree.getiterator():
        elem.tag = etree.QName(elem).localname

    with open(xml_file1, "r") as f:
        xml1_str = f.read()

    xml2_lines = []
    xml2_formatting = []
    for i, page in enumerate(xml2_boxes):
        xml2_lines.append(list(list(zip(*page))[0]))
        xml2_formatting.append(list(list(zip(*map(reversed, page)))[0]))

    page_num = -1
    line_num = 0
    for page in xml1_tree.iter():
        if page.tag == "page":
            page_num += 1
            line_num = 0
            for line in page.iter():
                if line.tag == "line":
                    format_to = xml2_formatting[page_num][line_num]
                    line_num += 1
                    format_to_flat = [item for item in format_to]
                    for formatting in line.getchildren():
                        charParams = iter(formatting.getchildren())
                        char_num = 0
                        for form in format_to:
                            while char_num < form[0]:
                                char_elem = next(charParams)
                                for ff in form[1]:
                                    f = form[1][ff]
                                    char_elem.set(ff, f)
                                char_num += 1
                            char_num = 0

    # with open("/home/mahad/Desktop/test.xml", "w") as f:
    xml1_tree.write(os.path.join(save_dir, Path(xml_file2).name))
