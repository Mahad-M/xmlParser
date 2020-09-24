import os
import numpy as np
import json
from lxml import etree
from pathlib import Path
import re
if __name__ == '__main__':
    html_file = "/home/mahad/abbyy_dummy_dataset/html/Table.html"
    xml_file = "/home/mahad/abbyy_dummy_dataset/xml/Table.xml"
    css_file = "/home/mahad/abbyy_dummy_dataset/html/Table.css"
    save_dir = "/home/mahad/abbyy_dummy_dataset/merged_xml"

    parser = etree.HTMLParser()
    html_tree = etree.parse(html_file, parser)
    html_str = ""
    char_styles = []
    style_len = []
    for elem in html_tree.iter():
        if elem.tag == 'span':
            elem_style = [elem.attrib]
            html_str += elem.text
            style_len.append(len(elem.text))
            # if "style" in elem.attrib:
            #     # elem.set("new_attr", "testing")
            #     elem_style.append(elem.attrib["style"])
            char_styles.extend(elem_style)
            # print(elem.attrib["style"])

    n_chars = 0
    run_ends = np.cumsum(style_len)
    xml_tree = etree.parse(xml_file)
    font_styles_cum = []
    font_family_cum = []
    # for elem in xml_tree.iter():
    #     if elem.tag == "par":
    #         line_count = 0
    #         for line in elem:
    #             if line_count >= len(elem.getchildren()):
    #                 for formatting in elem:
    #                     pos = len(formatting.getchildren())
    #                     c = etree.Element("charParams")
    #                     c.text = " "
    #                     formatting.insert(pos, c)
    #             line_count += 1
    # xml_tree.write(os.path.join(save_dir, Path(xml_file).name), pretty_print=True)
    xml_str = ""
    for elem in xml_tree.iter():
        if elem.tag.count("charParams") > 0 and "charConfidence" in elem.attrib.keys():
            xml_str += elem.text
    print(len(html_str))
    print(len(xml_str))
    with open("/tmp/xml.txt", "wt") as file:
        file.write(xml_str)
        file.close()
    with open("/tmp/html.txt", "wt") as file:
        file.write(html_str)
        file.close()
    for i in range(0, len(char_styles)):
        font_family = char_styles[i]["class"]
        n_chars = style_len[i]
        styles = []
        if "style" in char_styles[i]:
            style = char_styles[i]["style"]
            style_attribs = style.split(";")
            while "" in style_attribs:
                style_attribs.remove("")
            for key_value in style_attribs:
                styles.append(key_value.split(":")[1])
            font_style = ",".join(styles)
        else:
            font_style = ""
        font_styles_cum.extend([font_style]*n_chars)
        font_family_cum.extend([font_family]*n_chars)
    j = 0
    for elem in xml_tree.iter():
        if elem.tag.count('charParams') > 0:
            elem.set("font", font_family_cum[j])
            elem.set("style", font_styles_cum[j])
            j += 1
    save_name = os.path.join(save_dir, Path(xml_file).name)
    xml_tree.write(os.path.join(save_dir, Path(xml_file).name), pretty_print=True)


