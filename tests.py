from lxml import etree


def retrieve_style_text(xml_file, style):
    xml_tree = etree.parse(xml_file)
    text = []
    locs = []
    for elem in xml_tree.iter():
        if elem.tag.count('charParams') > 0 and elem.attrib["style"] == style:
            text.append(elem.text)
            locs.append([elem.attrib["l"], elem.attrib["t"], elem.attrib["r"], elem.attrib["b"]])
    return text, locs


if __name__ == '__main__':
    xml_file = "/home/mahad/abbyy_dummy_dataset/merged_xml/font_styles.xml"
    texts, locs = retrieve_style_text(xml_file, "bold,italic")
    print()
