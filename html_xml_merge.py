from difflib import SequenceMatcher
from bs4 import BeautifulSoup
import os
import numpy as np
import glob
from lxml import html, etree


def html_xml_merge(html_file, xml_file):

    html_par_sentences = {}
    list_of_html_par_sentences = []
    list_of_lists_of_html_par_sentences = []

    with open(html_file, 'r') as f:
        page = f.read()
    root = html.fromstring(page)

    for tag in root.iter('p'):
        if len(tag.getchildren()) >= 2:  # A tag has multiple <spans>
            for child in tag.findChildren():
                html_par_sentences['text'] = child.text
                html_par_sentences['meta'] = [child.get('style'), child.get('class')]
                list_of_html_par_sentences.append(html_par_sentences)
                html_par_sentences = {}

        elif len(tag.getchildren()) == 1:  # A tag has single <span>
            child = tag.getchildren()
            html_par_sentences['text'] = child[0].text

            if child[0].get('style'):  # If there is font style within span, fetch it else get from the tag 'p'
                font_style = child[0].get('style')
                html_par_sentences['meta'] = [font_style, child[0].get('class')]
            else:
                html_par_sentences['meta'] = [tag.get('style'), child[0].get('class')]

            list_of_html_par_sentences.append(html_par_sentences)
            html_par_sentences = {}

        list_of_lists_of_html_par_sentences.append(list_of_html_par_sentences)
        list_of_html_par_sentences = []
    words = []
    xml_par_scentences = []
    xml_dict = {}
    xml_par_meta = []
    with open(xml_file, 'r', encoding='utf8') as f:
        page = f.read()
    root = etree.fromstring(page)
    c_co = 0
    xml_par_meta_prev = []
    for elem in root.iter():
        if elem.tag.count('par') > 0:
            # children = elem.findchildren()
            for child in elem.iter():
                if child.tag.count('charParams') > 0:
                    words.append(child.text)
                    xml_par_meta.append([int(child.attrib["l"]), int(child.attrib["t"]), int(child.attrib["r"]),
                                         int(child.attrib["b"])])
                elif child.tag.count('line') > 0:
                    words.append(' ')
                    xml_par_meta.append([int(child.attrib["l"]), int(child.attrib["t"]), int(child.attrib["r"]),
                                         int(child.attrib["b"])])

            xml_dict[''.join(words)] = xml_par_meta
            xml_par_scentences.append(''.join(words))

        if elem.tag.count('block') > 0:  # If a new block occurs and it has a par
            block = 1

        elif elem.tag.count('par') > 0:
            if block == 0:
                prev_block_par_meta = xml_par_meta
                words = []
                xml_par_meta = []

            elif block == 1:
                xml_par_from_next_block = ''.join(words)
                print('xml_par_from_next_block: ', xml_par_from_next_block[1])
                frst_ltr_of_nxt_blk_para = xml_par_from_next_block[1]
                if frst_ltr_of_nxt_blk_para.islower():
                    try:
                        xml_par_from_prev_block = xml_par_scentences[-2]
                        del xml_par_scentences[-1]
                        del xml_par_scentences[-1]

                        # Remove entries from the meta dict
                        prev_block_par_meta = prev_block_par_meta + xml_par_meta
                        del xml_dict[xml_par_from_prev_block]
                        del xml_dict[xml_par_from_next_block]
                        xml_par_scentences.append(xml_par_from_prev_block + xml_par_from_next_block)
                        xml_dict[xml_par_from_prev_block + xml_par_from_next_block] = prev_block_par_meta
                    except:
                        pass

                block = 0
                xml_par_meta = []
                words = []

    # Stitch Algorithm:
    xml_with_html_content_final = []
    list_of_xml_with_html_content_final = []  # [par_text, [font_style, font_class], [par_xml_content]]
    for html_par_list in list_of_lists_of_html_par_sentences:
        result = {}
        for html_par_dict in html_par_list:
            if len(html_par_list) > 1:
                for k, v in html_par_dict.items():
                    if k in result.keys():
                        result[k] += v
                    else:
                        result[k] = v

                html_par_dict = result
            for xml_text, xml_meta in list(xml_dict.items()):
                xml_text_cp = xml_text
                xml_text = ''.join(xml_text).lstrip()
                xml_text = ''.join(c for c in xml_text if ord(c) < 128)
                html_para = ''.join(c for c in html_par_dict['text'] if ord(c) < 128)

                # String Matching
                s = SequenceMatcher(None, xml_text, html_para)
                if s.ratio() >= 0.95:
                    if len(html_par_list) == 1:
                        xml_with_html_content_final.append(html_par_dict['text'])
                        xml_with_html_content_final.append(html_par_dict['meta'])
                        xml_with_html_content_final.append(xml_meta)

                    elif len(html_par_list) > 1:
                        for html_par_dict in html_par_list:
                            xml_with_html_content_final.append(html_par_dict['text'])
                            xml_with_html_content_final.append(html_par_dict['meta'])
                        xml_with_html_content_final.append(xml_meta)
                    list_of_xml_with_html_content_final.append(xml_with_html_content_final)
                    xml_with_html_content_final = []

    new_xml_dict_with_html = {}
    font_style_dict = {}
    font_class_dict = {}
    par_xml_content_dict = {}
    par_sentence_text_dict = {}
    for i, xml_with_html_content_final in enumerate(list_of_xml_with_html_content_final):
        if len(xml_with_html_content_final) == 3:
            par_sentence_text_dict['text0'] = xml_with_html_content_final[0]
            font_style = xml_with_html_content_final[1][0]
            font_class = xml_with_html_content_final[1][1]
            par_xml_content = xml_with_html_content_final[2]

            font_style_dict['font style0'] = font_style
            font_class_dict['font class0'] = font_class
            par_xml_content_dict['xml content0'] = par_xml_content

            try:
                new_xml_dict_with_html['par' + str(i)].append({'text0': par_sentence_text_dict['text0']})
                new_xml_dict_with_html['par' + str(i)].append({'font_style0': font_style_dict['font style0']})
                new_xml_dict_with_html['par' + str(i)].append({'font_class0': font_class_dict['font class0']})
                new_xml_dict_with_html['par' + str(i)].append({'xml_content0': par_xml_content_dict['xml content0']})
            except:
                new_xml_dict_with_html['par' + str(i)] = [{'text0': par_sentence_text_dict['text0']}]
                new_xml_dict_with_html['par' + str(i)].append({'font_style0': font_style_dict['font style0']})
                new_xml_dict_with_html['par' + str(i)].append({'font_class0': font_class_dict['font class0']})
                new_xml_dict_with_html['par' + str(i)].append({'xml_content0': par_xml_content_dict['xml content0']})

            font_style_dict = {}
            font_class_dict = {}
            par_xml_content_dict = {}

        elif len(xml_with_html_content_final) > 3:
            par_xml_content = xml_with_html_content_final[-1]
            co_even = 0
            co_odd = 0
            for j, each in enumerate(xml_with_html_content_final[:-1]):
                if j % 2 == 0:
                    par_sentence_text_dict['text' + str(co_even)] = each
                    try:
                        new_xml_dict_with_html['par' + str(i)].append(
                            {'text' + str(co_even): par_sentence_text_dict['text' + str(co_even)]})
                    except:
                        new_xml_dict_with_html['par' + str(i)] = [
                            {'text' + str(co_even): par_sentence_text_dict['text' + str(co_even)]}]
                    co_even += 1

                elif j % 2 == 1:
                    try:
                        new_xml_dict_with_html['par' + str(i)].append({'font_style' + str(co_odd): each[0]})
                        new_xml_dict_with_html['par' + str(i)].append({'font_class' + str(co_odd): each[1]})
                    except:
                        new_xml_dict_with_html['par' + str(i)].append([{'font_style' + str(co_odd): each[0]}])
                        new_xml_dict_with_html['par' + str(i)].append([{'font_class' + str(co_odd): each[1]}])
                    co_odd += 1

            new_xml_dict_with_html['par' + str(i)].append({'xml_content0': par_xml_content})
    return list_of_xml_with_html_content_final


def test():
    html_dir = "/home/mahad/abbyy_dummy_dataset/html"
    xml_dir = "/home/mahad/abbyy_dummy_dataset/xml"
    html_files = glob.glob(html_dir + "/*.html")
    for html_file in html_files:
        xml_file = html_file.replace("html", "xml")
        merged_data = html_xml_merge(html_file, xml_file)
        tree = etree.parse(xml_file)
        test_tag = tree.findall("charParams")
        print()


if __name__ == '__main__':
    test()
