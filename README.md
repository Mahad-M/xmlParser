# xmlParser
DISCALIMER: This is a project under development and may not be considered an end product!


This repo assumes that you have the paragraph level bounding boxes and their corresponding texts. In case of tables, you must have the cell boundaries, cell texts, and cell organization in rows. This parser is written for ABBYY xml output and assumes the input in the same format. However, some portion of it is also being developed to work with AiXprt's Digital PDF parser and can work with their dataframes.


# Installation
1) clone the repo with `git clone https://github.com/Mahad-M/xmlParser.git`

2) install the requirements from requirements.txt e.g. `pip install -r requirements.txt`

# Usage

## merging the ABBYY xml and ABBYY html
use [merge_xml_html.py](https://github.com/Mahad-M/xmlParser/blob/master/merge_xml_html.py) and change the paths accordingly. The output produced is an xml file with font styling at character level.

## creating reading order from xml file
change the paths in [abbyy_parse.py](https://github.com/Mahad-M/xmlParser/blob/master/abbyy_parse.py) to your requirements and run. Note that the pdf is only for visualizing boxes. It has no effect on the reading order. The output is currently not being saved because it is still under development. We are only visualizing it for now to test on various test cases.

## creating reading order from digital pdf parser dataframe (with p tag)
change the paths in [ammar_test.py](https://github.com/Mahad-M/xmlParser/blob/master/ammar_test.py) and run. The output is currently not being saved because it is still under development. We are only visualizing it for now to test on various test cases.
