import os
import json
from pathlib import Path
if __name__ == '__main__':

    with open("/home/mahad/Downloads/midv500_coco.json", "r") as f:
        json_str = f.read()
    json_dat = json.loads(json_str)
    for image in json_dat["images"]:
        filename = Path(image["file_name"]).name
        image["file_name"] = filename
    with open('/home/mahad/Desktop/midv_500.json', 'w') as json_file:
        json.dump(json_dat, json_file)