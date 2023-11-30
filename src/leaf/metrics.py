import numpy as np
from pathlib import Path
import csv
from typing import Dict
import cv2
from tqdm import tqdm

class Evaluator:
    def __init__(self, results_path: str, debug: bool = False) -> None:
        # check if path is a csv, else raise Excecption
        results_path = Path(results_path)
        if results_path.suffix != '.csv':
             raise Exception("currently only .csv is supported")

        # give warning when file already exists
        if results_path.is_file():
             print("Overwriting the results file: {}".format(str(results_path)))

        # csv entries
        self.results_path = results_path
        self.keyes = ["name", "placl", "n_pyc", "leaf_1e6"]

        # rewrite the file
        with open(str(self.results_path), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.keyes)
                writer.writeheader()

    def predict(self, src: str):
        # Check what type of a source it is
        src_path = Path(src)

        if 	src_path.is_file():
            self.predict_image(src_path)

        elif src_path.is_dir():
            self.predict_folder(src_path)

    def predict_folder(self, src: Path):
        files = list(src.rglob('*.*'))
        
        for file in tqdm(files):
            if not file.is_file():
                raise Exception("Path: {} is not a file".format(str(file)))
            
            self.predict_image(file)

    def predict_image(self, src: Path) -> np.array:
    
        image = cv2.imread(str(src))

        if image is None:
            raise Exception("Reading in the image: {} was unsucessful".format(str(src)))
        
        # Convert from cv2 BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if all channels are not the same, RGB mask has been provided. This is currently not supported
        if not (np.all(image[:,:,0] == image[:,:,1])) or not (np.all(image[:,:,2] == image[:,:,1])) or not (np.all(image[:,:,0] == image[:,:,2])):
             raise Exception("Unsupported mask as input.") 
        
        placl_val = placl(image[:,:,0])
        pycni_num = pycndia_count(image[:,:,0])
        leaf_1e6 = leaf_area(image[:,:,0])

        vals = [str(src.name), placl_val, pycni_num, leaf_1e6]

        if len(vals) != len(self.keyes):
             raise Exception("Wrong number of values returned for the keyes: {}".format(self.keyes))

        res = {self.keyes[i]: vals[i] for i in range(len(vals))}

        self.log_results(res)

    def log_results(self, results_line: Dict):
        with open(str(self.results_path), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.keyes)
                writer.writerow(results_line)

def leaf_area(segmentations: np.array, leaf_id: int = 1) -> float:
    # Ignore background and assume that everything else besides lesion and background is leaf
    leaf = np.sum(segmentations != 0)

    return leaf/1e6

def placl(segmentations: np.array, lesion_id: int = 2) -> float:
    # Ignore background and assume that everything else besides lesion and background is leaf
    leaf = np.sum(segmentations != 0)
    lesion = np.sum(segmentations == lesion_id)
    
    return lesion /  leaf

def pycndia_count(segmentations: np.array, pycnidia_id: int = 5) -> int:

    
    return np.sum(segmentations == pycnidia_id)


if __name__=="__main__":
    # evaluator = Evaluator('results.csv', debug=True)
    # evaluator.predict('export/predictions/test.png')

    evaluator = Evaluator('luzia_results.csv', debug=True)
    evaluator.predict('/projects/leaf-toolkit/data/predictions_Luzia/predictions')

