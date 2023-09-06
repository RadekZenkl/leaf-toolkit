import onnxruntime
import numpy as np
from pathlib import Path
import cv2
from typing import Tuple
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time
from skimage.util import view_as_blocks
import gc      
import matplotlib


class Leafnet:

    def __init__(
            self, 
            seg_model: Path = Path('segmentation.onnx'), 
            key_model: Path = Path('keypoint.onnx'),
            export_path: Path = Path('export'), 
            img_sz: int = 1024,
            debug: bool = False
            ) -> None:
                
        if debug:
            pass
        else:
            matplotlib.use('Agg')  # Without using the write only backend, memory leak occurs. 

        if img_sz == 1024:
            key_model_path = '1024px-' + str(key_model)
            seg_model_path = '1024px-' + str(seg_model)
            self.download_file(key_model_path, 'https://polybox.ethz.ch/index.php/s/EHDui6JfpLrkZij/download')
            self.download_file(seg_model_path, 'https://polybox.ethz.ch/index.php/s/fXC6ajQzTEmwXat/download')
        elif img_sz == 4096:
            key_model_path = '4096px-' + str(key_model)
            seg_model_path = '4096px-' + str(seg_model)
            self.download_file(key_model_path, 'https://polybox.ethz.ch/index.php/s/Bm4Bb8bMgnVSEPy/download')
            self.download_file(seg_model_path, 'https://polybox.ethz.ch/index.php/s/99dOEfS42s08Ukh/download')
        else:
            raise Exception("Unexpected dimension for the model, please choose from: 1024, 4096 pixel")

        self.key_session = onnxruntime.InferenceSession(
            key_model_path, 
            providers=['CPUExecutionProvider']

            )
        
        self.seg_session = onnxruntime.InferenceSession(
            seg_model_path, 
            providers=['CPUExecutionProvider']

            )

        self.export_path = export_path

        self.img_sz = img_sz
        
        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple 
        self.patch_stride = self.img_sz  

        self.debug = debug

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
        
        if self.debug:
            start_time = time.time()     
        image = cv2.imread(str(src))
        if image is None:
            raise Exception("Reading in the image: {} was unsucessful".format(str(src)))
        
        # Convert from cv2 BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # subdivide input
        patches = self.subdivide_image(image, src)

        # allocate space for results
        segmentations = np.zeros((patches.shape[:-1]))
        n_patches = patches.shape[:2]

        # inference
        for indices in np.ndindex(n_patches):   
            patch = patches[indices]
            model_input = self.prepare_model_input(patch)
            segmentations[indices] = self.models_predict(model_input)

        result = self.merge_patches(segmentations)
        if self.debug:
            print("execution time: {}".format(time.time()-start_time))

        if self.debug:
            pass
    
        else:
            # save predictions
            predictions_path = self.export_path / 'predictions' / src.name
            predictions_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(predictions_path),result.astype(np.uint8))
            

        self.visualize_predictions(src, result)
        gc.collect()

    def subdivide_image(self, image, src: Path) -> np.array:
        if image.shape[0] < self.img_sz or image.shape[1] < self.img_sz:
            raise Exception("Image: {} is smaller then required size".format(str(src)))

        # check if image should be subdivided because it is too large
        if image.shape[0] != self.img_sz or image.shape[1] != self.img_sz:
            patches = self.split_into_patches(image)  # n x img_sz x img_sz x 3
        # exactly one pass
        elif image.shape[0] == self.img_sz and image.shape[1] == self.img_sz:
            patches = np.expand_dims(image, axis=(0,1))
        else:
            raise Exception("Unexpected size: {}".format(str(src)))
        
        return patches
    
    def split_into_patches(self, image):
        patches = view_as_blocks(image, (self.img_sz,self.img_sz,3))
        patches = np.squeeze(patches, axis=2)  # remove the RGB axis which is just 1 entry

        return patches
    
    def merge_patches(self, patches: np.array) -> np.array:
        # currently only the case where patch size and its stride are equal is considered
        if self.img_sz != self.patch_stride:
            raise Exception(NotImplemented)
        
        n_blocks = patches.shape[:2]
        block_shape = tuple(np.array(patches.shape[2:]))
        merged_shape = tuple(np.array(n_blocks) * np.array(block_shape))
        merged = np.zeros(merged_shape, dtype=patches.dtype)

        for indices in np.ndindex(n_blocks):
            block = patches[indices]
            start = tuple(np.array(indices) * np.array(block_shape))
            end = tuple(np.array(start) + np.array(block_shape))
            merged[start[0]:end[0], start[1]:end[1], ...] = block

        return merged

    def prepare_model_input(self, image: np.array) -> np.array:
        input_img = image / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor
    
    def normalize_model_input(self, 
                              image: np.array,  
                              mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                              std: Tuple[float, float, float] = (0.229, 0.224, 0.225), 
                              max_pixel_value: float = 1.0) -> np.array:
        
        image = np.swapaxes(image, axis1=0, axis2=2)
        mean = np.array(mean, dtype=np.float32)
        mean *= max_pixel_value

        std = np.array(std, dtype=np.float32)
        std *= max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        image = image.astype(np.float32)
        image -= mean
        image *= denominator

        image = np.swapaxes(image, axis1=2, axis2=0)

        return image        

    
    def models_predict(self, input_tensor: np.array) -> Tuple[np.array, np.array]:

        # keypoints
        input_name = self.key_session.get_inputs()[0].name
        ort_inputs = {input_name: input_tensor}
        ort_outs = self.key_session.run(None, ort_inputs)

        predictions = np.squeeze(ort_outs[0]).T
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:6], axis=1)
        predictions = predictions[scores > 0.1, :]
        scores = scores[scores > 0.1]
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:6], axis=1)
        points = predictions[:,6:].astype(int)
    
        # segmentations 
        # input for segementations needs to be normalized
        segmentation_input = self.normalize_model_input(np.squeeze(input_tensor, 0))
        segmentation_input = np.expand_dims(segmentation_input, 0)

        input_name = self.seg_session.get_inputs()[0].name
        ort_inputs = {input_name: segmentation_input}
        ort_outs = self.seg_session.run(None, ort_inputs)

        predictions = np.squeeze(ort_outs[0])
        mask = np.argmax(predictions, axis=0).squeeze()

        # softmax (probs = torch.softmax(..., axis=0))
        axis = 0
        x_max = np.amax(predictions, axis=axis, keepdims=True)
        exp_x_shifted = np.exp(predictions - x_max)
        probs = exp_x_shifted / np.sum(exp_x_shifted, axis=axis, keepdims=True)

        low_conf = np.bitwise_not(np.sum(probs >= 0.5, axis=0))
        mask[low_conf] = 0

        # Define a dictionary to map classes to integer values
        class_mapping = {0: 5, 1: 6, 2: 7}  # Customize the class mapping as desired

        # Assign the class values to the corresponding pixels
        for cls, value in class_mapping.items():
            class_mask = class_ids == cls
            if len(class_mask) == 0:  # check if list/array is empty
                continue

            mask[points[class_mask, 1], points[class_mask, 0]] = value

        return mask
    
    def visualize_predictions(self, image_src: Path, segmentations: np.array, 
                              pycnidia_id: int = 5, rust_id: int = 6, insect_damage_id: int = 3, lesion_id: int = 2, leaf_id: int = 1):
        pycndia = np.where(segmentations == pycnidia_id)
        rust = np.where(segmentations == rust_id)

        image_bgr = cv2.imread(str(image_src))
        leaf_mask = np.zeros((image_bgr.shape), dtype=np.uint8)
        lesion_mask = np.zeros((image_bgr.shape), dtype=np.uint8)
        insect_mask = np.zeros((image_bgr.shape), dtype=np.uint8)

        color_leaf = [255,0,180]
        color_lesion = [255,150,0]
        color_insect = [80, 230, 80]

        leaf_mask[segmentations != 0 ] = color_leaf
        lesion_mask[np.bitwise_or(segmentations == lesion_id, segmentations == pycnidia_id)] = color_lesion
        insect_mask[segmentations == insect_damage_id] = color_insect

        leaf_cnts, _ = cv2.findContours(leaf_mask[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        lesi_cnts, _ = cv2.findContours(lesion_mask[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        inse_cnts, _ = cv2.findContours(insect_mask[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image_bgr, leaf_cnts, -1, color_leaf, 3)
        cv2.drawContours(image_bgr, lesi_cnts, -1, color_lesion, 3)
        cv2.drawContours(image_bgr, inse_cnts, -1, color_insect, 3)

        alpha = 0.75
        beta = (1.0 - alpha)
        image_bgr[segmentations == leaf_id] = (alpha*(image_bgr)+beta*(leaf_mask))[segmentations == leaf_id]
        image_bgr[segmentations == lesion_id] = (alpha*(image_bgr)+beta*(lesion_mask))[segmentations == lesion_id]
        image_bgr[segmentations == insect_damage_id] = (alpha*(image_bgr)+beta*(insect_mask))[segmentations == insect_damage_id]

        p_y_coords, p_x_coords = pycndia
        r_y_coords, r_x_coords = rust
        # Draw circles at the specified coordinates
        for x, y in zip(p_y_coords, p_x_coords):
            cv2.circle(image_bgr, (y, x), 5, (0,0,255), 1)
        for x, y in zip(r_y_coords, r_x_coords):
            cv2.circle(image_bgr, (y, x), 5, (0,255,0), 1)

        if self.debug:
            print('press any key within the window to continue')
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
            cv2.imshow("output", image_bgr)                       # Show image
            cv2.waitKey(0)               
        else:
            save_path = self.export_path / 'visualization' / image_src.name
            save_path.parents[0].mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), image_bgr)
    
    def download_file(self, file_path, url):
        file_path = Path(file_path)
        if not file_path.exists():
            urllib.request.urlretrieve(url, file_path)
            print(f"File downloaded successfully: {file_path}")
        else:
            print(f"File already exists: {file_path}")

    def test(self):
        test_name = Path('test.png')
        self.download_file(test_name, 'https://polybox.ethz.ch/index.php/s/Gz7bBBzHmlbl1sg/download')
        self.predict(test_name)

if __name__=='__main__':
    leafnet = Leafnet(debug=False)
    leafnet.test()
