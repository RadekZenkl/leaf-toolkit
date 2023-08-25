import numpy as np
from pathlib import Path
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
from tqdm import tqdm
import time
from skimage.util import view_as_blocks
import gc      
import matplotlib
import cgi
import torch
import psutil
import torchvision.transforms.functional as F


# TODO get rid of numpy in favor of torch


class Leafnet:

    def __init__(
            self, 
            seg_model_name: str = 'latest', 
            key_model_name: str = 'latest',
            export_path: str = 'export', 
            img_sz: int = 1024,
            debug: bool = False,
            use_gpu: bool = False,
            cuda_device: str = 'cuda:0',
            ) -> None:
        
        self.debug = debug
                
        if self.debug:
            pass
        else:
            matplotlib.use('Agg')  # Without using the write only backend, memory leak occurs. 

        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        self.seg_model_path = None
        self.key_model_path = None

        # if img_sz == 1024:
        #     key_model_path = '1024px-' + str(key_model)
        #     seg_model_path = '1024px-' + str(seg_model)
        #     self.download_file(key_model_path, 'https://polybox.ethz.ch/index.php/s/EHDui6JfpLrkZij/download')
        #     self.download_file(seg_model_path, 'https://polybox.ethz.ch/index.php/s/fXC6ajQzTEmwXat/download')
        # elif img_sz == 4096:
        #     key_model_path = '4096px-' + str(key_model)
        #     seg_model_path = '4096px-' + str(seg_model)
        #     self.download_file(key_model_path, 'https://polybox.ethz.ch/index.php/s/Bm4Bb8bMgnVSEPy/download')
        #     self.download_file(seg_model_path, 'https://polybox.ethz.ch/index.php/s/99dOEfS42s08Ukh/download')
        # else:
        #     raise Exception("Unexpected dimension for the model, please choose from: 1024, 4096 pixel")

        # Download the models if not present
        # self.download_file(str(key_model), 'https://polybox.ethz.ch/index.php/s/CttzqTimBSZFRpy/download')
        # self.download_file(str(key_model), 'https://polybox.ethz.ch/index.php/s/w7CFG1RQPTFgs3Q/download')
    
        # self.download_file(str(seg_model), 'https://polybox.ethz.ch/index.php/s/DvDvQRP6Y1Kp2E8/download')
        # self.download_file(seg_model, 'https://polybox.ethz.ch/index.php/s/ZkN0usVSuMRYDiF/download')
        # self.download_file(seg_model, 'https://polybox.ethz.ch/index.php/s/MjSfybQuXlFILcO/download')
        # self.download_file(seg_model, 'https://polybox.ethz.ch/index.php/s/fXC6ajQzTEmwXat/download')

        # TODO reference to Juliens models

        if seg_model_name == 'latest':
            # Since this model is fully convolutinal, we do not care about the input size
            self.seg_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/btBCq8dNSSxJtDW/download')
        else:
            raise Exception("Unexpected Segmentation Model Name")    
        
        if key_model_name == 'latest':
            if img_sz == 1024:
                self.key_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/WqoL0ESNeyVi5EF/download')
            elif img_sz == 2048:
                self.key_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/FDPZLock7W0uOzg/download')
            elif img_sz == 4096:
                self.key_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/gvtgrwtYf3y9Wjo/download')
            else:
                raise Exception("Unexpected Keypoint Detection Model Input Size")
        else:
            raise Exception("Unexpected Keypoint Detection Model Name")    

        if use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise Exception("GPU requested but torch cannot utilize it, please check your torch and cuda installation.")

            self.key_model = torch.jit.optimize_for_inference(torch.jit.load(self.key_model_path, map_location=self.cuda_device))
            self.seg_model = torch.jit.optimize_for_inference(torch.jit.load(self.seg_model_path, map_location=self.cuda_device))
            
        else:
            self.key_model = torch.jit.optimize_for_inference(torch.jit.load(self.key_model_path, map_location='cpu'))
            self.seg_model = torch.jit.optimize_for_inference(torch.jit.load(self.seg_model_path, map_location='cpu'))

            # use all available threads for inference
            num_threads = psutil.cpu_count(logical=True)
            torch.set_num_threads(num_threads) 
        
        self.export_path = Path(export_path)
        self.img_sz = img_sz
        
        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple 
        self.patch_stride = self.img_sz  


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
        # TODO expand this to a more general case
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
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_array      
    
    def models_predict(self, input_array: np.array) -> Tuple[np.array, np.array]:

        with torch.no_grad():
            # input for segementations needs to be normalized
            segmentation_input = torch.from_numpy(input_array)
            segmentation_input = F.normalize(segmentation_input, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            keypoints_input = torch.from_numpy(input_array)

            if self.use_gpu:
                segmentation_input = segmentation_input.to(self.cuda_device)
                keypoints_input = keypoints_input.to(self.cuda_device)

            # TODO models are exported with grad
            # keypoints
            keypoints_preds = self.key_model(keypoints_input)

            # segmentations 
            segmentation_preds = self.seg_model(segmentation_input)
            
            # Keypoint post processing
            keypoints_preds = keypoints_preds[0].squeeze().T
            # Filter out object confidence scores below threshold
            scores, _ = torch.max(keypoints_preds[:, 4:6], axis=1)

            keypoints_preds = keypoints_preds[scores > 0.1, :]

            scores = scores[scores > 0.1]
            # Get the class with the highest confidence
            class_ids = torch.argmax(keypoints_preds[:, 4:6], axis=1)
            points = keypoints_preds[:,6:].int()

            segmentation_preds = segmentation_preds[0].squeeze()
            mask = torch.argmax(segmentation_preds, axis=0).squeeze()

            probs = torch.softmax(segmentation_preds, axis=0)
            low_conf = torch.bitwise_not(torch.sum(probs >= 0.5, axis=0))
            mask[low_conf] = 0

            # Define a dictionary to map classes to integer values
            class_mapping = {0: 5, 1: 6, 2: 7}  # Customize the class mapping as desired

            # Assign the class values to the corresponding pixels
            for cls, value in class_mapping.items():
                class_mask = class_ids == cls
                if len(class_mask) == 0:  # check if list/array is empty
                    continue

                mask[points[class_mask, 1], points[class_mask, 0]] = value

            return mask.cpu().detach().numpy()
    
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

    def download_file(self, url):

        remotefile = urllib.request.urlopen(url)
        contentdisposition = remotefile.info()['Content-Disposition']
        _, params = cgi.parse_header(contentdisposition)
        filename = params["filename"]

        file_path = Path(filename)
        if not file_path.exists():
            urllib.request.urlretrieve(url, file_path)
            print(f"File downloaded successfully: {file_path}")
        else:
            print(f"File already exists: {file_path}")

        return filename

    def test(self):
        test_name = self.download_file('https://polybox.ethz.ch/index.php/s/Gz7bBBzHmlbl1sg/download')
        self.predict(test_name)

if __name__=='__main__':
    # leafnet = Leafnet(debug=True)
    # leafnet = Leafnet(debug=False, img_sz=4096)
    # leafnet.test()

    # leafnet.predict('/home/radekz/Downloads/export_julien')
    # leafnet.predict('/home/radekz/Datasets/diseasenet/val')
    # leafnet.predict('/leafnet/data/setup2/cropped')
    leafnet = Leafnet(debug=False)
    leafnet.test()

