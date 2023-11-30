
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
import cgi
import torch
import psutil
import torchvision.transforms.functional as F
from ultralytics import YOLO
from data_prep import check_file



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
            keypoints_thresh: float = 0.15,
            max_det_patch: int = 10000,
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
        self.keypoints_thresh = keypoints_thresh
        self.max_det_patch = max_det_patch


        if seg_model_name == 'latest':
            # self.seg_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/axfYtbvX32TawJn/download')
            self.seg_model_path = '/projects/leaf-toolkit/src/leaf/fpn_mitb1_dsnv6.torchscript'
        else:
            raise Exception("Unexpected Segmentation Model Name")    
        
        if key_model_name == 'latest':
            # self.key_model_path = self.download_file('https://polybox.ethz.ch/index.php/s/7OwuVJO6igaew9g/download')
            self.key_model_path = '/projects/leaf-toolkit/src/leaf/yolov8m_dsnv6.pt'
        else:
            raise Exception("Unexpected Keypoint Detection Model Name")    

        if use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise Exception("GPU requested but torch cannot utilize it, please check your torch and cuda installation.")

            self.key_model = YOLO(self.key_model_path)
            self.key_model.model.to(self.cuda_device)
            self.seg_model = torch.jit.optimize_for_inference(torch.jit.load(self.seg_model_path, map_location=self.cuda_device))
            
        else:
            self.key_model = YOLO(self.key_model_path)
            self.key_model.model.to('cpu')
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
            
            if not check_file(str(file)):
                continue
            
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
            cv2.imwrite(str(predictions_path.with_suffix('.png')),result.astype(np.uint8))
           
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
        input_array = np.ascontiguousarray(input_array)

        return input_array      
    
    def models_predict(self, input_array: np.array) -> Tuple[np.array, np.array]:

        with torch.no_grad():
            # input for segementations needs to be normalized
            model_input = torch.from_numpy(input_array)
            model_input = F.normalize(model_input, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

            if self.use_gpu:
                model_input = model_input.to(self.cuda_device)

            # keypoints
            # Keypoint model only needs input scaled to [0,1] no normalization is required
            res = self.key_model.predict(torch.from_numpy(input_array), conf=self.keypoints_thresh, max_det=self.max_det_patch,
                                         imgsz=self.img_sz, iou=0.6)

            # segmentations 
            segmentation_preds = self.seg_model(model_input)

            class_ids = res[0].boxes.cls
            points = res[0].keypoints.xy.squeeze().int()

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

                if len(points.shape) == 1:
                    points = points.unsqueeze(0)

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
        test_name = self.download_file('https://polybox.ethz.ch/index.php/s/YamWv4LWcuM9Tto/download')
        self.predict(test_name)


if __name__=='__main__':
    # leafnet = Leafnet(debug=False, img_sz=1024, use_gpu=False, keypoints_thresh=0.15, export_path='/projects/leaf-toolkit/data/images_exp')
    # # leafnet.test()
    # leafnet.predict('/projects/leaf-toolkit/data/images')

    # leafnet = Leafnet(debug=False, img_sz=1024, use_gpu=True, keypoints_thresh=0.15, export_path='/projects/leaf-toolkit/src/leaf/test_export')
    # leafnet.predict('/projects/leaf-toolkit/src/leaf/test_data')

    leafnet = Leafnet(debug=False, img_sz=1024, use_gpu=True, keypoints_thresh=0.1, export_path='/projects/leaf-toolkit/data/predictions_Luzia')
    leafnet.predict('/projects/leaf-toolkit/data/export_Luzia')


