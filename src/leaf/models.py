import cv2
import numpy as np
import torch
from ultralytics import YOLO

from torchvision.transforms import Compose
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from numpy.core.multiarray import array as array

from pathlib import Path
import urllib.request
import logging
from tqdm import tqdm
from skimage.util import view_as_blocks
import psutil
import matplotlib.pyplot as plt
from typing import Union, Tuple

from leaf.visualization import save_histogram, save_depth_overlay, save_image


# TODO when no imagesize is given, use full resolution
class BaseModel:
    """
    A base class to use for implementing new models to the pipeline. It defines most of the utilities required with respect
    to initialization, handling images, predicting and saving results. 
    """
    def __init__(self,
        export_pattern_pred: Union[str, Tuple[str, str]],  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        patch_sz: Union[int, Tuple[int, int]],
        input_scaling: Union[float, Tuple[float, float]],        
        classes_dict: dict, 
        debug: bool = False, 
        model_name: str = 'latest',
        use_gpu: bool = True, 
        cuda_device: str = 'cuda:0',
        search_pattern: str = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'],

    ) -> None:
        """
        This is a constructor placeholder, which needs to be overriden when used, otherwise
        it raises a NotImplementedError.

        Args:
            export_pattern_pred (Union[str, Tuple[str, str]]): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what. 
            patch_sz (Union[int, Tuple[int, int]]): Size of patches cropped to use for inference. The patchsize needs to be chosen so 
                that it recreates the whole image i.e., image is dividided without rests and overlaps. (e.g. 4096x4096 can 
                use 2048x2048 patches but not 2000x2000)
            input_scaling (Union[float, Tuple[float, float]]): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. 
            classes_dict (dict): This denotes a mapping between class names and their integer id.
            debug (bool, optional): When run in debug mode the models do not save predictions and some models provide additional
                insights. Defaults to False.
            model_name (str): Which model should be used. For this refer to model zoo. Defaults to 'latest'.
            use_gpu (bool, optional): When true the inference is run on a GPU. Defaults to True.
            cuda_device (str, optional): which cuda device to use. Not recommended to use. Rather use the 
                environment variable export CUDA_VISIBLE_DEVICES = ... . Defaults to 'cuda:0'.
            search_pattern (str, optional): List of image extensions used to search for images. 
                Defaults to ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'].

        Raises:
            NotImplementedError: Error to notify that this method is meant to be implemented in the inherited class.
        """
        self.classes_dict = None  # {class integer id: class name}
        self.debug = None
        self.export_pattern_pred = None  # This parameter can be adjusted when a more complex export paths need to be composed
        self.patch_sz = None
        self.input_scaling = None,


        self.use_gpu = None
        self.cuda_device = None
        self.model_path = None

        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple
        # TODO make this 2D
        self.patch_stride = self.patch_sz
        self.search_pattern = None

        # placeholder for a current image or patch name for debugging purposes
        self.current_input_name = None
        
        raise NotImplementedError

    def predict(self, src: str) -> None:
        """
        Predicts on a folder or file. Uses pathlib Path internally

        Args:
            src (str): Path to a specific file or folder
        """

        # Check what type of a source it is
        src_path = Path(src)

        if 	src_path.is_file():
            self.predict_image(src_path)

        elif src_path.is_dir():
            self.predict_folder(src_path)

    def predict_folder(self, src: Path) -> None:
        """
        This function predicts a folder. It uses recursive file search with extensions defined in the search_pattern 
            attribute.

        Args:
            src (pathlib.Path): Pathlib Path pointing to a folder.
        """

        files = sorted(
            [file for ext in self.search_pattern for file in src.rglob(ext)]
        )
        
        for file in tqdm(files):
            
            self.predict_image(file)

    def predict_image(self, src: Path) -> None:
        """
        Tests if a file is read successfully. Then it subdivides an image to patches according to the specified patch size.
        Then it conducts the inference and merges the results from individual patches and saves the results, if not in 
        debug mode. 

        Args:
            src (pathlib.Path): Pathlib Path to the image to be predicted.
        """
        # TODO this could be moved to a separate process to speed up (see pytorch datamodule)
        # it should return an iterable of images and image paths
        logging.debug("Processing File: {}".format(str(src)))
        image = cv2.imread(str(src))

        # check if the image was read sucessfully
        if image is None:
            logging.error("Reading in the image: {} was unsucessful".format(str(src)))
            return
        # Convert from cv2 BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TODO revise this 
        # subdivide input
        patches = self.subdivide_image(image, src)
        if patches is None:
            # TODO add more info here 
            logging.error(f"Skipping image: {str(src)}")
            return

        # allocate space for results
        results = np.zeros((patches.shape[:-1]))
        n_patches = patches.shape[:2]

        # inference
        for indices in np.ndindex(n_patches):
            id_str = '-'.join(map(str, indices))  # Convert indices to string 
            self.current_input_name = str(src.with_name(f"{src.stem}_{id_str}{src.suffix}"))

            patch = patches[indices]
            model_input = self.rgb_image2input(patch)
            results[indices] = self.infer_image(model_input)

        result = self.merge_patches(results)

        # Saving predictions
        if not self.debug:
            self.save_predictions(str(src), result)

    
    def subdivide_image(self, image: np.array, src: Path) -> np.array:
        """
        This function subdivides an image according to the patch_sz attribute and returns a array of patches in form of
        n x patch_sz[0] x patch_sz[1] x 3 . Currently it only allows for subdivision that mathches the image size exactly.

        Args:
            image (np.array): Image array
            src (pathllib.Path): Path to the corresponding image

        Returns:
            np.array: Batched image patches
        """

        if type(self.patch_sz) == int:
            patchsz  = (self.patch_sz, self.patch_sz)
        elif type(self.patch_sz) == tuple:
            patchsz = self.patch_sz
        elif type(self.patch_sz) == list:
            patchsz = self.patch_sz
        else:
            logging.error("Unexpected image size format: {}".format(self.patch_sz))
            return None

        if image.shape[0] < patchsz[0] or image.shape[1] < patchsz[1]:
            logging.error("Image: {} is smaller then required size".format(str(src)))
            return None

        # check if image should be subdivided because it is too large
        if image.shape[0] != patchsz[0] or image.shape[1] != patchsz[1]:
            patches = self.split_into_patches(image, patchsz)  # n x patch_sz x patch_sz x 3
        # exactly one pass
        elif image.shape[0] == patchsz[0] and image.shape[1] == patchsz[1]:
            patches = np.expand_dims(image, axis=(0,1))
        else:
            logging.error("Unexpected size: {}".format(str(src)))
            return None
        
        return patches
    
    def split_into_patches(self, image: np.array, patchsz: Union[tuple, list]) -> np.array:
        """
        Splits the image into patches according the patchsz and returns batched patches array.

        Args:
            image (np.array): numpy array with the image
            patchsz (Union[tuple, list]): patchsize denoting the x and y size

        Returns:
            np.array: batched patches array
        """

        patches = view_as_blocks(image, (*patchsz,3))
        patches = np.squeeze(patches, axis=2)  # remove the RGB axis which is just 1 entry

        return patches
    
    def merge_patches(self, patches: np.array) -> np.array:
        """
        Merge patches back to recreate the original image/pattern. It can only handle a stride which is equal to the 
        patch size.

        Args:
            patches (np.array): Batched array of patches

        Raises:
            NotImplementedError: Throw and exception regarding differing stride not implemented.

        Returns:
            np.array: merged output with the dimensions of the riginal image
        """
        # currently only the case where patch size and its stride are equal is considered
        # TODO expand this to a more general case

        if self.patch_sz != self.patch_stride:
            raise NotImplementedError
        
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
    
    def infer_image(self, model_input: torch.tensor) -> np.array:
        """
        A placeholder for downstream models inference. Needs to be implemented

        Args:
            model_input (torch.tensor): model_input tensor

        Raises:
            NotImplementedError: Error denoting, it is not implemented.
        Returns:
            np.array: model output mask
        """
        raise NotImplementedError

    def rgb_image2input(self, image: np.array) -> torch.tensor:
        """
        A placeholder for model specific preparation of the input for inference. Needs to be implemented in 
        inherited classes.

        Args:
            image (np.array): image array

        Raises:
            NotImplementedError: Error denoting, it is not implemented.

        Returns:
            torch.tensor: torch tensor ready to be processed by the model
        """
        raise NotImplementedError
    
    def test(self) -> None:
        """
        A test method to see if inference can be done sucessfully. It downloads a sample image to test/images and 
        predicts on it
        """
        test_name = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.0/BF0A1199_4096px.png', root='test/images')
        self.predict(test_name)

    def download_file(self, url: str, root: str = 'models') -> str:
        """
        Download file at a given url to a specified directory if it does not already exists and keep the name of 
        the file same as on the server.

        Args:
            url (str): url to what to download
            root (str, optional): directory where to download to. Defaults to 'models'.

        Returns:
            str: path to the downloaded file
        """

        remotefile = urllib.request.urlopen(url)
        contentdisposition = remotefile.info()['Content-Disposition']
        _, params = self.parse_name_from_header(contentdisposition)
        filename = params["filename"]

        file_path = Path(filename) if root is None else Path(root) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            logging.info(f"Downloading: {file_path}")
            urllib.request.urlretrieve(url, file_path)
            logging.info(f"File downloaded successfully: {file_path}")
        else:
            logging.info(f"File already exists: {file_path}")

        return str(file_path)

    def parse_name_from_header(self, header_value: str) -> Tuple[str, dict[str, str]]:
        """
        Parses the Content-Disposition header value and extracts the disposition type and parameters.

        Args:
            header_value (str): The raw header string, e.g., 'form-data; name="file"; filename="example.txt"'.

        Returns:
            Tuple[str, Dict[str, str]]: A tuple where the first element is the disposition type (e.g., 'form-data'),
            and the second is a dictionary of parameter key-value pairs (e.g., {'name': 'file', 'filename': 'example.txt'}).
        """
        parts = header_value.split(";")
        disposition = parts[0].strip().lower()
        params = {}

        for part in parts[1:]:
            if "=" in part:
                key, value = part.strip().split("=", 1)
                value = value.strip('"')  # Remove quotes around values
                params[key.lower()] = value

        return disposition, params

    
    def save_predictions(self, image_path: str, predictions: np.array) -> None:
        """
        This function determines the save path based on the source image name and export_pattern_pred attribute. 
        Then it saves the predictions on a specified location. 

        Args:
            image_path (str): image path of the original image
            predictions (np.array): predictions array

        Raises:
            Exception: Error when the export path cannot be determined
        """
        src = Path(image_path)

        # Compose appropriate Export path 
        if type(self.export_pattern_pred) == str:
            predictions_path = Path(self.export_pattern_pred) / src.name
        elif type(self.export_pattern_pred) == tuple:
            predictions_path = Path(str(src).replace(*self.export_pattern_pred))
        else:
            raise Exception("Unexpected path pattern")

        # save predictions
        predictions_path.parents[0].mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(predictions_path.with_suffix('.png')), predictions.astype(np.uint8))

    def get_model(self, model_name: str):
        """
        Download if not already existing and use a desired model.

        Args:
            model_name (str): Model Name. Refer to Model Zoo

        Raises:
            NotImplementedError: Error that it is not implemented
        """
        raise NotImplementedError


class TorchscriptTransformer(BaseModel):
    """
    Extension of the BaseModel class which implements further model specifics to torscript exported transformer model.
    """
    def __init__(self, 
        export_pattern_pred: Union[str, Tuple[str, str]],  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        patch_sz: Union[int, Tuple[int, int]],
        input_scaling: Union[float, Tuple[float, float]],
        classes_dict: dict,
        debug: bool = False, 
        model_name: str = 'latest',
        use_gpu: bool = True, 
        cuda_device: str = 'cuda:0',
        search_pattern: str = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'],
                 ):
        """
        Constructor of this class

        Args:
            export_pattern_pred (Union[str, Tuple[str, str]]): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what. 
            patch_sz (Union[int, Tuple[int, int]]): Size of patches cropped to use for inference. The patchsize needs to be chosen so 
                that it recreates the whole image i.e., image is dividided without rests and overlaps. (e.g. 4096x4096 can 
                use 2048x2048 patches but not 2000x2000)
            input_scaling (Union[float, Tuple[float, float]]): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. 
            classes_dict (dict): This denotes a mapping between class names and their integer id.
            debug (bool, optional): When run in debug mode the models do not save predictions and some models provide additional
                insights. Defaults to False.
            model_name (str): Which model should be used. For this refer to model zoo. Defaults to 'latest'.
            use_gpu (bool, optional): When true the inference is run on a GPU. Defaults to True.
            cuda_device (str, optional): which cuda device to use. Not recommended to use. Rather use the 
                environment variable export CUDA_VISIBLE_DEVICES = ... . Defaults to 'cuda:0'.
            search_pattern (str, optional): List of image extensions used to search for images. 
                Defaults to ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'].

        Raises:
            Exception: Error when GPU is requested but cannot be utilized.
        """
        
        self.classes_dict = classes_dict  # {class integer id: class name}
        self.debug = debug
        self.export_pattern_pred = export_pattern_pred  # This parameter can be adjusted when a more complex export paths need to be composed
        self.patch_sz = patch_sz
        self.input_scaling = input_scaling if type(input_scaling) == tuple else (input_scaling, input_scaling)

        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        self.model_path = None

        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple
        # TODO make this 2D
        self.patch_stride = self.patch_sz  
        self.search_pattern = search_pattern

        self.get_model(model_name)
        
        if use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise Exception("GPU requested a GPU but torch cannot utilize it, please check your torch and cuda installation.")

            self.model = torch.jit.optimize_for_inference(torch.jit.load(self.model_path, map_location=self.cuda_device))
            
        else:
            self.model = torch.jit.optimize_for_inference(torch.jit.load(self.model_path, map_location='cpu'))

            # use all available threads for inference
            num_threads = psutil.cpu_count(logical=True)
            torch.set_num_threads(num_threads)

    def infer_image(self, model_input: torch.tensor) -> np.array:
        """
        Torchscript Transformer specific implementation of inference using a preprossed input.

        Args:
            model_input (torch.tensor): prepared tensor for inference

        Returns:
            np.array: results encoded as a mask
        """
        with torch.no_grad():

            # scale input according to self.input_scaling
            h_orig = model_input.shape[-2]
            w_orig = model_input.shape[-1]
            h = int(model_input.shape[-2]*self.input_scaling[0])
            w = int(model_input.shape[-1]*self.input_scaling[1])
            model_input = F.resize(model_input, (h, w), interpolation=InterpolationMode.BILINEAR)

            if self.use_gpu:
                model_input = model_input.to(self.cuda_device)

            # segmentations 
            segmentation_preds = self.model(model_input)
            segmentation_preds = segmentation_preds[0].squeeze()

            mask = torch.argmax(segmentation_preds, axis=0).squeeze()
            probs = torch.softmax(segmentation_preds, axis=0)

            low_conf = torch.bitwise_not(torch.sum(probs >= 0.5, axis=0))
            mask[low_conf] = 0

            # scale back to input size
            mask = F.resize(mask.unsqueeze(0), (h_orig, w_orig), interpolation=InterpolationMode.NEAREST_EXACT).squeeze()
    
            return mask.cpu().numpy()

    def rgb_image2input(self, image: np.array) -> torch.tensor:
        """
        Prepare an image for inference with pytorch. This is specific for torchscript transformer.

        Args:
            image (np.array): array with the image

        Returns:
            torch.tensor: model input tensor ready for inference
        """
        input_img = image / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_array = np.ascontiguousarray(input_array)

        # input for segmentations needs to be normalized
        model_input = torch.from_numpy(input_array)
        model_input = F.normalize(model_input, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return model_input


class SymptomsSegmentation(TorchscriptTransformer):
    """
    Implementation of the symptoms segmentation bulding block. 
    """
    def __init__(self,
        classes_dict: dict = {1: 'leaf',
                              2: 'necrosis',
                              3: 'insect_damage',
                              4: 'powdery_mildew',
                              }, 
        export_pattern_pred: Union[str, Tuple[str, str]] = 'export/symptoms_seg/pred',  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        input_scaling: Union[float, Tuple[float, float]] = 0.5,
        patch_sz: Union[int, Tuple[int, int]] = 2048,
        *args,
        **kwargs
        ):
        """Constructor of symptoms segmentation. It provides the correct default values for various parameters.

        Args:
            classes_dict (dict, optional): This denotes a mapping between class names and their integer id. 
                Defaults to {1: 'leaf', 2: 'necrosis', 3: 'insect_damage', 4: 'powdery_mildew', }.
            export_pattern_pred (Union[str, Tuple[str, str]], optional): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what. Defaults to 'export/symptoms_seg/pred'.
            input_scaling (Union[float, Tuple[float, float]], optional): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. Defaults to 0.5.
            patch_sz (Union[int, Tuple[int, int]], optional): Size of patches cropped to use for inference. The patchsize needs to be chosen so 
                that it recreates the whole image i.e., image is dividided without rests and overlaps. (e.g. 4096x4096 can 
                use 2048x2048 patches but not 2000x2000). Defaults to 2048.
        """
        
        super().__init__(classes_dict=classes_dict, export_pattern_pred=export_pattern_pred, input_scaling=input_scaling, patch_sz=patch_sz, *args, **kwargs)

    def get_model(self, model_name: str):
        """
        Download if not already existing and use a desired model. A specific implementation for symptoms segmentation.

        Args:
            model_name (str): Model Name

        Raises:
            Exception: Error when model not found
        """
        if model_name == 'latest':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/seg_fpn_mitb3_tkbkocy7.torchscript')
        elif model_name == 'zenkl_et_al_2025b':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/seg_fpn_mitb3_tkbkocy7.torchscript')
        elif model_name == 'tracking_latest':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/seg_fpn_mitb2_0.001_1024_seg_tracking_v1.torchscript')
        elif model_name == 'latest_large':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.1/seg_fpn_mitb2_0.001_1024_seg_tracking_v1.torchscript')
        elif model_name == 'zenkl_et_al_2025a':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.0/fpn_mitb1_v4.torchscript')
        elif model_name == 'anderegg_et_al_2024':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.1/fpn_mitb1_dsnv6.torchscript')
        else:
            raise Exception("Unexpected Segmentation Model Name") 


class OrgansSegmentation(TorchscriptTransformer):
    """
    Implementation of the organ segmentation bulding block. 
    """
    def __init__(self,
        classes_dict: dict = {1: 'stem',
                              2: 'head'}, 
        export_pattern_pred: Union[str, Tuple[str, str]] = 'export/organs/pred',  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        input_scaling: Union[float, Tuple[float, float]] = 0.25,
        patch_sz: Union[int, Tuple[int, int]] = 4096,
        *args,
        **kwargs
        ):
        """
        Constructor of organ segmentation. It provides the correct default values for various parameters.

        Args:
            classes_dict (dict, optional): This denotes a mapping between class names and their integer id. 
                Defaults to {1: 'stem', 2: 'head'}.
            export_pattern_pred (Union[str, Tuple[str, str]], optional): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what. Defaults to 'export/organs/pred'.
            input_scaling (Union[float, Tuple[float, float]], optional): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. Defaults to 0.25.
            patch_sz (Union[int, Tuple[int, int]], optional): Size of patches cropped to use for inference. The patchsize needs to be chosen so 
                that it recreates the whole image i.e., image is dividided without rests and overlaps. (e.g. 4096x4096 can 
                use 2048x2048 patches but not 2000x2000). Defaults to 4096.
        """
        super().__init__(classes_dict=classes_dict, export_pattern_pred=export_pattern_pred, input_scaling=input_scaling, patch_sz=patch_sz, *args, **kwargs)

    def get_model(self, model_name: str):
        """
        Download if not already existing and use a desired model. A specific implementation for organ segmentation.

        Args:
            model_name (str): Model Name

        Raises:
            Exception: Error when model not found
        """
        if model_name == 'latest':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/org_fpn_mitb2_b6oo1xel.torchscript')
        elif model_name == 'zenkl_et_al_2025b':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/org_fpn_mitb2_b6oo1xel.torchscript')
        else:
            raise Exception("Unexpected Segmentation Model Name") 

class SymptomsDetection(BaseModel):
    """
    Implementation of the symptoms detection building block
    """
    def __init__(self, 
        export_pattern_pred: Union[str, Tuple[str, str]] = 'export/symptoms_det/pred',  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        patch_sz: Union[int, Tuple[int, int]] = 1024,
        input_scaling: Union[float, Tuple[float, float]] = 1.0,

        classes_dict: dict = {1: 'pycnidia',
                              2: 'rust'},
        debug: bool = False,
        model_name: str = 'latest',
        use_gpu: bool = True,
        cuda_device: str = 'cuda:0',
        keypoints_thresh: float = 0.212,  # optimal for pycnidia with https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/yolo11l-pose_t11z7ymj.pt
        max_det: int = 100000,
        search_pattern: str = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'],
        ):
        """
        Constructor of symptoms detection. It provides the correct default values for various parameters.

        Args:
            export_pattern_pred (Union[str, Tuple[str, str]], optional): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what. Defaults to 'export/symptoms_det/pred'.
            input_scaling (Union[float, Tuple[float, float]], optional): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. Defaults to 1.0.
            classes_dict (dict, optional): This denotes a mapping between class names and their integer id. Defaults to {1: 'pycnidia', 2: 'rust'}.
            debug (bool, optional): When run in debug mode the models do not save predictions and some models provide additional
                insights. Defaults to False.
            model_name (str, optional): Which model should be used. For this refer to model zoo. Defaults to 'latest'.
            use_gpu (bool, optional): When true the inference is run on a GPU. Defaults to True.
            cuda_device (str, optional): which cuda device to use. Not recommended to use. Rather use the 
                environment variable export CUDA_VISIBLE_DEVICES = ... . For the current version of ultralytics only 'cuda:0' works 
                as pointed out in https://github.com/ultralytics/ultralytics/issues/5801 . Defaults to 'cuda:0'.
            keypoints_thresh (float, optional): Confidence threshold for acceptance of predictions. Typically try to use
                a value that optimizes the f1 score during training. Defaults to 0.212.
            search_pattern (str, optional):  List of image extensions used to search for images. 
                Defaults to ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'].

        Raises:
            Exception: Error when GPU is requested but cannot be utilized.
        """
                
        
        self.classes_dict = classes_dict  # {class integer id: class name}
        self.debug = debug
        self.export_pattern_pred = export_pattern_pred  # This parameter can be adjusted when a more complex export paths need to be composed
        self.patch_sz = patch_sz
        self.input_scaling = input_scaling if type(input_scaling) == tuple else (input_scaling, input_scaling)

        self.use_gpu = use_gpu
        self.cuda_device = cuda_device  # Please note that only cuda:0 works as pointed out in https://github.com/ultralytics/ultralytics/issues/5801
        self.model_path = None

        self.keypoints_thresh = keypoints_thresh
        self.max_det = max_det

        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple 
        # TODO make this 2D
        self.patch_stride = self.patch_sz  
        self.search_pattern = search_pattern

        self.get_model(model_name)

        if use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise Exception("GPU requested a GPU but torch cannot utilize it, please check your torch and cuda installation.")

            self.model = YOLO(self.model_path)
            self.model.model.to(self.cuda_device)
        else:
            self.model = YOLO(self.model_path)
            self.model.model.to('cpu')
    
    
    def get_model(self, model_name: str):
        """
        Download if not already existing and use a desired model. A specific implementation for symptoms detection.

        Args:
            model_name (str): Model Name

        Raises:
            Exception: Error when model not found
        """
        if model_name == 'latest':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/yolo11l-pose_t11z7ymj.pt')
        elif model_name == 'zenkl_et_al_2025b':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/yolo11l-pose_t11z7ymj.pt')
        elif model_name == 'zenkl_et_al_2025a':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.0/yolov8m-pose-v4.4.pt')
        elif model_name == 'anderegg_et_al_2024':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v0.3.1/yolov8m-pose_v7.pt')
        else:
            raise Exception("Unexpected Keypoint Detection Model Name")  

    def infer_image(self, model_input: torch.tensor) -> np.array:
        """        
        Symptoms detection specific implementation of inference using a preprossed input.

        Args:
            model_input (torch.tensor): prepared tensor for inference

        Returns:
            np.array: results encoded as a mask
        """
        with torch.no_grad():

            # Make a placeholder for results at original size
            mask = torch.zeros_like(model_input[0,0,:,:]).int()

            # scale input according to self.input_scaling
            h = int(model_input.shape[-2]*self.input_scaling[0])
            w = int(model_input.shape[-1]*self.input_scaling[1])
            model_input = F.resize(model_input, (h, w), interpolation=InterpolationMode.BILINEAR)

            if self.use_gpu:
                mask = mask.to(self.cuda_device)
                model_input = model_input.to(self.cuda_device)

                res = self.model.predict(model_input, conf=self.keypoints_thresh, max_det=self.max_det,
                                        imgsz=(h,w), verbose=False, device=self.cuda_device)
            else:
                res = self.model.predict(model_input, conf=self.keypoints_thresh, max_det=self.max_det,
                                        imgsz=(h,w), verbose=False, device='cpu')
            # render as segmentations 
            # Assign the class values to the corresponding pixels
            if len(res[0]) == 0:  # check if list/array is empty
                return mask.cpu().detach().numpy()

            class_ids = res[0].boxes.cls.int() + 1  # offset the class ids as 0 is used as background
            # scale up the points to original resolution
            scale_tensor = torch.tensor(self.input_scaling[::-1]).reshape(1, 1, 2).to(device=res[0].keypoints.xy.device)
            points = (res[0].keypoints.xy / scale_tensor).squeeze()
  
            if len(points.shape) == 1:
                points = points.unsqueeze(0)

            # make sure that due to rounding coordinates do not exceed the image size
            points[:, 0] = torch.clip(points[:, 0], min=0, max=mask.shape[1]-1)
            points[:, 1] = torch.clip(points[:, 1], min=0, max=mask.shape[0]-1)

            mask[points[:, 1].int(), points[:, 0].int()] = class_ids

            return mask.cpu().numpy()

    def rgb_image2input(self, image: np.array) -> torch.tensor:
        """
        Prepare an image for inference with pytorch. This is specific for symptoms detection.

        Args:
            image (np.array): array with the image

        Returns:
            torch.tensor: model input tensor ready for inference
        """
        
        # input for segmentations needs to be normalized
        # model_input = F.to_tensor(image).unsqueeze(0)
        model_input = torch.from_numpy(image).unsqueeze(0)
        input_img = image / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_array = np.ascontiguousarray(input_array)
        model_input = torch.from_numpy(input_array)

        # model_input = F.normalize(model_input, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        return model_input

# Helper Classes for Depth Anything v2 adjusted from the original implementation
# https://github.com/DepthAnything/Depth-Anything-V2/tree/31dc97708961675ce6b3a8d8ffa729170a4aa273

# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
# https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/transform.py


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_AREA,
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height

        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])
        
        # resize sample
        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__resize_target:
            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)
                
            if "mask" in sample:
                sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        
        return sample


class NormalizeImage(object):
    """Normlize image by given mean and std.
    """

    def __init__(self, mean, std):
        self.__mean = mean
        self.__std = std

    def __call__(self, sample):
        sample["image"] = (sample["image"] - self.__mean) / self.__std

        return sample


class PrepareForNet(object):
    """Prepare sample for usage as network input.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample["image"], (2, 0, 1))
        sample["image"] = np.ascontiguousarray(image).astype(np.float32)

        if "depth" in sample:
            depth = sample["depth"].astype(np.float32)
            sample["depth"] = np.ascontiguousarray(depth)
        
        if "mask" in sample:
            sample["mask"] = sample["mask"].astype(np.float32)
            sample["mask"] = np.ascontiguousarray(sample["mask"])
        
        return sample


class FocusSegmentation(BaseModel):
    """
    Implementation of the focus segmentation building block
    """

    def __init__(self, 
        export_pattern_pred: Union[str, Tuple[str, str]] = 'export/focus/pred',  # single str --> export path, tuple --> used to with str.replace(*) to change parts of orignal path
        patch_sz: Union[int, Tuple[int, int]] = 4096,
        classes_dict: dict = {1: 'out_of_focus'},
        debug: bool = False, 
        model_name: str = 'latest',
        use_gpu: bool = True, 
        cuda_device: str = 'cuda:0',
        input_scaling: Union[float, Tuple[float, float]] = 0.25,
        search_pattern: str = ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'],
        buffer_scaling: float = 1.0,
                 ):
        """
        Constructor of focus segmentation. It provides the correct default values for various parameters.

        Args:
            export_pattern_pred (Union[str, Tuple[str, str]], optional): This argument controls how the results are saved.
                You can provide a single path string which defines where the images are saved. This then takes the 
                filename and the provided path to create the path for the predictions. This means that the source folder
                structure is flattened. If you want to keep the source folder structure you can provide a tuple of two 
                strings. This is then used with the str.replace() method and allows for specifying which part of the 
                file path of the image is replaced with what.. Defaults to 'export/focus/pred'.
            classes_dict (dict, optional): This denotes a mapping between class names and their integer id.. Defaults to {1: 'out_of_focus'}.
            debug (bool, optional): When run in debug mode the models do not save predictions and some models provide additional
                insights.. Defaults to False.
            model_name (str, optional): Which model should be used. For this refer to model zoo. Defaults to 'latest'.
            use_gpu (bool, optional):  When true the inference is run on a GPU. Defaults to True.
            cuda_device (str, optional): which cuda device to use. Not recommended to use. Rather use the 
                environment variable export CUDA_VISIBLE_DEVICES = ... . Defaults to 'cuda:0'.
            input_scaling (Union[float, Tuple[float, float]], optional): If one number is provide, the scaling is done symmetrically.
                It denotes a scaling factor applied before inference. The results are scaled back to the original resolution
                before saving. Defaults to 0.25.
            search_pattern (str, optional): List of image extensions used to search for images. 
                Defaults to ['*.jpg', '*.JPG', '*.jpeg', '*.png', '*.PNG'].
            buffer_scaling (float, optional): Paramer of the focus acceptance buffer. Lower values make the focus estimation
                more aggressive. Defaults to 1.0.

        Raises:
            Exception: _description_
        """
        
        self.classes_dict = classes_dict  # {class integer id: class name}
        self.debug = debug
        self.export_pattern_pred = export_pattern_pred  # This parameter can be adjusted when a more complex export paths need to be composed
        self.patch_sz = patch_sz
        self.input_scaling = input_scaling if type(input_scaling) == tuple else (input_scaling, input_scaling)

        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        self.model_path = None

        self.buffer_scaling = buffer_scaling

        # Currently the only supported stride is the input size
        # Stride and Imagesize lead to image cropping when the image isn't a multiple
        # TODO make this 2D
        self.patch_stride = self.patch_sz  
        self.search_pattern = search_pattern

        self.get_model(model_name)
 
        if use_gpu:
            # Check if GPU is available
            if not torch.cuda.is_available():
                raise Exception("GPU requested a GPU but torch cannot utilize it, please check your torch and cuda installation.")

            self.model = torch.jit.optimize_for_inference(torch.jit.load(self.model_path, map_location=self.cuda_device))
            
        else:
            self.model = torch.jit.optimize_for_inference(torch.jit.load(self.model_path, map_location='cpu'))

            # use all available threads for inference
            num_threads = psutil.cpu_count(logical=True)
            torch.set_num_threads(num_threads) 

    
    def get_model(self, model_name: str):
        """
        Download if not already existing and use a desired model. A specific implementation for focus segmentation.

        Args:
            model_name (str): Model Name

        Raises:
            Exception: Error when model not found
        """
        if model_name == 'latest':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x6144_1x3x1036x1540_DepthAnythingV2_vits.pt')
        elif model_name == '6144x4096':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/6144x4096_1x3x1540x1036_DepthAnythingV2_vits.pt')
        elif model_name == '4096x6144':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x6144_1x3x1036x1540_DepthAnythingV2_vits.pt')
        elif model_name == '4096x4096':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/4096x4096_1x3x1036x1036_DepthAnythingV2_vits.pt')
        elif model_name == '2048x2048':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/2048x2048_1x3x518x518_DepthAnythingV2_vits.pt')
        elif model_name == '1024x1024':
            self.model_path = self.download_file('https://github.com/RadekZenkl/leaf-models/releases/download/v1.0.0/1024x1024_1x3x266x266_DepthAnythingV2_vits.pt')
        else:
            raise Exception("Unexpected Keypoint Detection Model Name")  

    def infer_image(self, model_input: torch.tensor) -> np.array:
        """        
        Focus segmentation specific implementation of inference using a preprossed input.

        Args:
            model_input (torch.tensor): prepared tensor for inference

        Returns:
            np.array: results encoded as a mask
        """
        with torch.no_grad():

            image_tensor = model_input

            image = model_input.numpy()
            h, w = image.shape[:2]

            # scale input according to self.input_scaling
            model_input_2 = cv2.resize(image, (0,0), fx=self.input_scaling[1], fy=self.input_scaling[0]) 

            input_tensor = self.image2tensor(model_input_2, model_input_2.shape[1], model_input_2.shape[0])

            if self.use_gpu:
                input_tensor = input_tensor.to(self.cuda_device)
                image_tensor = image_tensor.to(self.cuda_device)

            # Depth 
            depth = self.model.forward(input_tensor)

            # scale input according to self.input_scaling
            depth = F.resize(depth, (h, w), interpolation=InterpolationMode.BILINEAR).squeeze()

            # apply postprocessing for in-focus and out-of-focus areas
            focus = self.determine_focus(depth, image_tensor).cpu().detach().numpy()

            return focus

    def rgb_image2input(self, image: np.array) -> torch.tensor:
        """
        Prepare an image for inference with pytorch. This is specific for focus segmentation.

        Args:
            image (np.array): array with the image

        Returns:
            torch.tensor: model input tensor ready for inference
        """
        
        # just pass the same array through
        model_input = torch.from_numpy(image)

        return model_input   
    
    def image2tensor(self, raw_image: np.array, input_size_w: int, input_size_h: int) -> torch.tensor:
        """
        Focus segmentation specific implementation of converting the image to a tensor for inference.

        Args:
            raw_image (np.array): image to be processed
            input_size_w (int): width shape of the image
            input_size_h (int): heigth shape of the image

        Returns:
            torch.tensor: tensor ready for inference with DepthAnythingv2
        """
                
        transform = Compose([
            Resize(
                width=input_size_w,
                height=input_size_h,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
                
        image = raw_image / 255.0
        
        image = transform({'image': image})['image']
        input_tensor = torch.from_numpy(image).unsqueeze(0)
        
        return input_tensor

    def determine_focus(self, depth: np.array, image: np.array) -> np.array:
        """
        Using the predicted depth map and original image, determine regions that are in focus.

        Args:
            depth (np.array): depth map
            image (np.array): original image

        Returns:
            np.array: _description_
        """
        k1 = 5
        k2 = 11
        dog_offset = 12
        buffer_scaling = self.buffer_scaling  
        feature_threshold = 1e4
        outlier_factor = 3

        image = torch.moveaxis(image.squeeze(), 2, 0)

        blur1 = F.gaussian_blur(image, k1)
        blur2 = F.gaussian_blur(image, k2)

        img_dog1 = blur1 - blur2 + dog_offset

        # threshold the channels simultaneously
        rgb_thresh = img_dog1 >= 14
        bin_tresh = torch.sum(rgb_thresh, axis=0) >= 3

        relevant_depths = depth[bin_tresh]
        center = torch.median(relevant_depths)
        std = torch.std(relevant_depths)

        # filter outliers
        relevant_depths_filtered  = relevant_depths[torch.abs(relevant_depths - center) < outlier_factor * std]

        # check if sufficient number of features has been found
        # TODO tune this number
        if len(relevant_depths_filtered) < feature_threshold:
            return torch.zeros_like(depth, dtype=int)
        
        # center = torch.median(relevant_depths_filtered)
        num_bins = 50  # Define the number of bins
        hist = torch.histc(relevant_depths_filtered, bins=num_bins, min=relevant_depths_filtered.min(), max=relevant_depths_filtered.max())

        # Step 3: Find the peak (the bin with the highest frequency)
        peak_bin = torch.argmax(hist)  # Index of the bin with the maximum frequency
        peak_value = hist[peak_bin]  # The frequency (value) of the peak

        # Step 4: Calculate the bin edges
        bin_edges = torch.linspace(relevant_depths_filtered.min(), relevant_depths_filtered.max(), num_bins + 1)

        # Step 5: Calculate the bin center for each bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # The bin center for the peak bin
        center = bin_centers[peak_bin.item()]


        std = torch.std(relevant_depths_filtered)

        mask = torch.bitwise_and(depth > (center - std*buffer_scaling), depth < (center + std*buffer_scaling))

        if self.debug:

            # Setup paths
            src_path = Path(self.current_input_name)
            debug_root = 'debug'
            debug_path = Path(debug_root) / src_path.stem
            debug_path.parent.mkdir(parents=True, exist_ok=True)

            # Save RGB image
            img_rgb = torch.moveaxis(image.squeeze(), 0, 2)
            save_image(f"{debug_path}_rgb.png", img_rgb, color_convert=cv2.COLOR_RGB2BGR)

            # Save DoG image
            dog_img = (torch.moveaxis(img_dog1, 0, 2).cpu().numpy().astype(float) * 255 / 14).astype(np.uint8)
            cv2.imwrite(f"{debug_path}_DoG.png", dog_img)

            # Save RGB-DoG threshold image
            rgb_thresh_img = (torch.moveaxis(rgb_thresh, 0, 2).cpu().numpy().astype(np.uint8)) * 255
            cv2.imwrite(f"{debug_path}_RGB-DoG-threshold.png", rgb_thresh_img)

            # Save DoG threshold image
            bin_thresh_img = (1 - bin_tresh.cpu().numpy()).astype(np.uint8) * 255
            cv2.imwrite(f"{debug_path}_DoG-threshold.png", bin_thresh_img)

            # Analyze depth stats
            depth_np = relevant_depths.cpu().numpy()
            initial_mean, initial_std = np.mean(depth_np), np.std(depth_np)

            filtered = depth_np[(depth_np >= initial_mean - 3 * initial_std) & 
                                (depth_np <= initial_mean + 3 * initial_std)]
            mean_d = center.cpu().numpy()
            std_d = std.cpu().numpy()

            # Normalize depth map for visualization
            float_image = depth.cpu().numpy()
            min_val, max_val = float_image.min(), float_image.max()
            normalized_image = (float_image - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(float_image)

            # Save normalized depth image
            plt.figure(figsize=(6, 9), dpi=320)
            plt.imshow(normalized_image, cmap='Spectral_r')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{debug_path}_normalized_depth.png")
            plt.close()

            # Save accepted depth overlay
            save_depth_overlay(normalized_image, mask, f"{debug_path}_accepted_depth.png")

            # Save colored histogram with peak highlight
            save_histogram(
                data=depth_np,
                mean_d=mean_d,
                std_d=std_d,
                initial_mean=initial_mean,
                initial_std=initial_std,
                output_path=f"{debug_path}_colored_hist.png",
                highlight_line=mean_d + 0.065
            )
            
        return mask
    
def test():
    """
    This function runs a dry run of the complete pipeline to validate your installation. It produces predictions in newly created `test` folder.
    """
    
    logging.basicConfig(level=logging.DEBUG)

    print("testing models")

    s_det = SymptomsDetection(export_pattern_pred='test/export/symptoms_det/pred', patch_sz=(2048, 2048))
    s_det.test()

    s_seg = SymptomsSegmentation(export_pattern_pred='test/export/symptoms_seg/pred', patch_sz=(2048, 2048))
    s_seg.test()

    o_seg = OrgansSegmentation(export_pattern_pred='test/export/organs/pred', patch_sz=(4096, 4096))
    o_seg.test()

    f_seg = FocusSegmentation(export_pattern_pred='test/export/focus/pred', patch_sz=(4096, 4096), model_name='4096x4096')
    f_seg.test()

    print('done')

if __name__ == "__main__":
    test()
