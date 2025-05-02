import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import (mean_absolute_percentage_error,
                                     mean_squared_error)
from torchmetrics.functional.classification import binary_jaccard_index
from tqdm import tqdm


class NamingConstants:
    """
    A collection of constant names used for organizing different types of
    image predictions and ground truths. These names refer to the folder 
    structure of the results, respectively where to find what. If new
    block are introduced to the processing pipeline, extend it here. 

    Attributes:
        SYMPTOMS_DET (str): Identifier for symptom detection predictions.
        SYMPTOMS_SEG (str): Identifier for symptom segmentation predictions.
        ORGANS (str): Identifier for organ segmentation predictions.
        FOCUS (str): Identifier for focus classification.
        GT_SYMPTOMS_DET (str): Identifier for ground truth symptom detection.
        GT_SYMPTOMS_SEG (str): Identifier for ground truth symptom segmentation.
        GT_ORGANS (str): Identifier for ground truth organ segmentation.
    """
    SYMPTOMS_DET = 'symptoms_det'
    SYMPTOMS_SEG = 'symptoms_seg'
    ORGANS = 'organs'
    FOCUS = 'focus'
    GT_SYMPTOMS_DET = 'gt_symptoms_det'  # ground truth
    GT_SYMPTOMS_SEG = 'gt_symptoms_seg'  # ground truth
    GT_ORGANS = 'gt_organs'  # ground truth


class PredictionIds:
    """
    Integer ID mappings for different prediction classes used in segmentation
    and detection tasks for plant imagery. When new models can handle additional
    classes, their new respective ids should be added here

    Attributes:
        LEAF_BACKGROUND (int): Background class in leaf segmentation (ID: 0).
        LEAF (int): Leaf class in leaf segmentation (ID: 1).
        LESION (int): Lesion symptom in segmentation (ID: 2).
        INSECT_DAMAGE (int): Insect damage symptom in segmentation (ID: 3).
        POWDERY_MILDEW (int): Powdery mildew symptom in segmentation (ID: 4).

        PYCNIDIA (int): Pycnidia symptom in detection (ID: 1).
        RUST (int): Rust symptom in detection (ID: 2).

        OUT_OF_FOCUS (int): Class for out-of-focus regions (ID: 0).
        IN_FOCUS (int): Class for in-focus regions (ID: 1).

        ORGAN_BACKGROUND (int): Background class in organ segmentation (ID: 0).
        HEAD (int): Head organ class in organ segmentation (ID: 1).
        STEM (int): Stem organ class in organ segmentation (ID: 2).
    """

    # Symptoms segmentation
    LEAF_BACKGROUND = 0
    LEAF = 1
    LESION = 2
    INSECT_DAMAGE = 3
    POWDERY_MILDEW = 4

    # Symptoms detection
    PYCNIDIA = 1
    RUST = 2

    # Focus
    OUT_OF_FOCUS = 0
    IN_FOCUS = 1

    # Organs
    ORGAN_BACKGROUND = 0
    HEAD = 1
    STEM = 2


class BasePredictionsMerger(Dataset):
    """
    Base class for merging prediction masks from multiple subfolders in a structured directory.

    This class is used to iterate through synchronized sets of prediction files
    (e.g., segmentation masks) located in multiple subfolders. It ensures that
    each set of files is present in all expected subfolders and prepares them
    for further processing.

    Attributes:
        root_folder (str): Root directory containing prediction subfolders.
        file_extension (str): File extension used to identify prediction files.
        subfolder_filepaths (dict): Mapping of subfolder names to their respective file paths.
        max_count (int): Number of synchronized prediction sets.
        current_id (int): Current index used for iteration.
    """
    def __init__(self,
                 root_folder: str, 
                 file_extension: str,
                ):
        """
        Initializes the BasePredictionsMerger.

        Args:
            root_folder (str): Root directory containing prediction subfolders.
            file_extension (str): File extension to look for (e.g., '*.png').
        """
        
        self.root_folder = root_folder
        self.file_extension = file_extension
        self.subfolder_filepaths = {}

        self.max_count = self.scan_predictions_root(self.root_folder)
        self.current_id = 0

    def __len__(self):
        """
        Returns the total number of prediction sets.

        Returns:
            int: Total number of file sets across subfolders.
        """
        return int(self.max_count)
    
    # resets iteration
    def __iter__(self):
        """
        Resets the iterator.

        Returns:
            BasePredictionsMerger: Iterator object.
        """
        self.current_id = 0
        return self
    
    def __getitem__(self, idx):
        """
        Retrieves the prediction stack for a given index.

        Args:
            idx (int): Index of the prediction set.

        Returns:
            tuple: Filepath and dictionary of masks for each subfolder.
        """
        # yield current stack of predictions
        filepath, stack = self.processing_step(idx)
        return filepath, stack

    # provides next object, resp. finishes the iteration
    def __next__(self):
        """
        Retrieves the next prediction stack in the iteration.

        Returns:
            tuple: Filepath and dictionary of masks for each subfolder.

        Raises:
            StopIteration: When all prediction sets have been iterated through.
        """
        # check if iterations are finished
        if self.current_id >= self.max_count:
            raise StopIteration
        
        # yield current stack of predictions
        filepath, stack = self.processing_step(self.current_id)
        self.current_id += 1
        return filepath, stack
     
    def scan_predictions_root(self, root_folder: str):
        """
        Scans the root directory for prediction files and validates file consistency across subfolders.

        Args:
            root_folder (str): The root folder containing prediction subfolders.

        Returns:
            int: Number of prediction sets found.

        Raises:
            Exception: If subfolders have mismatched file counts or missing files.
        """

        # check if all subfolders yield the same number of files
        n_files = []
        for subfolder in self.prediction_subfolders.values():
            subfolder_files = (Path(root_folder) / subfolder).rglob(self.file_extension)
            subfolder_files = [file for file in subfolder_files]
            n_subfolder_files = len(subfolder_files)

            if n_subfolder_files == 0:
                raise Exception(f"no files found for: {str((Path(root_folder) / subfolder))}")
                
            n_files.append(n_subfolder_files)

        if len(set(n_files)) != 1:
            raise Exception(f"different number of files found for: {root_folder}")
        

        # take the first subfolder as a reference
        reference_subfolder = self.get_reference_subfolder()
        reference_files = sorted([ref_file for ref_file in (Path(root_folder) / reference_subfolder).rglob(self.file_extension)])

        # iterate and check if all corresponding files are present 
        print("checking if all files are present")
        for reference_file in tqdm(reference_files):
            self.check_if_missing(str(reference_file))
        
        print(f"{n_files[0]} prediction sets found")
        print("all files present \n")

        # Save all filepaths for each subfolder
        self.subfolder_filepaths = {}
        for key, subfolder in self.prediction_subfolders.items():
            subfolder_files = (Path(root_folder) / subfolder).rglob(self.file_extension)
            subfolder_files = sorted([file for file in subfolder_files])
        
            self.subfolder_filepaths.update({key: subfolder_files})

        return n_files[0]

    def get_reference_subfolder(self):
        """
        Returns the reference subfolder used for consistency checks.

        Returns:
            str: Name of the first subfolder to use as reference.
        """
        return list(self.prediction_subfolders.values())[0]

    def check_if_missing(self, file_path: str):
        """
        Checks whether a given reference file is present in all other subfolders.

        Args:
            file_path (str): Path to the file in the reference subfolder.

        Raises:
            Exception: If the corresponding file is missing in any subfolder.
        """
        reference_subfolder = self.get_reference_subfolder()
 
        # Iterate all subfolders
        for subfolder in self.prediction_subfolders.values():
            # skip reference subfolder 
            if subfolder == reference_subfolder:
                continue

            # check if file exists
            querry_path = Path(file_path.replace(reference_subfolder, subfolder))
            if not querry_path.exists():
                raise Exception(f"File {str(querry_path)} does not exist")


    def processing_step(self, id: int) -> tuple[str, dict[str: np.array]]:
        """
        Loads prediction masks for a specific index from all subfolders.

        Args:
            id (int): Index of the prediction set.

        Returns:
            tuple:
                str: Filepath of the reference mask.
                dict: Dictionary mapping subfolder keys to their corresponding prediction masks (as tensors).

        Raises:
            Exception: If the mask encoding is not supported.
        """

        results = {}
        for key in self.subfolder_filepaths.keys():
            filepath = str(self.subfolder_filepaths[key][id])
            # Read in Mask
            mask = cv2.imread(filepath)
            # Check if mask is grayscale or RGB and adjust accordingly to produce one channel
            if len(mask.shape) == 3:  # BGR image
                if np.all(mask[:,:,0] == mask[:,:,1]) and np.all(mask[:,:,1] == mask[:,:,2]):  # All all values exactly the same?
                   mask = mask[:,:,2]  # pick Red channel
                elif (np.sum(mask[:,:,0]) == 0) or (np.sum(mask[:,:,1] == 0)):  # There are no values in other channels then Red
                    mask = mask[:,:,2]  # pick Red channel
                else:
                    raise Exception(f"Unexpected class encoding in the file {filepath}")

            results.update({key: torch.from_numpy(mask)})
        
        return filepath, results


class CanopyPredictionsMerger(BasePredictionsMerger):
    """
    Merges canopy prediction masks from predefined subfolders.

    Inherits from `BasePredictionsMerger` and specifies subfolders for canopy-related predictions:
    symptoms detection, segmentation, organs, and focus. This class handles merging of prediction
    masks for specific tasks within the canopy.
    """

    def __init__(self,
                 root_folder: str, 
                 file_extension: str = '*.png', 
                 # Do not change the keys, only adjust the values if necessary
                 prediction_subfolders: dict = {
                     NamingConstants.SYMPTOMS_DET: 'symptoms_det/pred',
                     NamingConstants.SYMPTOMS_SEG: 'symptoms_seg/pred',
                     NamingConstants.ORGANS: 'organs/pred',
                     NamingConstants.FOCUS: 'focus/pred',
                     },
                ):
        """
        Initializes the `CanopyPredictionsMerger` object with the specified prediction subfolders.

        Args:
            root_folder (str): Root directory containing prediction subfolders.
            file_extension (str): File extension to look for (default: '*.png').
            prediction_subfolders (dict): Mapping of prediction types to subfolder paths.
        """
        
        self.prediction_subfolders = prediction_subfolders
        super().__init__(root_folder=root_folder, file_extension=file_extension)        


class CanopyBenchmarkMerger(BasePredictionsMerger):
    """
    Merges benchmark prediction masks and ground truth from predefined subfolders.

    This class handles both predictions and corresponding ground truth masks for evaluation purposes.
    It inherits from `BasePredictionsMerger` and includes additional ground truth folders.

    Attributes:
        prediction_subfolders (dict): Mapping of prediction and ground truth types to subfolder paths.
        focus_override (str, optional): Custom path to override the default focus path.
    """
    def __init__(self,
                 root_folder: str, 
                 file_extension: str = '*.png', 
                 # Do not change the keys, only adjust the values if necessary
                 prediction_subfolders: dict = {
                     NamingConstants.SYMPTOMS_DET: 'symptoms_det/pred',
                     NamingConstants.SYMPTOMS_SEG: 'symptoms_seg/pred',
                     NamingConstants.ORGANS: 'organs/pred',
                     NamingConstants.FOCUS: 'focus/pred',
                     NamingConstants.GT_SYMPTOMS_DET: 'symptoms_det/gt',
                     NamingConstants.GT_SYMPTOMS_SEG: 'symptoms_seg/gt',
                     NamingConstants.GT_ORGANS: 'organs/gt',
                     },
                 focus_override: str = None,
                 ):
        """
        Initializes the `CanopyBenchmarkMerger` object with the specified prediction and ground truth subfolders.

        Args:
            root_folder (str): Root directory containing prediction subfolders.
            file_extension (str): File extension to look for (default: '*.png').
            prediction_subfolders (dict): Mapping of prediction and ground truth types to subfolder paths.
            focus_override (str, optional): Custom path to override the default focus path (default: None).
        """

        if focus_override is not None:
            prediction_subfolders[NamingConstants.FOCUS] = focus_override

        self.prediction_subfolders = prediction_subfolders
        super().__init__(root_folder=root_folder, file_extension=file_extension)        


class FlatLeavesPredictionsMerger(BasePredictionsMerger):
    """
    Merges flat leaf prediction masks from predefined subfolders.

    This class handles merging of prediction masks specific to flat leaf prediction tasks, including
    symptoms detection and segmentation.

    Attributes:
        prediction_subfolders (dict): Mapping of prediction types to subfolder paths.
    """

    def __init__(self,
                 root_folder: str, 
                 file_extension: str = '*.png', 
                 # Do not change the keys, only adjust the values if necessary
                 prediction_subfolders: dict = {
                     NamingConstants.SYMPTOMS_DET: 'symptoms_det/pred',
                     NamingConstants.SYMPTOMS_SEG: 'symptoms_seg/pred',
                     },
                ):
        """
        Initializes the `FlatLeavesPredictionsMerger` object with the specified prediction subfolders.

        Args:
            root_folder (str): Root directory containing prediction subfolders.
            file_extension (str): File extension to look for (default: '*.png').
            prediction_subfolders (dict): Mapping of prediction types to subfolder paths.
        """
        
        self.prediction_subfolders = prediction_subfolders
        super().__init__(root_folder=root_folder, file_extension=file_extension)        


class BaseEvaluator:
    """
    Base class for evaluating prediction results and logging them to a CSV file.

    This class provides methods for evaluating prediction results, logging them to a CSV file, and
    computing evaluation metrics. It is intended to be subclassed, with the `compute_metrics` method
    implemented in subclasses to provide specific evaluation logic.

    Attributes:
        results_path (str): Path to the CSV file for storing evaluation results.
        filename_key (str): Column name for filenames in the results file.
    """
    def __init__(self, results_path: str) -> None:
        """
        Initializes the evaluator and prepares the output CSV file.

        Args:
            results_path (str): Path to a .csv file or directory where the results will be saved.

        Raises:
            ValueError: If the provided path does not end with '.csv'.
        """

        results_path = Path(results_path)

        # Handle directory input: create a results.csv inside it
        if results_path.is_dir():
            results_path = results_path / "results.csv"

        # If path has no .csv suffix now, raise exception
        if results_path.suffix != '.csv':
            raise ValueError(f"Invalid results path: {results_path}. Must be a .csv file or directory.")

        # Warn if file exists and will be overwritten
        if results_path.exists():
            logging.warning(f"Overwriting the results file: {results_path}")

        # Ensure parent directory exists
        results_path.parent.mkdir(parents=True, exist_ok=True)

        # csv entries
        self.results_path = str(results_path)

        # rewrite the file
        self.filename_key = 'filename'
        header = [self.filename_key] + self.resulting_keys
        with open(str(self.results_path), 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()

    def predict(self, filepath: str, prediction_stack: dict[str: np.array]):
        """
        Method to predict and log results.

        Args:
            filepath (str): Path of the image or file.
            prediction_stack (dict[str: np.array]): Prediction results in a dictionary form.
        """

        res = self.compute_metrics(prediction_stack)
        self.log_results(filepath[0], res)  # due to batching, reduce dim of filepath

    def log_results(self, filepath: str, results: Dict):
        """
        Method to log the results into a CSV file.

        Args:
            filepath (str): Path to the file.
            results (Dict): The results dictionary to be logged.
        """

        # Extract filename 
        filename = str(Path(filepath).name)

        # Update the results dictionary
        results_line = {self.filename_key: filename}
        results_line.update(results)

        with open(str(self.results_path), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results_line.keys())
                writer.writerow(results_line)

    def compute_metrics(self, prediction_stack: dict[str: np.array]) -> dict[str: float]:
        """
        Abstract method to compute evaluation metrics.
        This method should be implemented by subclasses.

        Args:
            prediction_stack (dict[str: np.array]): Dictionary of predicted data.

        Returns:
            dict[str: float]: A dictionary of computed metrics.
        """
        raise NotImplementedError


# TODO expand for powdery mildew
class CanopyEvaluator(BaseEvaluator):
    """
    A subclass of BaseEvaluator for evaluating canopy-related metrics.

    This evaluator focuses on metrics specific to canopy evaluation, such as leaf area, 
    pycnidia density, rust density, and the fraction of damaged areas. It computes these metrics 
    based on the prediction results for canopy symptoms and structures.

    Attributes:
        resulting_keys (list): The list of keys for metrics related to canopy evaluation.
    """

    def __init__(self, results_path: str):
        """
        Initializes the CanopyEvaluator and sets up the result keys for canopy evaluation.

        Args:
            results_path (str): Path to a .csv file or directory where the results will be saved.
        """

        self.resulting_keys = ['reference_leaf_1e6', 'placl', 'n_pycnidia', 'pycnidia_density_1e-6', 'n_rust', 'rust_density_1e-6', 'focus_fraction', 'insect_fraction']
        super().__init__(results_path)

    def compute_metrics(self, prediction_stack: dict[str: np.array]) -> dict[str: float]:
        """
        Method to compute specific evaluation metrics for canopy evaluation.

        Args:
            prediction_stack (dict[str: np.array]): Dictionary of predicted data.

        Returns:
            dict[str: float]: A dictionary containing computed metrics.
        """

        # Check if all required data is present in the prediction stack
        required_data = [NamingConstants.SYMPTOMS_DET, NamingConstants.SYMPTOMS_SEG, NamingConstants.FOCUS, NamingConstants.ORGANS]
        for required_key in required_data:
            if not required_key in prediction_stack.keys():
                raise Exception(f"required data {required_key} not present in the provided data")
            
        # Define keys for the results
        focus_fraction = torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS) / (
            torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS) + torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.OUT_OF_FOCUS))
        
        relevant_area = torch.logical_and((prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS), (prediction_stack[NamingConstants.ORGANS] == PredictionIds.ORGAN_BACKGROUND))  # Area in focus which is not head or stem
        reference_leaf_area_1e6 = torch.sum(relevant_area) / 1e6

        lesions = torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION))  # Relevant Area which is lesions
        placl = torch.sum(lesions) / torch.sum(relevant_area)

        n_pycnidia = torch.sum(torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.PYCNIDIA)))  # Relevant Area which is pycnidia
        pycnidia_density = n_pycnidia / torch.sum(lesions) / 1e6
        n_rust = torch.sum(torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.RUST)))  # Relevant Area which is pycnidia
        rust_density = n_rust / reference_leaf_area_1e6

        insect_damage = torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE))  # Relevant Area which is insect damage
        insect_fraction = torch.sum(insect_damage) / torch.sum(relevant_area)

        return dict(zip(self.resulting_keys, [float(reference_leaf_area_1e6), 
                                              float(placl), 
                                              int(n_pycnidia), 
                                              float(pycnidia_density), 
                                              int(n_rust), 
                                              float(rust_density), 
                                              float(focus_fraction), 
                                              float(insect_fraction)]))
    

class CanopyBenchmarkEvaluator(BaseEvaluator):
    """
    A subclass of BaseEvaluator for evaluating benchmark metrics for canopy predictions.

    This evaluator extends the base evaluator to include benchmark metrics, 
    such as intersection-over-union (IoU), mean squared error (MSE), and mean absolute percentage error (MAPE) 
    for various symptoms and lesions on the canopy. It compares predicted values against ground truth.

    Attributes:
        resulting_keys (list): The list of keys for benchmark metrics for canopy evaluation.
    """

    def __init__(self, results_path: str):
        """
        Initializes the CanopyBenchmarkEvaluator and sets up the result keys for benchmark evaluation.

        Args:
            results_path (str): Path to a .csv file or directory where the results will be saved.
        """

        self.resulting_keys = [
            'necrosis_iou', 'pred_placl', 'gt_placl', 'placl_mse', 'pred_n_pycndia', 'gt_n_pycnidia', 'pred_n_rust', 'gt_n_rust', 'pycnidia_mse', 'pycnidia_mape', 'rust_mse', 'rust_mape','focus_fraction', 'pred_reference_leaf_area_1e6', 'gt_reference_leaf_area_1e6',
            'iou_damage', 'iou_necrosis', 'iou_insect_damage', 'iou_powdery_mildew', 'iou_heads', 'iou_stems',
            'focus_iou_damage', 'focus_iou_necrosis', 'focus_iou_insect_damage', 'focus_iou_powdery_mildew', 'focus_iou_heads', 'focus_iou_stems',
            'ratio_necrosis', 'ratio_insect_damage', 'ratio_powdery_mildew', 'ratio_heads', 'ratio_stems'
            ]
        super().__init__(results_path) 

    def compute_metrics(self, prediction_stack: dict[str: np.array]) -> dict[str: float]:
        """
        Method to compute benchmark evaluation metrics.

        Args:
            prediction_stack (dict[str: np.array]): Dictionary of predicted data.

        Returns:
            dict[str: float]: A dictionary containing computed metrics.
        """

        # Check if all required data is present in the prediction stack
        required_data = [NamingConstants.SYMPTOMS_DET, NamingConstants.SYMPTOMS_SEG, NamingConstants.FOCUS, NamingConstants.ORGANS]
        for required_key in required_data:
            if not required_key in prediction_stack.keys():
                raise Exception(f"required data {required_key} not present in the provided data")
            
        # Define keys for the results
        focus_fraction = torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS) / (
            torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS) + torch.sum(prediction_stack[NamingConstants.FOCUS] == PredictionIds.OUT_OF_FOCUS))

        
        # Predictions
        relevant_area = torch.logical_and((prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS), (prediction_stack[NamingConstants.ORGANS] == PredictionIds.ORGAN_BACKGROUND))  # Area in focus which is not head or stem
        reference_leaf_area_1e6 = torch.sum(relevant_area) / 1e6
        lesions = torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION))  # Relevant Area which is lesions
        placl = torch.sum(lesions) / torch.sum(relevant_area)

        # Ground truth
        gt_relevant_area = torch.logical_and((prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS), (prediction_stack[NamingConstants.GT_ORGANS] == PredictionIds.ORGAN_BACKGROUND))  # Area in focus which is not head or stem
        gt_reference_leaf_area_1e6 = torch.sum(gt_relevant_area) / 1e6
        gt_lesions = torch.logical_and(gt_relevant_area, (prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.LESION))  # Relevant Area which is lesions
        gt_placl = torch.sum(gt_lesions) / torch.sum(gt_relevant_area)

        # Pred vs Ground Truth 
        placl_mse = mean_squared_error(placl, gt_placl)
        necrosis_iou = binary_jaccard_index(lesions, gt_lesions)



        n_pycnidia = torch.sum(torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.PYCNIDIA)))  # Relevant Area which is pycnidia
        gt_n_pycnidia = torch.sum(torch.logical_and(gt_relevant_area, (prediction_stack[NamingConstants.GT_SYMPTOMS_DET] == PredictionIds.PYCNIDIA)))  # Relevant Area which is pycnidia

        # pycnidia_density = n_pycnidia / torch.sum(lesions) / 1e6
        n_rust = torch.sum(torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.RUST)))  # Relevant Area which is pycnidia
        gt_n_rust = torch.sum(torch.logical_and(gt_relevant_area, (prediction_stack[NamingConstants.GT_SYMPTOMS_DET] == PredictionIds.RUST)))  # Relevant Area which is pycnidia

        pycnidia_mse = mean_squared_error(n_pycnidia, gt_n_pycnidia)
        pycnidia_mape = mean_absolute_percentage_error(n_pycnidia, gt_n_pycnidia)
        rust_mse = mean_squared_error(n_rust, gt_n_rust)
        rust_mape = mean_absolute_percentage_error(n_rust, gt_n_rust)

        # evaluation iou metrics
        iou_damage = binary_jaccard_index(prediction_stack[NamingConstants.SYMPTOMS_SEG] != 0, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] != 0)
        iou_necrosis = binary_jaccard_index(prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.LESION)
        iou_insect_damage = binary_jaccard_index(prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE)
        iou_powdery_mildew = binary_jaccard_index(prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.POWDERY_MILDEW, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.POWDERY_MILDEW)
        iou_heads = binary_jaccard_index(prediction_stack[NamingConstants.ORGANS] == PredictionIds.HEAD, prediction_stack[NamingConstants.GT_ORGANS] == PredictionIds.HEAD)
        iou_stems = binary_jaccard_index(prediction_stack[NamingConstants.ORGANS] == PredictionIds.STEM, prediction_stack[NamingConstants.GT_ORGANS] == PredictionIds.STEM)

        # evaluation iou metrics w. focus
        focus_iou_damage = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.SYMPTOMS_SEG] != 0), 
                                                torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] != 0))
        focus_iou_necrosis = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION), 
                                                  torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.LESION))
        focus_iou_insect_damage = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE), 
                                                       torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE))
        focus_iou_powdery_mildew = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.POWDERY_MILDEW), 
                                                        torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_SYMPTOMS_SEG] == PredictionIds.POWDERY_MILDEW))
        focus_iou_heads = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.ORGANS] == PredictionIds.HEAD), 
                                               torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_ORGANS] == PredictionIds.HEAD))
        focus_iou_stems = binary_jaccard_index(torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.ORGANS] == PredictionIds.STEM), 
                                               torch.logical_and(prediction_stack[NamingConstants.FOCUS] == PredictionIds.IN_FOCUS, prediction_stack[NamingConstants.GT_ORGANS] == PredictionIds.STEM))


        # class extent
        ratio_necrosis = torch.mean((prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION).int().float())
        ratio_insect_damage = torch.mean((prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE).int().float())
        ratio_powdery_mildew = torch.mean((prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.POWDERY_MILDEW).int().float())
        ratio_heads = torch.mean((prediction_stack[NamingConstants.ORGANS] == PredictionIds.HEAD).int().float())
        ratio_stems = torch.mean((prediction_stack[NamingConstants.ORGANS] == PredictionIds.STEM).int().float())

        return dict(zip(self.resulting_keys, [
            float(necrosis_iou), 
            float(placl),
            float(gt_placl),
            float(placl_mse), 
            int(n_pycnidia),
            int(gt_n_pycnidia),
            int(n_rust),
            int(gt_n_rust),
            float(pycnidia_mse),
            float(pycnidia_mape),
            float(rust_mse),
            float(rust_mape),
            float(focus_fraction), 
            float(reference_leaf_area_1e6),
            float(gt_reference_leaf_area_1e6),

            float(iou_damage),
            float(iou_necrosis), 
            float(iou_insect_damage), 
            float(iou_powdery_mildew), 
            float(iou_heads), 
            float(iou_stems), 

            float(focus_iou_damage),
            float(focus_iou_necrosis), 
            float(focus_iou_insect_damage), 
            float(focus_iou_powdery_mildew), 
            float(focus_iou_heads), 
            float(focus_iou_stems), 

            float(ratio_necrosis),
            float(ratio_insect_damage),
            float(ratio_powdery_mildew),
            float(ratio_heads),
            float(ratio_stems),

            ]))


class FlatLeavesEvaluator(BaseEvaluator):
    """
    A subclass of BaseEvaluator for evaluating leaf-related metrics for flat leaves.

    This evaluator focuses on metrics related to leaf symptoms, such as lesions and pycnidia,
    within flat leaves. It computes relevant metrics such as leaf area, lesion area, and pycnidia density.

    Attributes:
        resulting_keys (list): The list of keys for metrics related to flat leaf evaluation.
    """

    def __init__(self, results_path: str):
        """
        Initializes the FlatLeavesEvaluator and sets up the result keys for flat leaf evaluation.

        Args:
            results_path (str): Path to a .csv file or directory where the results will be saved.
        """

        self.resulting_keys = ['reference_leaf_1e6', 'placl', 'n_pycnidia', 'pycnidia_density_1e-6', 'n_rust', 'rust_density_1e-6', 'insect_fraction']
        super().__init__(results_path)

    def compute_metrics(self, prediction_stack: dict[str: np.array]) -> dict[str: float]:
        """
        Method to compute specific evaluation metrics for flat leaves evaluation.

        Args:
            prediction_stack (dict[str: np.array]): Dictionary of predicted data.

        Returns:
            dict[str: float]: A dictionary containing computed metrics.
        """

        # Check if all required data is present in the prediction stack
        required_data = [NamingConstants.SYMPTOMS_DET, NamingConstants.SYMPTOMS_SEG]
        for required_key in required_data:
            if not required_key in prediction_stack.keys():
                raise Exception(f"required data {required_key} not present in the provided data")
            
        # Define keys for the results
        relevant_area = prediction_stack[NamingConstants.SYMPTOMS_SEG] != PredictionIds.LEAF_BACKGROUND
        reference_leaf_area_1e6 = torch.sum(relevant_area) / 1e6

        lesions = torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.LESION))  # Relevant Area which is lesions
        placl = torch.sum(lesions) / torch.sum(relevant_area)

        n_pycnidia = torch.sum(np.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.PYCNIDIA)))  # Relevant Area which is pycnidia
        pycnidia_density = n_pycnidia / torch.sum(lesions) / 1e6
        n_rust = torch.sum(np.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_DET] == PredictionIds.RUST)))  # Relevant Area which is pycnidia
        rust_density = n_rust / reference_leaf_area_1e6

        insect_damage = torch.logical_and(relevant_area, (prediction_stack[NamingConstants.SYMPTOMS_SEG] == PredictionIds.INSECT_DAMAGE))  # Relevant Area which is insect damage
        insect_fraction = torch.sum(insect_damage) / torch.sum(relevant_area)

        return dict(zip(self.resulting_keys, [float(reference_leaf_area_1e6), 
                                              float(placl), 
                                              int(n_pycnidia), 
                                              float(pycnidia_density), 
                                              int(n_rust), 
                                              float(rust_density), 
                                              float(insect_fraction)]))


def flat_leaves_predictions_iterator(root_folder: str, num_workers: int = 16):
    """
    Creates a DataLoader for iterating over flat leaves prediction data.

    This function prepares a DataLoader to iterate over flat leaves prediction data from the specified
    root folder. It utilizes a custom data-merging class (`FlatLeavesPredictionsMerger`) to load and 
    process the data in parallel using multiple workers.

    Args:
        root_folder (str): The path to the folder containing the flat leaves predictions data.
        num_workers (int, optional): The number of workers to use for data loading. Default is 16.

    Returns:
        DataLoader: A DataLoader object that loads the flat leaves prediction data in batches.
    """

    data = FlatLeavesPredictionsMerger(root_folder)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)

    return dataloader


def canopy_predictions_iterator(root_folder: str, num_workers: int = 16):
    """
    Creates a DataLoader for iterating over canopy prediction data.

    This function prepares a DataLoader to iterate over canopy prediction data from the specified
    root folder, utilizing the `CanopyPredictionsMerger` class to handle data merging and loading.

    Args:
        root_folder (str): The path to the folder containing the canopy predictions data.
        num_workers (int, optional): The number of workers to use for data loading. Default is 16.

    Returns:
        DataLoader: A DataLoader object that loads the canopy prediction data in batches.
    """

    data = CanopyPredictionsMerger(root_folder)
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=num_workers)

    return dataloader


def canopy_benchmark_iterator(root_folder: str, num_workers: int = 16, focus_src_override: str = None):
    """
    Creates a DataLoader for iterating over canopy benchmark data.

    This function prepares a DataLoader to iterate over canopy benchmark data, leveraging the 
    `CanopyBenchmarkMerger` class. It is used for tasks that involve benchmarking predictions against 
    ground truth data. The optional `focus_src_override` argument allows overriding the focus data source.

    Args:
        root_folder (str): The path to the folder containing the canopy benchmark data.
        num_workers (int, optional): The number of workers to use for data loading. Default is 16.
        focus_src_override (str, optional): An optional path to override the focus data source.

    Returns:
        DataLoader: A DataLoader object that loads the canopy benchmark data in batches.
    """

    data = CanopyBenchmarkMerger(root_folder, focus_override=focus_src_override)
    dataloader = DataLoader(data,  batch_size=1, shuffle=False, num_workers=num_workers)

    return dataloader


def canopy_evaluation_wrapper(root_folder: str = 'export', 
                              results_path: str = 'canopy_results.csv', 
                              num_workers: int = 16):
    """
    Wrapper for evaluating canopy predictions and computing metrics.

    This function wraps the entire process of evaluating canopy predictions, including loading the 
    prediction data, computing evaluation metrics using a `CanopyEvaluator`, and saving the results 
    in a CSV file. The computation is parallelized using a thread pool to improve performance.

    Args:
        root_folder (str, optional): The path to the folder containing the canopy prediction data.
            Default is 'export'.
        results_path (str, optional): The path to the CSV file where results will be saved.
            Default is 'canopy_results.csv'.
        num_workers (int, optional): The number of workers to use for data loading. Default is 16.
    """

    data = canopy_predictions_iterator(root_folder, num_workers=max(1, int(num_workers/4)))
    evaluator = CanopyEvaluator(results_path)

    def process_file(evaluator, filepath, results):
        evaluator.predict(filepath, results)

    logging.info("evaluating data:")
    futures = []

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers/4*3))) as executor:
        logging.info("collecting all data and computing metrics:")
        for (filepath, results) in tqdm(data, total=len(data)):
            futures.append(executor.submit(process_file, evaluator, filepath, results))

        logging.info("finishing computing metrics:")
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

    logging.info("done")


def flat_leaves_evaluation_wrapper(root_folder: str = 'export', 
                                   results_path: str = 'flat_leaves_results.csv', 
                                   num_workers: int = 16):
    """
    Wrapper for evaluating flat leaves predictions and computing metrics.

    This function wraps the entire process of evaluating flat leaves predictions, including loading 
    the prediction data, computing evaluation metrics using a `FlatLeavesEvaluator`, and saving the 
    results in a CSV file. It also uses parallel computation for efficient processing.

    Args:
        root_folder (str, optional): The path to the folder containing the flat leaves prediction data.
            Default is 'export'.
        results_path (str, optional): The path to the CSV file where results will be saved.
            Default is 'flat_leaves_results.csv'.
        num_workers (int, optional): The number of workers to use for data loading. Default is 16.
    """

    data = flat_leaves_predictions_iterator(root_folder, num_workers=max(1, int(num_workers/4)))
    evaluator = FlatLeavesEvaluator(results_path)

    def process_file(evaluator, filepath, results):
        evaluator.predict(filepath, results)

    logging.info("evaluating data:")
    futures = []

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers/4*3))) as executor:
        logging.info("collecting all data and computing metrics:")
        for (filepath, results) in tqdm(data, total=len(data)):
            futures.append(executor.submit(process_file, evaluator, filepath, results))
        
        logging.info("finishing computing metrics:")
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()
   
    logging.info("done")


def focus_evaluation_wrapper(root_folder: str = 'benchmark', 
                             results_path: str = 'benchmark.csv', 
                             focus_src_override: str = None):
    """
    Wrapper for evaluating focus data in canopy benchmark tasks.

    This function wraps the process of evaluating focus-specific data by loading the data and 
    computing metrics with the `CanopyBenchmarkEvaluator`. It then logs the results in the specified CSV file.

    Args:
        root_folder (str, optional): The path to the folder containing the benchmark focus data.
            Default is 'benchmark'.
        results_path (str, optional): The path to the CSV file where results will be saved.
            Default is 'benchmark.csv'.
        focus_src_override (str, optional): An optional path to override the focus data source.
    """

    data  = canopy_benchmark_iterator(root_folder, focus_src_override=focus_src_override)
    evaluator = CanopyBenchmarkEvaluator(results_path)

    logging.info("evaluating data:")
    for (filepath, results) in tqdm(data):
        evaluator.predict(filepath, results)
    logging.info("done")


def test() -> None:
    """
    Test function for running the canopy evaluation.

    This function configures logging and then triggers the canopy evaluation 
    process on a specified dataset by calling the `canopy_evaluation_wrapper` function.

    Returns:
        None
    """

    logging.basicConfig(level=logging.INFO)

    canopy_evaluation_wrapper(root_folder='test/export', results_path='test/canopy_test.csv')


if __name__=="__main__":

    test()
