import numpy as np
from pathlib import Path
import cv2
# from numpy.core.multiarray import array as array
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging

class BaseVisMap:
    """
    Base class for visualization mapping.
    Defines placeholders for name, accepted/rejected IDs and their associated colors.
    """
    NAME = 'name_placeholder'
    ACCEPTED_ID = None
    ACCEPTED_COLOR = [None, None, None, None]
    REJECTED_ID = None
    REJECTED_COLOR = [None, None, None, None]


class VisualizationMapping:
    # BGRA format for colors to align with opencv
    # rejected ids are the same ids subsctracted from 8bit (i.e. 255)
    # TODO use the formula to set the id instead of hardcoding them
    
    class Symptoms_det:
        class Pycnidia(BaseVisMap):
            NAME = 'pycnidia'
            ACCEPTED_ID = 1
            ACCEPTED_COLOR = [255, 246, 0]  # CYAN
            REJECTED_ID = 254
            REJECTED_COLOR = [0, 4, 255]  # red, oposite of CYAN on wheel

        class Rust(BaseVisMap):
            NAME = 'rust'
            ACCEPTED_ID = 2
            ACCEPTED_COLOR = [255, 132, 0]  # Petrol
            REJECTED_ID = 253
            REJECTED_COLOR = [0, 112, 255]  # red-orange

    class Symptoms_seg:
        # This is currently still with leaf class prediction
        class LeafDamage(BaseVisMap):
            NAME = 'necrosis'
            ACCEPTED_ID = 2
            ACCEPTED_COLOR = [255, 0, 88]  # Purple
            REJECTED_ID = 253
            REJECTED_COLOR = [0, 255, 173]  # Lime

        class InsectDamage(BaseVisMap):
            NAME = 'insect_damage'
            ACCEPTED_ID = 3
            ACCEPTED_COLOR = [255, 0, 0]  # blue
            REJECTED_ID = 252
            REJECTED_COLOR = [0, 255, 250]  # yellow
        
        class PowderyMildew(BaseVisMap):
            NAME = 'powdery_mildew'
            ACCEPTED_ID = 4
            ACCEPTED_COLOR = [151, 255, 0]  # Mint
            REJECTED_ID = 251
            REJECTED_COLOR = [113, 0, 255]  # fuchsia

        class Background(BaseVisMap):
            NAME = 'background'
            ACCEPTED_ID = 0
            ACCEPTED_COLOR = [60, 20, 200]  # Crimson
            REJECTED_ID = 255
            REJECTED_COLOR = [80, 127, 255]  # Coral

    # TODO potentially extend if multuple thresholds will be used (eg. one for pycnidia and one for PLACL)
    class Focus:
        class Sharp(BaseVisMap):
            NAME = 'sharp'
            ACCEPTED_ID = 1
            ACCEPTED_COLOR = None # should be transparent
        
        class Blurry(BaseVisMap):
            NAME = 'blurry'
            ACCEPTED_ID = 0
            # ACCEPTED_COLOR = [248, 0, 255]  # pink
            ACCEPTED_COLOR = [105, 105, 105]  # dark gray


    class Organs:
        class Head(BaseVisMap):
            NAME = 'wheat_head'
            ACCEPTED_ID = 1
            ACCEPTED_COLOR = [154, 255, 0]  # Spring Green
            REJECTED_ID = 254
            REJECTED_COLOR = [60, 20, 220]  # Crimson

        class Stem(BaseVisMap):
            NAME = 'stem'
            ACCEPTED_ID = 2
            ACCEPTED_COLOR = [255, 191, 0]  # deep sky blue
            REJECTED_ID = 253
            REJECTED_COLOR = [96, 164, 244]  # sandy brown


class Visualizer:
    """
    General Visualization class capable of collecting and visualizing various predictions
    """
    def __init__(
            self,
            vis_all: bool = True,
            vis_symptoms: bool = True,
            visualize_acceptance: bool = True,
            vis_organs: bool = True,
            vis_focus: bool = True,
            vis_symptoms_det: bool = True,
            vis_symptoms_seg: bool = True,
            vis_background: bool = False,
            src_root: str = 'export',
            rgb_root: str = 'images',
            export_root: str = 'export/visualizations',
            organs_subfolder: str = 'organs/pred',
            focus_subfolder: str = 'focus/pred',
            symptoms_det_subfolder: str = 'symptoms_det/pred',
            symptoms_seg_subfolder: str = 'symptoms_seg/pred',
        ):
        """
        Initializes the visualizer with paths and flags to control what types of data are visualized.

        Args:
            vis_all (bool): If True, visualize all available predictions.
            vis_symptoms (bool): If True, symptom segmentation and symptom detection will be combined.
            visualize_acceptance (bool): Whether to distinguish between accepted and rejected predictions.
            vis_organs (bool): Visualize organ segmentation predictions.
            vis_focus (bool): Visualize image focus predictions.
            vis_symptoms_det (bool): Visualize detected symptoms.
            vis_symptoms_seg (bool): Visualize segmented symptoms.
            vis_background (bool): Visualize segmented symptom background.
            src_root (str): Root directory containing prediction outputs.
            rgb_root (str): Directory containing original RGB images.
            export_root (str): Directory where visualizations will be saved.
            organs_subfolder (str): Path to organ predictions.
            focus_subfolder (str): Path to focus predictions.
            symptoms_det_subfolder (str): Path to symptom detection predictions.
            symptoms_seg_subfolder (str): Path to symptom segmentation predictions.
        """
        
        self.vis_all = vis_all
        self.vis_symptoms = vis_symptoms
        self.visualize_acceptance = visualize_acceptance
        self.vis_organs = vis_organs
        self.vis_focus = vis_focus
        self.vis_symptoms_det = vis_symptoms_det
        self.vis_symptoms_seg = vis_symptoms_seg
        self.visualize_background = vis_background

        self.src_root = src_root
        self.rgb_root = rgb_root
        self.export_root = export_root
        self.organs_subfolder = organs_subfolder
        self.focus_subfolder = focus_subfolder
        self.symptoms_det_subfolder = symptoms_det_subfolder
        self.symptoms_seg_subfolder = symptoms_seg_subfolder

    def map_data(self) -> list[dict]:
        """
        Maps prediction files and RGB images to a structured dictionary.

        Returns:
            list[dict]: A list of dictionaries where each dictionary contains paths to relevant data types for one image.

        Raises:
            Exception: If there's a mismatch in file counts or insufficient inputs.
        """

        # gather all data as denoted in the constructor
        data_container = {}

        # collect rgb images, they are always required
        rgb = self.find_images(self.rgb_root)
        data_container.update({'rgb': rgb})

        if self.vis_all:
            # gather all data
            organs = self.find_images(Path(self.src_root)/self.organs_subfolder)
            focus = self.find_images(Path(self.src_root)/self.focus_subfolder)
            symptoms_det = self.find_images(Path(self.src_root)/self.symptoms_det_subfolder)
            symptoms_seg = self.find_images(Path(self.src_root)/self.symptoms_seg_subfolder)
            
            data_container.update({'organs': organs, 'focus': focus, 'symptoms_det': symptoms_det, 'symptoms_seg': symptoms_seg})

        else:
            if self.vis_symptoms:
                symptoms_det = self.find_images(Path(self.src_root)/self.symptoms_det_subfolder)
                data_container.update({'symptoms_det': symptoms_det})

                symptoms_seg = self.find_images(Path(self.src_root)/self.symptoms_seg_subfolder)
                data_container.update({'symptoms_seg': symptoms_seg})

            if self.vis_organs:
                organs = self.find_images(Path(self.src_root)/self.organs_subfolder)
                data_container.update({'organs': organs})

            if self.vis_focus:
                focus = self.find_images(Path(self.src_root)/self.focus_subfolder)
                data_container.update({'focus': focus})

            if self.vis_symptoms_det:
                symptoms_det = self.find_images(Path(self.src_root)/self.symptoms_det_subfolder)
                data_container.update({'symptoms_det': symptoms_det})

            if self.vis_symptoms_seg:
                symptoms_seg = self.find_images(Path(self.src_root)/self.symptoms_seg_subfolder)
                data_container.update({'symptoms_seg': symptoms_seg})

        # check if everything is present for the desired visualization scheme
        if len(data_container.keys()) < 2:
            raise Exception("Not enogh arguments provided for visualization")
        
        for i, (key, val) in enumerate(data_container.items()):
            if i == 0:
                length = len(val)
                querry_key = key
            else:
                if length != len(val):
                    raise Exception(f"different number of images for {key} and {querry_key}")

        # return an iterable which yields a dict of paths for each desired prediction type
        data_container = [dict(zip(data_container.keys(), values)) for values in zip(*data_container.values())]

        # check if all entries point to the same data sample
        for entry in data_container:
            filenames = {p.stem for p in entry.values()}

            if len(filenames) == 1:
                continue
            else:
                print("Filenames are different:", filenames)

        return data_container

    def find_images(self, search_root: str | Path, img_extensions: list = ['*.jpg', '*.jpeg', '*.JPG', '*.png']) -> list[Path]:
        """
        Recursively finds images under the specified root directory.

        Args:
            search_root (str | pathlib.Path): Directory to search.
            img_extensions (list): List of file extensions to include.

        Returns:
            list[pathlib.Path]: Sorted list of found image paths.
        """
        images = []
        for ext in img_extensions:
            images.extend(Path(search_root).rglob(ext))
        images = [path for path in images]  # convert to a list
        # images = sorted(images)
        images = sorted(images, key=lambda p: p.name)  # This sorts the list based on filenames instead of complete filepath
        return images

    def visualize(self) -> None:
        """
        Main method that triggers visualization of all enabled types.
        """

        # TODO handle the naming of the export subfolders in a better way

        data = self.map_data()

        logging.info(f"Visualizing: {len(data)} images")

        for data_set in tqdm(data):

            if self.vis_all:
                self.visualize_all(data_set)

            if self.vis_symptoms:
                self.visualize_symptoms(data_set)
            
            if self.vis_organs:
                img_bgr = self.visualize_organs(
                    self.read_image(data_set['organs'], grayscale=True),
                    self.read_image(data_set['rgb']), 
                    )
                
                self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'organs/vis')

            if self.vis_focus:
                img_bgr = self.visualize_focus(
                    self.read_image(data_set['focus'], grayscale=True),
                    self.read_image(data_set['rgb']),
                    )
                
                self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'focus/vis')

            if self.vis_symptoms_det:
                img_bgr = self.visualize_symptoms_det(
                    self.read_image(data_set['symptoms_det'], grayscale=True),
                    self.read_image(data_set['rgb']),
                    )
                
                self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'symptoms_det/vis')

            if self.vis_symptoms_seg:
                img_bgr = self.visualize_symptoms_seg(
                    self.read_image(data_set['symptoms_seg'], grayscale=True),
                    self.read_image(data_set['rgb']),
                    )
                
                self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'symptoms_seg/vis')

    def visualize_all(self, data_set: dict) -> None:
        """
        Creates a composite visualization with all predictions.

        Args:
            data_set (dict): A dictionary of file paths to predictions and RGB image.
        """

        predictions = {'img_bgr': self.read_image(data_set['rgb']),
                       'organs': self.read_image(data_set['organs'], grayscale=True),
                       'focus': self.read_image(data_set['focus'], grayscale=True),
                       'symptoms_det': self.read_image(data_set['symptoms_det'], grayscale=True),
                       'symptoms_seg': self.read_image(data_set['symptoms_seg'], grayscale=True),
                       }

        if self.visualize_acceptance:
            predictions = self.combine_predictions(predictions)

        img_bgr = predictions['img_bgr']
        reference_image = predictions['img_bgr'].copy()

        # img_bgr = self.visualize_focus(predictions['focus'], img_bgr)
        img_bgr = self.visualize_organs(predictions['organs'], img_bgr)
        img_bgr = self.visualize_symptoms_det(predictions['symptoms_det'], img_bgr)
        img_bgr = self.visualize_symptoms_seg(predictions['symptoms_seg'], img_bgr)
        img_bgr = self.visualize_focus(predictions['focus'], img_bgr, reference_image)

        # save the results
        self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'visualization_combined')

    def visualize_symptoms(self, data_set: dict) -> None:
        """
        Creates a composite visualization of symptoms detections and symptoms segmentation.

        Args:
            data_set (dict): A dictionary of file paths to predictions and RGB image.
        """

        predictions = {'img_bgr': self.read_image(data_set['rgb']),
                       'symptoms_det': self.read_image(data_set['symptoms_det'], grayscale=True),
                       'symptoms_seg': self.read_image(data_set['symptoms_seg'], grayscale=True),
                       }
        
        img_bgr = predictions['img_bgr']

        img_bgr = self.visualize_symptoms_det(predictions['symptoms_det'], img_bgr)
        img_bgr = self.visualize_symptoms_seg(predictions['symptoms_seg'], img_bgr)

        # save the results
        self.save_visualization(str(Path(data_set['rgb']).stem), img_bgr, 'visualization_symptoms')

    def read_image(self, path: str | Path, grayscale: bool = False, bgr: bool = True) -> np.array:
        """
        Loads an image from the given path.

        Args:
            path (str | pathlib.Path): Path to the image file.
            grayscale (bool): Load in grayscale if True.
            bgr (bool): Return in BGR (default OpenCV) if True, otherwise convert to RGB.

        Returns:
            np.array: Loaded image.

        Raises:
            FileNotFoundError: If the image could not be loaded.
        """
        path = str(path)  # Convert potential Pathlib Path to a pure string. Opencv cannot handle Pathlib paths directly

        if grayscale:
            # Load image in grayscale mode
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            # Load image in color mode
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            
            # OpenCV loads images in BGR by default, convert to RGB if needed
            if not bgr:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Check if the image was successfully loaded
        if image is None:
            raise FileNotFoundError(f"Image at path '{path}' could not be loaded.")

        return image

    def combine_predictions(self, data: dict) -> dict:
        """
        Combines predictions with logic to mark accepted/rejected predictions.

        Args:
            data (dict): Dictionary containing segmentation and detection arrays.

        Returns:
            dict: Modified dictionary with adjusted labels based on acceptance rules.
        """
        # Keep  ids of accepted symptoms, introduce new ids for rejected symptoms which will be
        # visualized with different color encoding
        focus = data['focus']
        organs = data['organs']
        symptoms_det = data['symptoms_det']
        symptoms_seg = data['symptoms_seg']

        accepted_symptoms_mask = np.bitwise_and(organs == 0, focus == 1) 
        accepted_organs_maks = focus == 1 

        symptoms_det_combined = np.zeros_like(symptoms_det)
        symptoms_det_combined[accepted_symptoms_mask] = symptoms_det[accepted_symptoms_mask]
        symptoms_det_combined[~ accepted_symptoms_mask] = 255 - symptoms_det[~ accepted_symptoms_mask]  # rejected ids are just same order but substracted from 8 bit

        symptoms_seg_combined = np.zeros_like(symptoms_seg)
        symptoms_seg_combined[accepted_symptoms_mask] = symptoms_seg[accepted_symptoms_mask]
        symptoms_seg_combined[~ accepted_symptoms_mask] = 255 - symptoms_seg[~ accepted_symptoms_mask]  # rejected ids are just same order but substracted from 8 bit

        organs_combined = np.zeros_like(organs)
        organs_combined[accepted_organs_maks] = organs[accepted_organs_maks]
        organs_combined[~ accepted_organs_maks] = 255 - organs[~ accepted_organs_maks]  # rejected ids are just same order but substracted from 8 bit

        combined_data = data
        combined_data['organs'] = organs_combined
        combined_data['symptoms_det'] = symptoms_det_combined
        combined_data['symptoms_seg'] = symptoms_seg_combined

        return combined_data
    
    def visualize_focus(self, predictions: np.array, img_bgr: np.array, reference_img: np.array = None) -> np.array:
        """
        Overlays focus predictions on the RGB image.

        Args:
            predictions (np.array): Focus prediction mask.
            img_bgr (np.array): BGR image to overlay predictions on.
            reference_img (np.array): Reference image for grayscale fallback.

        Returns:
            np.array: Modified BGR image with focus overlay.
        """

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if reference_img is None else cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

        # Convert single-channel grayscale to 3-channel grayscale
        img_gray = cv2.merge([gray, gray, gray])

        img_bgr[predictions == VisualizationMapping.Focus.Blurry.ACCEPTED_ID] = img_gray[predictions == VisualizationMapping.Focus.Blurry.ACCEPTED_ID]

        return img_bgr

    def visualize_organs(self, predictions: np.array, img_bgr: np.array) -> np.array:
        """
        Visualizes organ segmentation predictions.

        Args:
            predictions (np.array): Organ prediction mask.
            img_bgr (np.array): BGR image to overlay predictions on.

        Returns:
            np.array: Image with organ overlay.
        """

        color_mapping = {
            VisualizationMapping.Organs.Head.ACCEPTED_ID: VisualizationMapping.Organs.Head.ACCEPTED_COLOR,
            VisualizationMapping.Organs.Head.REJECTED_ID: VisualizationMapping.Organs.Head.REJECTED_COLOR,
            VisualizationMapping.Organs.Stem.ACCEPTED_ID: VisualizationMapping.Organs.Stem.ACCEPTED_COLOR,
            VisualizationMapping.Organs.Stem.REJECTED_ID: VisualizationMapping.Organs.Stem.REJECTED_COLOR,
        }

        img_bgr = self.visualize_segmentations(img_bgr, predictions, color_mapping)

        return img_bgr

    def visualize_symptoms_det(self, predictions: np.array, img_bgr: np.array) -> np.array:
        """
        Visualizes detected symptoms.

        Args:
            predictions (np.array): Detection mask.
            img_bgr (np.array): Base image to draw detections on.

        Returns:
            np.array: Image with symptom detections.
        """
        color_mapping = {
            VisualizationMapping.Symptoms_det.Pycnidia.ACCEPTED_ID: VisualizationMapping.Symptoms_det.Pycnidia.ACCEPTED_COLOR,
            VisualizationMapping.Symptoms_det.Pycnidia.REJECTED_ID: VisualizationMapping.Symptoms_det.Pycnidia.REJECTED_COLOR,
            VisualizationMapping.Symptoms_det.Rust.ACCEPTED_ID: VisualizationMapping.Symptoms_det.Rust.ACCEPTED_COLOR,
            VisualizationMapping.Symptoms_det.Rust.REJECTED_ID: VisualizationMapping.Symptoms_det.Rust.REJECTED_COLOR,
        }

        img_bgr = self.visualize_detections(img_bgr, predictions, color_mapping)
        return img_bgr

    def visualize_symptoms_seg(self, predictions: np.array, img_bgr: np.array) -> np.array:
        """
        Visualizes segmented symptoms.

        Args:
            predictions (np.array): Segmentation mask.
            img_bgr (np.array): Image to draw segmentations on.

        Returns:
            np.array: Image with symptom segmentation.
        """

        color_mapping = {
            VisualizationMapping.Symptoms_seg.LeafDamage.ACCEPTED_ID: VisualizationMapping.Symptoms_seg.LeafDamage.ACCEPTED_COLOR,
            VisualizationMapping.Symptoms_seg.LeafDamage.REJECTED_ID: VisualizationMapping.Symptoms_seg.LeafDamage.REJECTED_COLOR,
            VisualizationMapping.Symptoms_seg.InsectDamage.ACCEPTED_ID: VisualizationMapping.Symptoms_seg.InsectDamage.ACCEPTED_COLOR,
            VisualizationMapping.Symptoms_seg.InsectDamage.REJECTED_ID: VisualizationMapping.Symptoms_seg.InsectDamage.REJECTED_COLOR,
            VisualizationMapping.Symptoms_seg.PowderyMildew.ACCEPTED_ID: VisualizationMapping.Symptoms_seg.PowderyMildew.ACCEPTED_COLOR,
            VisualizationMapping.Symptoms_seg.PowderyMildew.REJECTED_ID: VisualizationMapping.Symptoms_seg.PowderyMildew.REJECTED_COLOR,
        }

        if self.visualize_background:
            color_mapping.update({
                VisualizationMapping.Symptoms_seg.Background.ACCEPTED_ID: VisualizationMapping.Symptoms_seg.Background.ACCEPTED_COLOR,
                VisualizationMapping.Symptoms_seg.Background.REJECTED_ID: VisualizationMapping.Symptoms_seg.Background.REJECTED_COLOR,
            })

        img_bgr = self.visualize_segmentations(img_bgr, predictions, color_mapping)

        return img_bgr

    def visualize_detections(self, img_bgr: np.array, detections: np.array, color_mapping: dict, radius: int = 5, add_id: bool = False) -> np.array:
        """
        Draws detection circles on the image.

        Args:
            img_bgr (np.array): Image to draw on.
            detections (np.array): Detection mask.
            color_mapping (dict): Mapping from class ID to color.
            radius (int): Radius of circles to draw.
            add_id (bool): Whether to annotate detections with IDs.

        Returns:
            np.array: Annotated image.
        """

        for class_id, colors in color_mapping.items():
            centroids = np.where(detections == class_id)
            y_coords, x_coords = centroids

            # Draw circles at the specified coordinates
            for j, (x, y) in enumerate(zip(y_coords, x_coords)):
                cv2.circle(img_bgr, (y, x), radius, colors, 1)

                if add_id:
                    # Add Point Id
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f'{j}'
                    org = (y + 10, x + 5)  # Adjusted position for text
                    font_scale = 0.5
                    line_thickness = 1

                    cv2.putText(img_bgr, text, org, font, font_scale, colors, line_thickness, cv2.LINE_AA)

        return img_bgr

    def visualize_segmentations(self, img_bgr: np.array, segmentations: np.array, color_mapping: dict, alpha: float = 0.65) -> np.array:
        """
        Overlays segmentation masks with transparency.

        Args:
            img_bgr (np.array): Base image.
            segmentations (np.array): Segmentation mask.
            color_mapping (dict): Mapping of class IDs to colors.
            alpha (float): Blending factor.

        Returns:
            np.array: Image with overlaid segmentations.
        """
        beta = (1.0 - alpha)

        for class_id, colors in color_mapping.items():

            # construct a mask of a specific class
            mask = (segmentations == class_id).astype(np.uint8)

            # create the contours for this class
            cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_bgr, cnts, -1, colors, 3)

            img_bgr[segmentations == class_id] = (alpha*(img_bgr)+beta*(np.stack((mask,) * 3, axis=-1)*colors))[segmentations == class_id]

        return img_bgr
    
    def save_visualization(self, filename: str, image: np.array, visualization_category: str) -> None:
        """
        Saves a visualization image to disk.

        Args:
            filename (str): Output filename (without extension).
            image (np.array): Image to save.
            visualization_category (str): Folder name under export_root for saving.
        """
        quality = 90
        # handle right export path
        export_path = Path(self.export_root) / visualization_category / f"{filename}.jpg"
        export_path.parent.mkdir(parents=True, exist_ok=True)

        # handle saving compression 
        success = cv2.imwrite(str(export_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if not success:
            print(f"Failed to save the image: {str(export_path)}")

class CanopyVisualizer(Visualizer):
    """
    A derived visualizer class overloading some default values to simplify the visualization of typical canopy scenario.
    """
    def __init__(self, vis_all: bool = True, 
                 vis_symptoms: bool = True, 
                 visualize_acceptance: bool = True, 
                 vis_organs: bool = True, 
                 vis_focus: bool = True, 
                 vis_symptoms_det: bool = True, 
                 vis_symptoms_seg: bool = True, 
                 vis_background: bool = False,
                 src_root: str = 'export', 
                 rgb_root: str = 'images', 
                 export_root: str = 'export/visualizations', 
                 organs_subfolder: str = 'organs/pred', 
                 focus_subfolder: str = 'focus/pred', 
                 symptoms_det_subfolder: str = 'symptoms_det/pred', 
                 symptoms_seg_subfolder: str = 'symptoms_seg/pred',
                 ):
        """
        Initializes the visualizer with paths and flags to control what types of data are visualized. Default values 
        are adjusted to cover the canopy visualization use case.

        Args:
            vis_all (bool): If True, visualize all available predictions.
            vis_symptoms (bool): If True, symptom segmentation and symptom detection will be combined.
            visualize_acceptance (bool): Whether to distinguish between accepted and rejected predictions.
            vis_organs (bool): Visualize organ segmentation predictions.
            vis_focus (bool): Visualize image focus predictions.
            vis_symptoms_det (bool): Visualize detected symptoms.
            vis_symptoms_seg (bool): Visualize segmented symptoms.
            src_root (str): Root directory containing prediction outputs.
            rgb_root (str): Directory containing original RGB images.
            export_root (str): Directory where visualizations will be saved.
            organs_subfolder (str): Path to organ predictions.
            focus_subfolder (str): Path to focus predictions.
            symptoms_det_subfolder (str): Path to symptom detection predictions.
            symptoms_seg_subfolder (str): Path to symptom segmentation predictions.
        """

        super().__init__(vis_all, 
                         vis_symptoms, 
                         visualize_acceptance, 
                         vis_organs, vis_focus, 
                         vis_symptoms_det, 
                         vis_symptoms_seg, 
                         vis_background, 
                         src_root, 
                         rgb_root, 
                         export_root, 
                         organs_subfolder, 
                         focus_subfolder, 
                         symptoms_det_subfolder, 
                         symptoms_seg_subfolder
                         )

class FlattenedVisualizer(Visualizer):
    """
    A derived visualizer class overloading some default values to simplify the visualization of typical flattened leaves scenario.
    """
    def __init__(self, 
                 vis_all: bool = False, 
                 vis_symptoms: bool = True, 
                 visualize_acceptance: bool = False, 
                 vis_organs: bool = False, 
                 vis_focus: bool = False, 
                 vis_symptoms_det: bool = True, 
                 vis_symptoms_seg: bool = True, 
                 vis_background: bool = True, 
                 src_root: str = 'export', 
                 rgb_root: str = 'images', 
                 export_root: str = 'export/visualizations', 
                 organs_subfolder: str = 'organs/pred', 
                 focus_subfolder: str = 'focus/pred', 
                 symptoms_det_subfolder: str = 'symptoms_det/pred', 
                 symptoms_seg_subfolder: str = 'symptoms_seg/pred',
                 ):
        """
        Initializes the visualizer with paths and flags to control what types of data are visualized. Default values 
        are adjusted to cover the canopy visualization use case.

        Args:
            vis_all (bool): If True, visualize all available predictions.
            vis_symptoms (bool): If True, symptom segmentation and symptom detection will be combined.
            visualize_acceptance (bool): Whether to distinguish between accepted and rejected predictions.
            vis_organs (bool): Visualize organ segmentation predictions.
            vis_focus (bool): Visualize image focus predictions.
            vis_symptoms_det (bool): Visualize detected symptoms.
            vis_symptoms_seg (bool): Visualize segmented symptoms.
            src_root (str): Root directory containing prediction outputs.
            rgb_root (str): Directory containing original RGB images.
            export_root (str): Directory where visualizations will be saved.
            organs_subfolder (str): Path to organ predictions.
            focus_subfolder (str): Path to focus predictions.
            symptoms_det_subfolder (str): Path to symptom detection predictions.
            symptoms_seg_subfolder (str): Path to symptom segmentation predictions.
        """

        super().__init__(vis_all, 
                         vis_symptoms, 
                         visualize_acceptance, 
                         vis_organs, 
                         vis_focus, 
                         vis_symptoms_det, 
                         vis_symptoms_seg, 
                         vis_background,
                         src_root, 
                         rgb_root, 
                         export_root, 
                         organs_subfolder, 
                         focus_subfolder, 
                         symptoms_det_subfolder, 
                         symptoms_seg_subfolder,
                         )

def save_image(path: str, image: np.array, color_convert: int = None) -> None:
    """
    Saves an image to disk with optional color conversion.

    Args:
        path (str): Destination file path.
        image (np.array): Image tensor (usually from torch) to save.
        color_convert (int): Optional OpenCV color conversion flag.
    """
    img_np = image.cpu().numpy()
    if color_convert:
        img_np = cv2.cvtColor(img_np, color_convert)
    cv2.imwrite(path, img_np)

def save_depth_overlay(normalized_image: np.array, mask: np.array, output_path: str) -> None:
    """
    Creates an overlay of a binary mask on a normalized depth image and saves it.

    Args:
        normalized_image (np.array): Normalized depth image.
        mask (np.array): Binary mask (1 = keep, 0 = mask).
        output_path (str): Path to save the overlay.
    """
    depth_mask = (1 - mask.cpu().numpy()).astype(bool)
    overlay = np.zeros((*normalized_image.shape, 4))
    overlay[depth_mask] = [1, 1, 1, 0.95]  # White with high transparency

    plt.figure(figsize=(6, 9), dpi=320)
    plt.imshow(normalized_image, cmap='Spectral_r')
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_histogram(data: np.array, mean_d: float, std_d: float, initial_mean: float, initial_std: float, output_path: str, highlight_line: float = None) -> None:
    """
    Plots and saves a histogram with statistical highlights.

    Args:
        data (np.array): Input data for histogram.
        mean_d (float): Mean of the filtered data.
        std_d (float): Std dev of the filtered data.
        initial_mean (float): Original dataset mean.
        initial_std (float): Original dataset std dev.
        output_path (str): Output path for the histogram image.
        highlight_line (float, optional): Vertical line to highlight a value.
    """
    plt.figure(figsize=(9, 9), dpi=320)
    plt.rcParams.update({'font.size': 18})

    plt.hist(data, bins=50, edgecolor='black', label='All Data', alpha=0.7)
    plt.axvline(mean_d, color='red', linestyle='--', label=f'Mean (Filtered): {mean_d:.2f}')
    plt.axvspan(mean_d - std_d, mean_d + std_d, color='gray', alpha=0.4, label='±1 Std (Filtered)')
    plt.axvspan(initial_mean - 3 * initial_std, initial_mean + 3 * initial_std, color='lightgray', alpha=0.3, label='±3 Std (Initial)', zorder=0)

    if highlight_line:
        plt.axvline(highlight_line, color='red', linestyle='--', label=f'Peak Bin: {highlight_line:.2f}')

    # Color bins based on position
    counts, bins, patches = plt.hist(data, bins=50, edgecolor='black')
    norm = mcolors.Normalize(vmin=min(bins), vmax=max(bins))
    cmap = plt.get_cmap('Spectral_r')
    for bin_edge, patch in zip(bins[:-1], patches):
        patch.set_facecolor(cmap(norm(bin_edge)))

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def test():
    """
    This function runs a dry run of the complete visualization to validate your installation. 
    It produces visualizations in newly created `test/visualization` folder. 
    It requires predictions of the models.test() to create the desired output.
    """

    logging.basicConfig(level=logging.INFO)

    pred_root = "test/export"
    rgb_root = "test/images"
    vis_root = "test/visualizations"

    logging.info(f"Visualizing contents of: {pred_root} with images from {rgb_root} and saving to {vis_root}")

    vis = Visualizer(src_root=pred_root, rgb_root=rgb_root, export_root=vis_root)
    vis.visualize()

    logging.info("done")

if __name__ == "__main__":
    test()