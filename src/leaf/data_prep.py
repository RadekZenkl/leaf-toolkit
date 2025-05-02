import cv2
import matplotlib.pyplot as plt
from pyzbar import pyzbar
from pathlib import Path
from matplotlib.patches import Rectangle
import mplcursors
from matplotlib.widgets import Button, TextBox
import time
from tqdm import tqdm
from typing import List, Dict, Tuple
import numpy as np
import logging
from pyzbar.pyzbar import ZBarSymbol


N_QR_CODES: int = 8


def determine_parameters(img: np.array) -> dict[str: int]:
    """
    Determine cropping parameters based on image resolution.

    Supports A4 scanned at 600dpi and 1200dpi. Raises an error if the resolution
    is unsupported.

    Args:
        img (np.array): Input image as a NumPy array.

    Returns:
        dict[str, int]: Dictionary with keys 'cropping_size', 'scale_factor',
        'qr_offset_x', and 'qr_offset_y'.

    Raises:
        Exception: If image resolution does not match predefined formats.
    """

    # Determine QR code and cropping parameters based on the image shape

    # This corresponds to A4 scanned at 1200dpi
    if np.abs(img.shape[0] - 17_600) < 2000 and np.abs(img.shape[1] - 12_250) < 2000:
        params = {
            'cropping_size': (1024, 6144),
            'scale_factor': 1/4,
            'qr_offset_x': 2000,
            'qr_offset_y': 600,
        }

    # TODO have a look at what this is
    # This corresponds to A4 scanned at 1200dpi
    if np.abs(img.shape[0] - 17_600) < 4000 and np.abs(img.shape[1] - 12_250) < 4000:
        params = {
            'cropping_size': (1024, 6144),
            'scale_factor': 1/4,
            'qr_offset_x': 2000,
            'qr_offset_y': 500,
        }

    # This corresponds to A4 scanned at 600dpi
    elif np.abs(img.shape[0] - 7000) < 1000 and np.abs(img.shape[1] - 5000) < 1000:
        params = {
            'cropping_size': (512, 3072),
            'scale_factor': 1/3,
            'qr_offset_x': 800,
            'qr_offset_y': 275,
        }
    # If the options from above do not cover your usecase, implement a new option here
    else:
        raise Exception("The provided image has a resolution of {} x {} px. No fitting cropping profile found. Please check your data or update the cropping profiles.")

    return params

def is_hidden_file(file_path: str) -> bool:
    """
    Check if a file is hidden (Unix-style or Windows).

    Args:
        file_path (str): Full path to the file.

    Returns:
        bool: True if the file is hidden, False otherwise.
    """

    # Unix systems
    if Path(file_path).name.startswith('.'):
        return True
    
    # windows
    try:
        attrs = Path(file_path).stat().st_file_attributes
        # Check if the hidden attribute is set (bitwise AND with 2)
        if attrs & 2 != 0:
            return True
    except AttributeError:
        pass

    return False


def check_file(file_path: str) -> bool:
    """
    Validate if the file exists, is not hidden, and is a readable image.

    Args:
        file_path (str): Full path to the file.

    Returns:
        bool: True if valid, False otherwise.
    """
    # This is true when file is ok and false when there are some issues

    # Check if file is a folder
    if not Path(file_path).is_file():
        logging.warning("Path: {} is not a file.  Skipping".format(str(file_path)))
        return False

    # check if file is a hidden file
    if is_hidden_file(file_path):
        logging.warning("File: {} seems to be hidden. Skipping".format(file_path))
        return False

    # check if file can be read by opencv
    if cv2.imread(file_path) is None:
        logging.warning("File: {} seems not to be an image, or is corrupted. Skipping".format(file_path))
        return False

    return True


def prepare_folder(folder_path: str, export_path: str, error_logs_path: str, manual_correction: bool = False, debug: bool = False, correction_export_path: str = None) -> None:
    """
    Process a folder of images for QR detection and cropping.

    Optionally allows manual correction of failed detections.

    Args:
        folder_path (str): Path to the folder containing input images.
        export_path (str): Path to export cropped image patches.
        error_logs_path (str): Path to write error logs.
        manual_correction (bool, optional): If True, enables manual correction. Defaults to False.
        debug (bool, optional): If True, enables debug output. Defaults to False.
        correction_export_path (str, optional): Path for corrected exports if manual correction is enabled.
    """
    # convert strings to pathlib paths
    folder_path = Path(folder_path)
    export_path = Path(export_path)
    error_logs_path = Path(error_logs_path)

    errors: List[str] = []
    # List all images in the folder
    paths: List[Path] = [i for i in folder_path.rglob("*")]
    for path in tqdm(paths):
        # check if file is relevant and can be processed
        if not check_file(str(path)):
            continue

        # Iterate image per image
        qr_codes_info: List[Dict[str, object]] = find_qr_codes(str(path), debug)
        crop_qr_code_patches(str(path), qr_codes_info, export_path, debug)

        # Check if all QR codes have been detected in the image
        if len(qr_codes_info) != N_QR_CODES:
            errors.append(str(path))

    # Log images which have not all QR codes detected
    logging.warning("In {} images, not all QR codes have been detected".format(len(errors)))
    with open(str(error_logs_path), 'w') as fp:
        for error in errors:
            fp.write(error+'\n')

    if manual_correction:
        # Correct all the missing QR codes and select the cropping location
        if correction_export_path is None:
            raise Exception("Please provide a path for exporting manual corrections")
        correction(error_logs_path, Path(correction_export_path))


def correction(error_logs_path: Path, correction_export_path: Path) -> None:
    """
    Launch an interactive tool for manual correction of image crops.

    Args:
        error_logs_path (Path): Path to the file listing failed detections.
        correction_export_path (Path): Path to export manually cropped patches.
    """
    errorfiles: List[str] = []
    with open(str(error_logs_path), 'r') as fp:
        errorfiles = fp.readlines()

    if len(errorfiles) > 0:
        for errorfile in tqdm(errorfiles):

            image_path: Path = Path(errorfile)
            # Remove potential new lines
            image_path = str(image_path).strip()

            image: np.ndarray = cv2.imread(image_path)
            cropping_params = determine_parameters(image)

            # Have an interactive session in matplotlib to set the bounding box

            # Find QR codes in the image
            qr_codes: List[Dict[str, object]] = find_qr_codes(image_path)
            # Plot the image with BoundingBoxSelector
            bb_selector: BoundingBoxSelector = BoundingBoxSelector(image_path, qr_codes, cropping_params['cropping_size'], cropping_params['scale_factor'], correction_export_path)

            for qr_code in qr_codes:
                (x, y, w, h) = qr_code['coordinates']
                box: Rectangle = Rectangle((int(x * cropping_params['scale_factor']), int(y * cropping_params['scale_factor'])),
                                int(w * cropping_params['scale_factor']),
                                int(h * cropping_params['scale_factor']),
                                linewidth=2, edgecolor='b', facecolor='none')
                bb_selector.ax.add_patch(box)

            bb_selector.show()


def find_qr_codes(image_path: str, debug: bool = False) -> List[Dict[str, object]]:
    """
    Detect QR codes in an image and return their info.

    Args:
        image_path (str): Path to the image file.
        debug (bool, optional): If True, shows debug visualization. Defaults to False.

    Returns:
        List[Dict[str, object]]: List of QR code dictionaries with 'coordinates' and 'value'.
    """
    # Load the original image
    image: np.ndarray = cv2.imread(image_path)

    cropping_params = determine_parameters(image)
    scale_factor = cropping_params['scale_factor']

    # Calculate the downscaled size based on the scale factor
    downscaled_size: Tuple[int, int] = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))

    # Downscale the image based on the calculated size
    downscaled_image: np.ndarray = cv2.resize(image, downscaled_size, interpolation=cv2.INTER_LINEAR)

    # Convert the downscaled image to grayscale
    gray: np.ndarray = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2GRAY)

    # Find QR codes in the downscaled image
    qr_codes: List[pyzbar.ZBarSymbol] = pyzbar.decode(gray, symbols=[ZBarSymbol.QRCODE])

    # List to store QR code coordinates and values
    qr_code_info: List[Dict[str, object]] = []

    # Process each QR code
    for qr_code in qr_codes:
        # Extract the bounding box coordinates of the QR code
        (x, y, w, h) = qr_code.rect

        # Scale the coordinates back to the original image size
        x = int(x / scale_factor)
        y = int(y / scale_factor)
        w = int(w / scale_factor)
        h = int(h / scale_factor)

        # Extract the QR code data
        qr_data: str = qr_code.data.decode("utf-8")

        # Append the QR code coordinates and value to the list
        qr_code_info.append({
            'coordinates': (x, y, w, h),
            'value': qr_data
        })

        if debug:
            # Draw a rectangle around the QR code in the original image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 100)
            cv2.putText(image, qr_data, (x + 50, y - 150), cv2.FONT_HERSHEY_SIMPLEX, 8, (255, 0, 0), 50)

    if debug:
        # Display the original image with QR codes detected
        image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
        plt.title(label="Detected QR Codes")
        plt.axis('off')
        plt.show()
        plt.clf()
        plt.close("all")

    return qr_code_info


def crop_qr_code_patches(image_path: str, qr_codes_info: List[Dict[str, object]], save_dir: Path, debug: bool = False) -> None:
    """
    Crop patches around detected QR codes and save them.

    Args:
        image_path (str): Path to the input image.
        qr_codes_info (List[Dict[str, object]]): List of QR code metadata with coordinates and values.
        save_dir (Path): Directory to save cropped patches.
        debug (bool, optional): If True, shows cropped patches. Defaults to False.
    """
    # Create a directory to save the cropped patches
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the original image
    image: np.ndarray = cv2.imread(image_path)

    cropping_params = determine_parameters(image)

    # Process each QR code
    for qr_code in qr_codes_info:
        # Extract the QR code value
        qr_value: str = qr_code['value']

        # Extract the QR code coordinates
        (x, y, w, h) = qr_code['coordinates']

        # Apply the offset to adjust the patch size
        x = cropping_params['qr_offset_x']
        y += cropping_params['qr_offset_y']

        w = cropping_params['cropping_size'][1]
        h = cropping_params['cropping_size'][0]

        # Crop the image patch around the QR code
        qr_patch: np.ndarray = image[y:y + h, x:x + w]

        # Generate the filename using the QR code value
        filename: str = str(save_dir / (qr_value + ".png"))

        # Strip special characters
        filename = filename.replace('"', '')
        filename = filename.replace("'", "")

        # Save the image patch as a PNG file
        if not debug:
            cv2.imwrite(filename, qr_patch)
            logging.info(f"QR Code: {qr_value}, Patch saved as: {filename}")
        if debug:
            plt.imshow(cv2.cvtColor(qr_patch, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("resulting image crop")
            plt.show()
            plt.clf()
            plt.close("all")


class BoundingBoxSelector:
    """
    GUI tool for manually selecting and exporting image patches based on bounding boxes.
    """
    def __init__(self, image_path: str, qr_codes: List[Dict[str, object]], box_size: Tuple[int, int], scale_factor: float, export_dir: Path):
        """
        Initialize the bounding box selector.

        Args:
            image_path (str): Path to the image being processed.
            qr_codes (List[Dict[str, object]]): List of detected QR codes.
            box_size (Tuple[int, int]): Width and height of the crop box.
            scale_factor (float): Scale factor for display.
            export_dir (Path): Directory to export cropped patches.
        """
        self.original_image: np.ndarray = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        downscaled_size: Tuple[int, int] = (int(self.original_image.shape[1] * scale_factor), int(self.original_image.shape[0] * scale_factor))
        self.image: np.ndarray = cv2.resize(self.original_image, downscaled_size, interpolation=cv2.INTER_LINEAR)

        self.fig, self.ax = plt.subplots()
        self.fig.suptitle('Manual Correction of QR Codes', fontsize=16)
        self.ax.imshow(self.image)
        self.ax.axis('off')
        self.orig_box_size: Tuple[int, int] = box_size
        self.box_size: Tuple[int, int] = (int(box_size[1] * scale_factor), int(box_size[0] * scale_factor))
        self.user_boxes = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.qr_codes = qr_codes
        self.scale_factor = scale_factor
        self.selected_box = None

        # Create directory if not present
        self.export_dir = export_dir
        export_dir.mkdir(parents=True, exist_ok=True)

        # Buttons
        self.reset_button = Button(plt.axes([0.75, 0.15, 0.1, 0.04]), 'Reset', color='g', hovercolor='0.975')
        self.reset_button.on_clicked(self.reset)

        self.confirm_button = Button(plt.axes([0.75, 0.20, 0.1, 0.04]), 'Confirm', color='g', hovercolor='0.975')
        self.confirm_button.on_clicked(self.confirm)

        self.done_button = Button(plt.axes([0.75, 0.25, 0.1, 0.04]), 'Done / Next image', color='g', hovercolor='0.975')
        self.done_button.on_clicked(self.done)

        initial_text = "placeholder.png"
        self.text_box = TextBox(plt.axes([0.15, 0.025, 0.7, 0.05]), 'Filename', initial=initial_text)
        self.text_box.on_submit(self.submit)
        self.filename = None

    def on_press(self, event):
        """
        Handle mouse click event to define a bounding box.
        """
        if event.inaxes != self.ax:
            return

        x = int(event.xdata)
        y = int(event.ydata)

        self.reset(0)

        self.selected_box = Rectangle((x, y), self.box_size[0],
                                       self.box_size[1], linewidth=2, edgecolor='r', facecolor='none')
        self.ax.add_patch(self.selected_box)

        self.user_boxes = ((int(y/self.scale_factor), int(x/self.scale_factor)))
        self.fig.canvas.draw()

    def show(self):
        """
        Display the interactive selector interface.
        """
        mplcursors.cursor(self.ax, hover=False)
        plt.show()

    def reset(self, event):
        """
        Clear the currently drawn bounding box.
        """
        if self.selected_box is not None:
            self.selected_box.remove()
            self.selected_box = None

    def confirm(self, event):
        """
        Confirm the bounding box and trigger patch save.
        """

        if self.user_boxes is None:
            raise Exception("No coordinates present")
        if self.filename is None:
            raise Exception("No name given")

        image_patch = self.original_image[self.user_boxes[0]:self.user_boxes[0]+self.orig_box_size[0],
                                          self.user_boxes[1]:self.user_boxes[1]+self.orig_box_size[1]]
        image_path = self.export_dir / self.filename

        logging.info("Saving Patch: {}".format(self.filename))
        time.sleep(0.5)

        # Check if writing successful
        cv2.imwrite(str(image_path), cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))

        # Add confirmed selections
        confirmed_box = Rectangle((self.user_boxes[1]*self.scale_factor,
                                   self.user_boxes[0]*self.scale_factor),
                                  self.box_size[0],
                                  self.box_size[1], linewidth=2, edgecolor='g', facecolor='none')
        self.ax.add_patch(confirmed_box)

    def done(self, event):
        """
        Close the interface after completion.
        """
        plt.close(self.fig)

    def submit(self, text):
        """
        Handle filename input for saved patch.

        Args:
            text (str): Filename to save the cropped patch as.
        """
        self.filename = text
