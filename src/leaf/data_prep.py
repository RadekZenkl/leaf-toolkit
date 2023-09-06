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
Offending key for IP in /home/radek/.ssh/known_hosts:4


CROPPING_SIZE: Tuple[int, int] = (1024, 8192)
SCALE_FACTOR: float = 1/4
N_QR_CODES: int = 8


def prepare_folder(folder_path: str, export_path: str, error_logs_path: str, manual_correction: bool = False, debug: bool = False) -> None:
    # convert strings to pathlib paths
    folder_path = Path(folder_path)
    export_path = Path(export_path)
    error_logs_path = Path(error_logs_path)

    errors: List[str] = []
    # List all images in the folder
    paths: List[Path] = [i for i in folder_path.rglob("*")]
    for path in tqdm(paths):
        # Iterate image per image
        qr_codes_info: List[Dict[str, object]] = find_qr_codes(str(path), SCALE_FACTOR, debug)
        crop_qr_code_patches(str(path), qr_codes_info, export_path, debug)

        # Check if all QR codes have been detected in the image
        if len(qr_codes_info) != N_QR_CODES:
            errors.append(str(path))

    # Log images which have not all QR codes detected
    print("In {} image, not all QR codes have been detected".format(len(errors)))
    with open(str(error_logs_path), 'w') as fp:
        for error in errors:
            fp.write(error+'\n')

    if manual_correction:
        # Correct all the missing QR codes and select the cropping location
        correction(error_logs_path)


def correction(error_logs_path: Path) -> None:
    errorfiles: List[str] = []
    with open(str(error_logs_path), 'r') as fp:
        errorfiles = fp.readlines()

    if len(errorfiles) > 0:
        for errorfile in tqdm(errorfiles):

            image_path: Path = Path(errorfile)
            # Remove potential new lines
            image_path = str(image_path).strip()

            # Have an interactive session in matplotlib to set the bounding box

            # Find QR codes in the image
            qr_codes: List[Dict[str, object]] = find_qr_codes(image_path, SCALE_FACTOR)
            # Plot the image with BoundingBoxSelector
            bb_selector: BoundingBoxSelector = BoundingBoxSelector(image_path, qr_codes, CROPPING_SIZE, SCALE_FACTOR, Path('export'))

            for qr_code in qr_codes:
                (x, y, w, h) = qr_code['coordinates']
                box: Rectangle = Rectangle((int(x * SCALE_FACTOR), int(y * SCALE_FACTOR)),
                                int(w * SCALE_FACTOR),
                                int(h * SCALE_FACTOR),
                                linewidth=2, edgecolor='b', facecolor='none')
                bb_selector.ax.add_patch(box)

            bb_selector.show()


def find_qr_codes(image_path: str, scale_factor: float, debug: bool = False) -> List[Dict[str, object]]:
    # Load the original image
    image: np.ndarray = cv2.imread(image_path)

    # Calculate the downscaled size based on the scale factor
    downscaled_size: Tuple[int, int] = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))

    # Downscale the image based on the calculated size
    downscaled_image: np.ndarray = cv2.resize(image, downscaled_size, interpolation=cv2.INTER_LINEAR)

    # Convert the downscaled image to grayscale
    gray: np.ndarray = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2GRAY)

    # Find QR codes in the downscaled image
    qr_codes: List[pyzbar.ZBarSymbol] = pyzbar.decode(gray)

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
    # Create a directory to save the cropped patches
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load the original image
    image: np.ndarray = cv2.imread(image_path)

    # Process each QR code
    for qr_code in qr_codes_info:
        # Extract the QR code value
        qr_value: str = qr_code['value']

        # Extract the QR code coordinates
        (x, y, w, h) = qr_code['coordinates']

        # Apply the offset to adjust the patch size
        x = 750
        y += 600
        w = CROPPING_SIZE[1]
        h = CROPPING_SIZE[0]

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
            print(f"QR Code: {qr_value}, Patch saved as: {filename}")
        if debug:
            plt.imshow(cv2.cvtColor(qr_patch, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("resulting image crop")
            plt.show()
            plt.clf()
            plt.close("all")


class BoundingBoxSelector:
    def __init__(self, image_path: str, qr_codes: List[Dict[str, object]], box_size: Tuple[int, int], scale_factor: float, export_dir: Path):

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

        self.done_button = Button(plt.axes([0.75, 0.25, 0.1, 0.04]), 'Done', color='g', hovercolor='0.975')
        self.done_button.on_clicked(self.done)

        initial_text = "placeholder.png"
        self.text_box = TextBox(plt.axes([0.15, 0.025, 0.7, 0.05]), 'Filename', initial=initial_text)
        self.text_box.on_submit(self.submit)
        self.filename = None

    def on_press(self, event):
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
        mplcursors.cursor(self.ax, hover=False)
        plt.show()

    def reset(self, event):
        if self.selected_box is not None:
            self.selected_box.remove()
            self.selected_box = None

    def confirm(self, event):

        if self.user_boxes is None:
            raise Exception("No coordinates present")
        if self.filename is None:
            raise Exception("No name given")

        image_patch = self.original_image[self.user_boxes[0]:self.user_boxes[0]+self.orig_box_size[0],
                                          self.user_boxes[1]:self.user_boxes[1]+self.orig_box_size[1]]
        image_path = self.export_dir / self.filename

        print("Saving Patch: {}".format(self.filename))
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
        plt.close(self.fig)

    def submit(self, text):
        self.filename = text


if __name__ == "__main__":
    src_path = 'data/images'
    export_path = 'export/individual_samples'
    error_logs_path = 'err_log.txt'
    prepare_folder(src_path, export_path, error_logs_path, manual_correction=True)
