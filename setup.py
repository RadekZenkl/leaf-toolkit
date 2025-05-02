from setuptools import setup, find_packages

setup(
    name='leaf-toolkit',
    version='0.4.0',                  # Replace with your desired version
    packages=find_packages(where='src'),     
    package_dir={'': 'src'},
    author='Radek Zenkl',
    author_email='radek.zenkl@usys.ethz.ch',
    description='Leaf Analysis and Evaluation Toolkit',
    long_description_content_type='text/markdown',
    url='https://github.com/RadekZenkl/leaf-toolkit',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.6',          # Adjust to your package's compatibility requirements
    install_requires=[                # Add any dependencies your package needs
        "setuptools>=64.0",
        "ultralytics>=8.3.107", 
        "opencv-python", 
        "scikit-image", 
        "torch>=2.0.1",
        "torchvision",
        "torchmetrics",
        "matplotlib",
        "mpldatacursor",
        "mplcursors",
        "protobuf",
        "tqdm",
        "psutil"
    ],
)