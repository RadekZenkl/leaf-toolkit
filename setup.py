from setuptools import setup, find_packages

setup(
    name='leaf-toolkit',                # This will be the name of the package on PyPI
    version='0.1.0',                  # Replace with your desired version
    packages=find_packages(where='src'),         # Automatically find all packages in the directory
    package_dir={'': 'src'},
    author='Radek Zenkl',
    author_email='radek.zenkl@usys.ethz.ch',
    description='Leaf Analysis and Evaluation Toolkit',
    # long_description='A longer description if you have one',
    long_description_content_type='text/markdown',
    url='https://github.com/RadekZenkl/leaf-toolkit', 
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',          # Adjust to your package's compatibility requirements
    install_requires=[                # Add any dependencies your package needs
        "setuptools>=61.0", 
        "opencv-python", 
        "scikit-image", 
        "torch>=2.0.1",
        "torchvision",
        "matplotlib",
        "mpldatacursor",
        "mplcursors",
        "protobuf",
        "tqdm",
        "psutil"
    ],
)