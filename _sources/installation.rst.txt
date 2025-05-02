Installation
=============


You can install the package either from the Python Package Index (PyPI) or directly from the source on GitHub. Please not that this package requires **Python >= 3.12**. 

Install from PyPI
-----------------

The easiest way to install the package is via `pip`:

.. code-block:: bash

    pip install leaf-toolkit

This will install the latest released version of the package from PyPI.

Install from Source (GitHub)
----------------------------

If you prefer install the latest development version directly from GitHub, clone the repository manually and install in editable/development mode:

.. code-block:: bash

    git clone https://github.com/RadekZenkl/leaf-toolkit.git
    cd leaf-toolkit
    pip install -e .

This allows you to make changes to the source code and have them immediately reflected without reinstalling.

Requirements
------------

Make sure you have **Python >= 3.12** and ``pip`` installed. 

If you are going to use ``data_prep`` module make sure that your environment has a sufficient version of ``gcc``. 
If you are using a conda environment execute: ``conda install -c conda-forge gcc``. 
If you are running directly on the system running: ``sudo apt install build-essential`` (or equivalent) should help.

Installation Validation for Inferece and Evaluation using GPU
------------
Run the following code block to check if everything is installed correctly.

.. code-block:: Python

    from leaf import models, visualization, metrics

    models.test()
    visualization.test()
    metrics.test()

This downloads a sample image into **test/images/...** and all the necessary model weights. 
Afterwards it predicts on the image. Please note that the test runs on a sample image which has 
resolution of **4096 x 4096 px**. We adjusted the patchsize so **<8Gb of VRAM** is required. The inference 
results are saved in **test/export/...**. The visualization test then utilizes the inference results
to produce visualizations in **test/visualizations/...**. Consequently, the test of metrics utilizes 
the inference results to produce a computed metrics file in **test/canopy_test.csv**.