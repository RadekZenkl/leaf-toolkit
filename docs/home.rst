
Home
=========================================


**Leaf Toolkit** is a powerful tool that allows you to quantify various foliar diseases on images. This documentation provides an overview of the system, installation instructions, tutorials, and API reference to help you get the most out of Leaf Toolkit.


Getting Started
--------------

**Leaf Toolkit** is designed to be easy to set up and use. To get started, you need to install it and follow a few simple steps.

The easiest way to install the package is via `pip`:

.. code-block:: bash

    pip install leaf-toolkit

This will install the latest released version of the package from PyPI.

For more detailed installation instructions, refer to the :doc:`installation <installation>`.


Once installed, you can use **Leaf Toolkit** to perform various tasks. Below is a simple usage example:

.. code-block:: python

   from leaf import models

   # Test model prediction
   models.test()

Refer to the :doc:`guides <guides>` page for more detailed instructions and examples.

API Reference
--------------

The following sections provide a detailed reference of all the public classes, functions, and methods available in **Leaf Toolkit**.

:doc:`API reference <autoapi/index>`


Contributing
--------------

We welcome contributions to **Leaf Toolkit**! 

🐛 Bug Reports
~~~~~~~~~~~~~~~~~~~~
- Use the issue tracker to report bugs.
- Include details about your environment (Python version, OS).
- Provide a minimal reproducible example if possible.

✨ Feature Requests
~~~~~~~~~~~~~~~~~~~~
- Open an issue describing the feature you’d like.
- Explain why it would be useful.
- If you want to implement it, let us know!

🔧 Code Contributions
~~~~~~~~~~~~~~~~~~~~
- Make sure your code follows **PEP 8** and project conventions.
- Write or update tests for your changes.
- Add or update documentation where needed.
- Submit a pull request with a clear description of your changes.

If you’d like to contribute, please fork the repository, make your changes, 
and submit a pull request.

Citation
--------------

If you use **Leaf Toolkit** please cite following works:

.. code-block:: bibtex

   @software{Leaf-Toolkit,
   author = {Zenkl, Radek and Anderegg, Jonas and McDonald, Bruce},
   license = {GPLv3},
   month = april,
   title = {{leaf-toolkit}},
   url = {https://github.com/RadekZenkl/leaf-toolkit},
   version = {1.0.0},
   year = {2025}
   }

.. code-block:: bibtex

   @article{zenkl2025towards,
     title={Towards high throughput in-field detection and quantification of wheat foliar diseases using deep learning},
     author={Zenkl, Radek and McDonald, Bruce A and Walter, Achim and Anderegg, Jonas},
     journal={Computers and Electronics in Agriculture},
     volume={232},
     pages={109854},
     year={2025},
     publisher={Elsevier}
   }