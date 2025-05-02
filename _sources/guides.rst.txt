Guides
===============


Leaf Toolkit supports three scenarios for inference: 

#. Freeform canopy images `Zenkl et al. 2025b <TBD>`_ .
#. Flattened leaves as proposed by `Zenkl et al. 2025a <https://www.sciencedirect.com/science/article/pii/S0168169924012456>`_ and `Anderegg et al. 2024 <https://link.springer.com/article/10.1186/s13007-024-01290-4>`_ .
#. Scanned images using flatbed scanners as proposed by `Stewart et al. 2016 <https://apsjournals.apsnet.org/doi/full/10.1094/PHYTO-01-16-0018-R>`_ . 

In all scenarios the name of images has to be unique as it is used for identification.

Once installed, you can use **Leaf Toolkit** to perform various tasks. Below are guides for most common tasks:

.. :doc:`Canopy Images <guides/canopy_images>`
.. :doc:`Flattened Leaves <guides/flattened_leaves>`
.. :doc:`Flatbed Scanners <guides/flatbed>`

.. toctree::
   :maxdepth: 2
   :caption: Guides:


   guides/canopy_images
   guides/flattened_leaves
   guides/flatbed

General Inference
-----

The general inference is done as following:

.. code-block:: Python

    from leaf.inference import Predictor 

    pred = Predictor()
    pred.predict(images_src=<path to images to be predicted>, export_dst=<path to save the results>)

Per default this uses a configuration to predict on `6144 x 4096 px` canopy images with the optimized parameters. 
We provide 3 basic configurations which can be chosen:

-  `canopy_landscape`:  canopy images in landscape mode `4096 x 6144 px`
-  `canopy_portrait`:  canopy images in portrait mode `6144 x 4096 px`
-  `flattened`:  images of flattened leaves or flatbed scanner images `1024 x 6144 px`

The configuration can be changed when creating the `Predictor` object by passing the `config_name` argument, 
e.g. `Predictor(config_name='flattened')`. Furthermore, all parameters of individual models can be adjusted by 
passing an dictionary containing the corresponding parameters:

.. code-block:: Python

    pred = Predictor(
        symptoms_det_params={...}
        symptoms_seg_params={...}
        organs_params={...}
        focus_params={...}
        module_params={...}
    )

.. note::

   Inference on large images is very VRAM intensive. For example, running inference on a ``6144 x 4096 px`` image requires **24 GB** of VRAM. The required resources can be reduced by splitting the input into patches.

The most intensive parts of the pipeline are ``symptoms detection`` and ``symptoms segmentation``. Splitting of the input image can be controlled using the ``patch_sz`` argument (see above).

However, note that the current implementation only supports patch sizes that **exactly sum up to the image resolution** (e.g., ``1024 x 1024 px`` for a ``4096 x 6144 px`` image, but not ``1000 x 1000 px``).

All models besides ``focus estimation`` can handle arbitrary input sizes (multiples of at least 32). However, due to TorchScript export limitations, the ``DepthAnythingv2`` model only supports specific resolutions. A list of supported resolutions is available in the *Model Zoo*.

If you need an intermediate resolution, you can adjust the model's ``input_scaling`` argument to match one of the available models.

Visualization
-----

The visualization of predictions is significantly slower compared to inference. Therefore, it can be executed as a separate step. The visualizer is configured for **canopy images** by default.

.. code-block:: python

   from leaf.visualization import Visualizer

   vis = Visualizer(
       src_root=<path to the root of where predictions are saved>, 
       rgb_root=<path to rgb images used for prediction>, 
       export_root=<where to save visualizations>,
   )
   vis.visualize()

Per default, the visualizer attempts to visualize everything. When working with **flattened leaves** consider using the ``FlattenedVisualizer``, or disable ``focus`` and ``organs`` visualizations from the default setting. You can do this by setting the arguments:

``vis_all=False, vis_organs=False, vis_focus=False``

So the typical scenarios see the respective guides.

when creating the ``Visualizer`` object.

Prediction and Visualization Structure
-----

The raw results are saved in the form of image masks in `.png` format. Upon predicting and visualizing with the same export path, the following folder structure is created:

::

   <export path>/
   ├── focus/
   │   ├── pred/
   │   └── vis/
   ├── organs/
   │   ├── pred/
   │   └── vis/
   ├── symptoms_det/
   │   ├── pred/
   │   └── vis/
   ├── symptoms_seg/
   │   ├── pred/
   │   └── vis/
   └── visualization_combined/

Each ``pred`` folder contains class-encoded masks, and ``vis`` contains `.jpg` images with labels.  
Furthermore, ``visualization_combined`` contains merged predictions as used for computing metrics.

