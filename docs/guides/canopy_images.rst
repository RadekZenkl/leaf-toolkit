Canopy Images
-----

This guide provides shows you how to evaluate images of canopy following the method of `Zenkl et al. 2025b <TBD>`_, visualize results and compute disease metrics.

First make sure that all your images are exported to a common image format (.jpg, .png, etc.), they are named uniquely,
and cropped to a consistent resolution. For images acquired according to `Zenkl et al. 2025b <TBD>`_ we recommend  
`6144 x 4096 px` or `6144 x 4096 px` depending on the orientation of the camera.

Then you can use an existing configuration and predict simply a complete folder. You can either use a flat or 
nested folder structure.

Using a flat folder structure for export:

.. code-block:: Python

    from leaf.inference import Predictor 

    # 'canopy_portrait' uses 6144 x 4096 px images, for 6144 x 4096 px use 'canopy_landscape'
    pred = Predictor(config_name='canopy_portrait')  
    pred.predict(images_src='data/images', export_dst='export')


This code snippet produces all necessary predictions in the ``export`` folder for evaluating images of canopy. If you want to make visualizations
for manual inspection of the images execute the following:

.. code-block:: Python

    from leaf.visualization import CanopyVisualizer

    vis = CanopyVisualizer(
    src_root='export', 
    rgb_root='data/images', 
    export_root='export',
    )
    vis.visualize()

Now you should see new subfolders in the ``export`` directory with rendered visualizations for each individual results and for
the combined predictions. 

.. note::
    
    Visulization of predictions currently takes significantly longer than the actual inference. Consider visualizing 
    only a subset of the data.

Now you should have the following structure:

::

   export/
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
   ├── visualization_symptoms/
   └── visualization_combined/

Regardless of the visualization step you can compute the relevant disease metrics. To compute the relevant disease metrics per image
execute the following snippet

.. code-block:: Python

    from leaf.metrics import canopy_evaluation_wrapper

    canopy_evaluation_wrapper(root_folder='export', results_path='export/canopy_results.csv')