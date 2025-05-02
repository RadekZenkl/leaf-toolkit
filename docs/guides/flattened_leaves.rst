Flattened Leaves
-----

This guide provides shows you how to evaluate images of flattened leaves outdoors following the method of `Zenkl et al. 2025a <https://www.sciencedirect.com/science/article/pii/S0168169924012456>`_ and 
`Anderegg et al. 2024 <https://link.springer.com/article/10.1186/s13007-024-01290-4>`_, 
visualize results and compute disease metrics.

First make sure that all your images are exported to a common image format (.jpg, .png, etc.), they are named uniquely,
and cropped to a consistent resolution. For images acquired according to `Anderegg et al. 2024 <https://link.springer.com/article/10.1186/s13007-024-01290-4>`_ we recommend  
`2048 x 8192 px`.

Then you can use an existing configuration and predict simply a complete folder. You can either use a flat or 
nested folder structure.

Using a flat folder structure for export:

.. code-block:: Python

    from leaf.inference import Predictor 

    # 'flattened_leaves' uses 2048 x 8192 px images
    pred = Predictor(config_name='flattened_leaves')  
    pred.predict(images_src='data/images', export_dst='export')


This code snippet produces all necessary predictions in the ``export`` folder for evaluating images of canopy. If you want to make visualizations
for manual inspection of the images execute the following:

.. code-block:: Python

    from leaf.visualization import FlattenedVisualizer

    vis = FlattenedVisualizer(
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
   ├── symptoms_det/
   │   ├── pred/
   │   └── vis/
   ├── symptoms_seg/
   │   ├── pred/
   │   └── vis/
   └── visualization_symptoms/

Regardless of the visualization step you can compute the relevant disease metrics. To compute the relevant disease metrics per image
execute the following snippet

.. code-block:: Python

    from leaf.metrics import flat_leaves_evaluation_wrapper

    flat_leaves_evaluation_wrapper(root_folder='export', results_path='export/flattened_results.csv')