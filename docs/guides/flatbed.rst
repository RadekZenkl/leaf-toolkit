Flatbed Scanners
-----


This guide provides shows you how to evaluate images of flattened leaves outdoors following the method of `Stewart et al. 2016 <https://apsjournals.apsnet.org/doi/full/10.1094/PHYTO-01-16-0018-R>`_ , 
visualize results and compute disease metrics.

First make sure that all your images are exported to a common image format (.jpg, .png, etc.), they are named uniquely,
and cropped to a consistent resolution. For images prepared with the ``data_prep`` module of this package we use  
`1024 x 6144 px`.

Then you can use an existing configuration and predict simply a complete folder. You can either use a flat or 
nested folder structure.

Using a flat folder structure for export:

.. code-block:: Python

    from leaf.inference import Predictor 

    # 'flatbed' uses 1024 x 6144 px images
    pred = Predictor(config_name='flatbed')  
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