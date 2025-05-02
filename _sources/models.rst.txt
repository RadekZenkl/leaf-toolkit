Models
===============

- **Symptoms Detection**
  
  - ``zenkl_et_al_2025b`` *(latest)*
  - ``zenkl_et_al_2025a``
  - ``anderegg_et_al_2024``

- **Symptoms Segmentation**
  
  - ``zenkl_et_al_2025b`` *(latest)* – uses 1/2 downscaling on the input
  - ``latest_large`` – uses full resolution
  - ``zenkl_et_al_2025a`` – uses full resolution
  - ``tracking_latest`` – uses 1/2 downscaling on the input
  - ``anderegg_et_al_2024`` – uses full resolution

- **Organ Segmentation**
  
  - ``zenkl_et_al_2025b`` *(latest)* – uses 1/4 downscaling on the input

- **Focus Estimation**

  .. note::
     The ``Depth Anything v2`` model is sensitive to input size due to TorchScript export constraints and cannot predict on arbitrary image sizes. Pull requests to fix this are welcome.

  Supported input sizes:
  
  - ``4096x6144`` *(latest)*
  - ``6144x4096``
  - ``4096x4096``
  - ``2048x2048``
  - ``1024x1024``

  Please note that the focus estimation uses 1/4 downscaling on the input.