import logging
import torch
from typing import Union

from hydra import compose, initialize
from omegaconf import OmegaConf

from leaf.models import SymptomsDetection, SymptomsSegmentation, OrgansSegmentation, FocusSegmentation


class Predictor:
    """
    This class is used to unify the complete pipeline into a single object with a simple predict method. 
    It provides capability to use a yaml file configuration for repeatability. Furthermore, it allows for 
    passing all the arguments for cofigring the individual building blocks of the pipeline. If a new block 
    """

    def __init__(self,
                 config_path: str = "config", config_name: str = "canopy_portrait",
                 symptoms_det_params: Union[dict, None] = None,
                 symptoms_seg_params: Union[dict, None] = None,
                 organs_params: Union[dict, None] = None,
                 focus_params: Union[dict, None] = None,
                 module_params:  Union[dict, None] = None
                 ) -> None:
        """
        Constructor of the predictor object. For possible parameters to configure see either the individual
        models directly or see the configuration yaml files in the config folder. 

        Args:
            config_path (str, optional): relative path from the location of this file to a config directory. 
                Defaults to "config".
            config_name (str, optional): name of a config within the config_path directory. 
                New configurations can be added. Defaults to "canopy_portrait".
            symptoms_det_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to symptoms detection model. It overrides the parameters from the 
                configuration file. Defaults to None.
            symptoms_seg_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to symptoms segmentation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            organs_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to organs segmentation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            focus_params (Union[dict, None], optional): An optional dictionary which directly passes 
                the contents as **kwargs to focus estimation model. It overrides the parameters from the 
                configuration file. Defaults to None.
            module_params (Union[dict, None], optional): An optional dictionary which controls which parts 
                of the pipeline are executed. It overrides the parameters from the configuration file. 
                Defaults to None.
        """
        
        
        # load base config
        with initialize(version_base=None, config_path=config_path):
            cfg = compose(config_name=config_name)
            config = OmegaConf.to_container(cfg, resolve=True)

        self.module_params = config.get('module_params', None)
        self.symptoms_det_params = config.get('symptoms_det_params', None)
        self.symptoms_seg_params = config.get('symptoms_seg_params', None)
        self.organs_params =  config.get('organs_params', None)
        self.focus_params =  config.get('focus_params', None)

        # override with user params
        if module_params is not None:
            self.module_params.update(module_params)
        if symptoms_det_params is not None:
            self.symptoms_det_params.update(symptoms_det_params)
        if symptoms_seg_params is not None:
            self.symptoms_seg_params.update(symptoms_seg_params)
        if organs_params is not None:
            self.organs_params.update(organs_params)
        if focus_params is not None:
            self.focus_params.update(focus_params)
        
    def predict(self, images_src: str, export_dst: str) -> None:
        """
        This method provides a simple interface to predict on images from a specified folder and 
        save the results to a specified location.

        Args:
            images_src (str): Path to location of images.
            export_dst (str): Path where the results should be saved.
        """

        logging.info("Predicting ...")
        logging.info("Emptying CUDA cache")
        torch.cuda.empty_cache()
        

        if self.module_params['symptoms_det']:
            logging.info("Symptoms Detection is running ... ")

            s_det = SymptomsDetection(
                **self.symptoms_det_params,
                export_pattern_pred=f'{export_dst}/symptoms_det/pred',
                )
            s_det.predict(images_src)

            logging.info("Symptoms Detection finished")

            logging.info("Emptying CUDA cache")
            torch.cuda.empty_cache()

        if self.module_params['symptoms_seg']:
            logging.info("Symptoms Segmentation is running ...")

            s_seg = SymptomsSegmentation(
                **self.symptoms_seg_params,
                export_pattern_pred=f'{export_dst}/symptoms_seg/pred',
                )
            s_seg.predict(images_src)

            logging.info("Symptoms Segmentation finished")

            logging.info("Emptying CUDA cache")
            torch.cuda.empty_cache()

        if self.module_params['organs']:
            logging.info("Organ Segmentation is running ...")

            o_seg = OrgansSegmentation(
                **self.organs_params,
                export_pattern_pred=f'{export_dst}/organs/pred',
                )
            o_seg.predict(images_src)

            logging.info("Organ Segmentation finished")

            logging.info("Emptying CUDA cache")
            torch.cuda.empty_cache()


        if self.module_params['focus']:
            logging.info("Focus Estimation is running ...")

            f_seg = FocusSegmentation(
                **self.focus_params,
                export_pattern_pred=f'{export_dst}/focus/pred',
                )
            f_seg.predict(images_src)

            logging.info("Focus Estimation finished")

            logging.info("Emptying CUDA cache")
            torch.cuda.empty_cache()

        logging.info("Predicting finished")
