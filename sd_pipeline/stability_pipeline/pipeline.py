from .pipeline_mixer import create_codeformer_pipeline, create_image_pipeline, create_text_pipeline, create_imaginary_prompt_pipeline
import torch
from diffusers import schedulers
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import subprocess
import PIL
from typing import Callable
import gc
import requests
import io

__all__ = ['StabilityPipeline']


class StabilityPipeline():
    def __init__(self, pipe_type: str, scheduler: schedulers.SchedulerMixin = None, 
        device: torch.device = 'cpu', sd_model_id: str = None, gpt_model_id: str = None, **kwargs):
        """
        Creates a stability pipeline
        
        :param pipe_type: The type of pipeline to create, either "text2img" or "img2img"
        :param device: The device to run the pipeline on
        :param scheduler: The scheduler to use
        :param sd_model_id: The stable-diffusion model id to use
        :param gpt_model_id: The gpt model id to use
        """

        if sd_model_id is None:
            self.sd_model_id = 'dreamlike-art/dreamlike-diffusion-1.0'
        else:
            self.sd_model_id = sd_model_id

        if gpt_model_id is None:
            self.gpt_model_id = 'Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator'
        else:
            self.gpt_model_id = gpt_model_id

        if 'safety_check' in kwargs:
            self.safety_check = kwargs['safety_check']
            
        else:
            self.safety_check = False

        self.pipeline = None
        self.pipe_type = pipe_type

        self.setup_pipeline(pipe_type, device, scheduler, self.safety_check)

    def setup_pipeline(self, pipe_type: str, device: torch.device, scheduler: schedulers.SchedulerMixin = None,
        safety_check: bool = False):
        if pipe_type == 'text2img':
            pipeline: StableDiffusionPipeline = create_text_pipeline(model_id=self.sd_model_id, device=device, scheduler=scheduler, 
                safety_check=safety_check)
        
        elif pipe_type == 'img2img':
            pipeline: StableDiffusionImg2ImgPipeline = create_image_pipeline(model_id=self.sd_model_id, device=device, scheduler=scheduler,
                safety_check=safety_check)

        elif pipe_type == 'img2upscale':
            pipeline: Callable = self.upscale

        else:
            raise ValueError('Invalid pipe_type, must be either "text2img" or "img2img"')
        
        # XFormers memory efficient attention optimization
        pipeline.enable_xformers_memory_efficient_attention()
        self.pipeline = pipeline

    def set_scheduler(self, scheduler: schedulers.SchedulerMixin):
        """
        Sets the scheduler

        :param scheduler: The scheduler to use
        """

        _scheduler = scheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.scheduler = _scheduler

    def create_job(self, pipeline_job_func: Callable, device='cpu', use_unload_memory: bool = True, *args, **kwargs):
        """
        Creates a job to run on the cluster

        :param pipeline_job_func: The function to run on the pipeline
        :param device: The device to run the pipeline on
        :param use_unload_memory: Whether to unload memory after the job is done
        :param args: The args to pass to the pipeline job function
        :param kwargs: The kwargs to pass to the pipeline job function

        :return: The output of the pipeline job function
        """

        if self.pipe_type != 'img2upscale':
            # move pipeline to gpu
            self.pipeline.to(device)

        # unload cpu memory
        gc.collect()

        output = pipeline_job_func(self.pipeline, *args, **kwargs)
        
        if use_unload_memory:
            if self.pipe_type != 'img2upscale':
                # move pipeline to cpu
                self.pipeline.to('cpu')
            
            # unload memory
            torch.cuda.empty_cache()

        return output

    def upscale(self, codeformer_inference_path: str, input_path: str, 
        output_path: str, bg_upscale: bool = False, face_upsample: bool = True, 
        w: float = 0.75, upscale_ratio: int = 2):
        """
        Upscales an image using codeformer

        :param codeformer_inference_path: The path to the codeformer inference script
        :param input_path: The path to the input image
        :param output_path: The path to the output image
        :param bg_upscale: Whether to upscale the background
        :param face_upsample: Whether to upsample the face
        :param w: The weight of the face
        :param upscale_ratio: The ratio to upscale the image by

        :return: Whether the image was upscaled successfully
        """

        args = create_codeformer_pipeline(codeformer_inference_path, input_path,
            output_path, bg_upscale, face_upsample, w, upscale_ratio)

        # print(args)

        pipe = subprocess.Popen(args, stdout=subprocess.DEVNULL)

        pipe.wait()

        return True

    def imagine_prompt(self, prompt: str, device: torch.device, max_length: int = 77):
        """
        Creates an imaginary prompt
        
        :param prompt: The prompt to create the imaginary prompt from
        :param device: The device to run the pipeline on
        :param max_length: The max length of the imaginary prompt
        
        :return: The imaginary prompt
        """
        return create_imaginary_prompt_pipeline(
            prompt=prompt, model_id=self.gpt_model_id, device=device, max_length=max_length
        )

    def create_image_grid(self, images: list[PIL.Image], output_path: str, filename: str,
        cols=2, rows=2, image_side_size: int = None):
        """
        Creates an image grid from a list of images
        
        :param images: The list of images to create the grid from
        :param output_path: The path to save the image grid to
        :param filename: The filename to save the image grid as
        :param cols: The number of columns in the image grid
        :param rows: The number of rows in the image grid
        :param image_side_size: The size of the side of the image
        
        :return: None
        """
        # create image grid
        image_grid = PIL.Image.new('RGB', (images[0].width * cols, images[0].height * rows))

        for i, image in enumerate(images):
            image_grid.paste(image, (image.width * (i % cols), image.height * (i // cols)))

        # resize image grid
        if image_side_size is not None:
            image_grid = self.resize_image(image_grid, image_side_size, must_scale=False)

        image_grid.save(f'{output_path}/{filename}.png')

    # check if valid content for image is passed in url
    def is_image(self, url):
        """
        Checks if the url is an image
        
        :param url: The url to check
        
        :return: Whether the url is an image
        """
        try:
            response = requests.get(url)
            PIL.Image.open(io.BytesIO(response.content)).convert("RGB")
            return True
        except:
            return False

    # get image from url
    def get_image(self, url):
        """
        Gets an image from a url
        
        :param url: The url to get the image from
        
        :return: The image
        """
        try:
            response = requests.get(url)
        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(e)

        assert self.is_image(url) == True, "Invalid image url"
        return PIL.Image.open(io.BytesIO(response.content)).convert("RGB")

    # resize image from get_image(), after resizing one of side should be 512
    def resize_image(self, prompt_image, size=512, must_scale=True):
        """
        Resizes an image
        
        :param prompt_image: The image to resize
        :param size: The size to resize the image to
        :param must_scale: Whether the image must be scaled
        
        :return: The resized image
        """
        # если обе из сторон не превышают size, то возвращаем изображение
        if prompt_image.width < size and prompt_image.height < size and not must_scale:
            return prompt_image

        # получить соотношение сторон изображения
        ratio = prompt_image.width / prompt_image.height

        # если ширина больше высоты
        if ratio > 1:
            # то ширину делаем 512
            prompt_image = prompt_image.resize((size, int(size / ratio)), PIL.Image.ANTIALIAS)
        # если высота больше ширины
        elif ratio < 1:
            # то высоту делаем 512
            prompt_image = prompt_image.resize((int(size * ratio), size), PIL.Image.ANTIALIAS)
        # если соотношение сторон равно 1
        else:
            # то делаем 512 на 512
            prompt_image = prompt_image.resize((size, size), PIL.Image.ANTIALIAS)

        return prompt_image

    def __call__(self, *args, **kwargs):
        """
        Runs default diffusers pipeline in class
        """
        return self.pipeline(*args, **kwargs)
    
    def __repr__(self):
        return f'StabilityPipeline(model_id={self.sd_model_id})'
