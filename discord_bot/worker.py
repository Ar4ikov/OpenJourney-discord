from sd_pipeline import StabilityPipeline, StabilityPipelineType
from diffusers import schedulers
import pathlib
import uuid
import PIL
import torch
import random
import threading
from .prompt_factory import QueueItem, Prompt, Interaction
from typing import Callable
from time import sleep
import shutil
import re
import os


class Queue:
    def __init__(self):
        self.queue: list[QueueItem] = []

    def put(self, item):
        self.queue.append(item)

    def get(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            return None

    def __len__(self):
        return len(self.queue)


class OpenJourneyWorker(threading.Thread):
    def __init__(self, client, queue: Queue, upload_path: str, device="cpu", sd_model_ids: list[str] = None, *args, **kwargs):
        super().__init__()
        self.upload_path = pathlib.Path(upload_path)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.queue = queue
        self.device = device
        self.name = 'OpenJoureyWorker'

        self.sd_model_ids = sd_model_ids
        self.gpt_model_id = kwargs.get('gpt_model_id')
        self.gpu_first = kwargs.get('gpu_first', False)

        self.models_pipelines: dict[str, dict[StabilityPipelineType, StabilityPipeline]] = {}

        if not self.sd_model_ids:
            model_pipeline = {
                StabilityPipelineType.TEXT2IMG: None,
                StabilityPipelineType.IMG2IMG: None
            }

            model_pipeline[StabilityPipelineType.TEXT2IMG] = StabilityPipeline(
                pipe_type=StabilityPipelineType.TEXT2IMG,
                device='cpu' if not self.gpu_first else self.device,
                sd_model_id=self.sd_model_id,
                gpt_model_id=self.gpt_model_id,
                safety_check=not kwargs.get('nsfw_generate', True)
            )

            model_pipeline[StabilityPipelineType.IMG2IMG] = StabilityPipeline(
                pipe_type=StabilityPipelineType.IMG2IMG,
                device='cpu' if not self.gpu_first else self.device,
                sd_model_id=self.sd_model_id,
                gpt_model_id=self.gpt_model_id
            )

            self.models_pipelines['default'] = model_pipeline

        else:
            for model_id in self.sd_model_ids:
                model_pipeline = {
                    StabilityPipelineType.TEXT2IMG: None,
                    StabilityPipelineType.IMG2IMG: None
                }

                model_pipeline[StabilityPipelineType.TEXT2IMG] = StabilityPipeline(
                    pipe_type=StabilityPipelineType.TEXT2IMG,
                    device='cpu' if not self.gpu_first else self.device,
                    sd_model_id=model_id,
                    gpt_model_id=self.gpt_model_id,
                    safety_check=not kwargs.get('nsfw_generate', True)
                )

                model_pipeline[StabilityPipelineType.IMG2IMG] = StabilityPipeline(
                    pipe_type=StabilityPipelineType.IMG2IMG,
                    device='cpu' if not self.gpu_first else self.device,
                    sd_model_id=model_id,
                    gpt_model_id=self.gpt_model_id
                )

                self.models_pipelines[model_id] = model_pipeline

            self.models_pipelines['default'] = self.models_pipelines[self.sd_model_ids[0]]

        # for use in run()
        self.text2img_pipeline: StabilityPipeline = None
        self.img2img_pipeline: StabilityPipeline = None

        self.callback_success_function = None
        self.callback_failure_function = None

    def setup_success_callback(self, function: Callable, *args, **kwargs):
        self.callback_success_function = {
            'function': function,
            'args': args,
            'kwargs': kwargs
        }

    def setup_failure_callback(self, function: Callable, *args, **kwargs):
        self.callback_failure_function = {
            'function': function,
            'args': args,
            'kwargs': kwargs
        }

    def text_diffuse(self, pipeline, prompt, num_steps, num_samples, cfg, sizes: tuple[int] = (None, None), 
        negative_prompt = None, seed = None, scheduler = None):
        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if scheduler is not None:
            self.text2img_pipeline.set_scheduler(scheduler.value)

        images = []

        for i in range(num_samples):
            generator = torch.Generator(self.device).manual_seed(seed + i)
            
            img = pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                generator=generator,
                width=sizes[0],
                height=sizes[1]
            ).images[0]

            images.append(img)

        return images

    def image_diffuse(self, pipeline, prompt, ref_img: PIL.Image, num_steps, num_samples, cfg, strength,
        negative_prompt = None, seed = None, scheduler = None):
        if seed is None:
            seed = random.randint(0, 2 ** 32)

        if scheduler is not None:
            self.img2img_pipeline.set_scheduler(scheduler.value)

        images = []

        for i in range(num_samples):
            generator = torch.Generator(self.device).manual_seed(seed + i)
            
            img = pipeline(
                prompt=prompt,
                image=ref_img,
                num_inference_steps=num_steps,
                negative_prompt=negative_prompt,
                guidance_scale=cfg,
                strength=strength,
                generator=generator
            ).images[0]

            images.append(img)

        return images

    def run(self):
        while True:
            item = self.queue.get()

            if item is None:
                sleep(0.00001)
                continue
            
            prompt: Prompt
            interaction: Interaction
            self.text2img_pipeline: StabilityPipeline
            self.img2img_pipeline: StabilityPipeline

            prompt, interaction = item.prompt, item.interaction
            self.text2img_pipeline = self.models_pipelines.get(prompt.sd_model_id, self.models_pipelines['default'])[StabilityPipelineType.TEXT2IMG]
            self.img2img_pipeline = self.models_pipelines.get(prompt.sd_model_id, self.models_pipelines['default'])[StabilityPipelineType.IMG2IMG]

            # imagine a prompt
            if prompt.generated_prompt is None:
                prompt.generated_prompt = self.text2img_pipeline.imagine_prompt(
                    prompt=prompt.prompt,
                    device=self.device
                )

            if prompt.pipe_type == StabilityPipelineType.TEXT2IMG:
                images = self.text2img_pipeline.create_job(
                    self.text_diffuse, device=self.device,
                    prompt=prompt.generated_prompt,
                    num_steps=prompt.steps,
                    num_samples=prompt.num_samples,
                    cfg=prompt.guidance_scale,
                    sizes=prompt.aspect_ratio_sizes,
                    negative_prompt=prompt.negative_prompt,
                    seed=prompt.seed,
                    scheduler=prompt.scheduler
                )

            elif prompt.pipe_type == StabilityPipelineType.IMG2IMG:
                # using regex for check if image_url is valid url of valid local filename
                is_valid_url = self.text2img_pipeline.is_image(prompt.image_url)
                is_valid_local_filename = os.path.isfile(prompt.image_url)
                
                if not is_valid_url and not is_valid_local_filename:
                    # emit a failure in the asyncio loop
                    coro = self.callback_failure_function['function'](
                        interaction, 'Invalid image url',
                        *self.callback_failure_function['args'],
                        **self.callback_failure_function['kwargs']
                    )

                    self.client.loop.create_task(coro)

                    continue
                
                if is_valid_url:
                    # get an image
                    try:
                        image = self.img2img_pipeline.get_image(prompt.image_url)
                    except Exception as e:
                        # emit a failure in the asyncio loop
                        coro = self.callback_failure_function['function'](
                            interaction, e,
                            *self.callback_failure_function['args'],
                            **self.callback_failure_function['kwargs']
                        )

                        self.client.loop.create_task(coro)

                        continue

                else:
                    # load an image
                    image = PIL.Image.open(prompt.image_url).convert('RGB')

                # resize image
                image = self.img2img_pipeline.resize_image(image, prompt.image_resize, must_scale=True)

                # create a job
                images = self.img2img_pipeline.create_job(
                    self.image_diffuse, device=self.device,
                    prompt=prompt.generated_prompt,
                    ref_img=image,
                    num_steps=prompt.steps,
                    num_samples=prompt.num_samples,
                    cfg=prompt.guidance_scale,
                    strength=prompt.strength,
                    negative_prompt=prompt.negative_prompt,
                    seed=prompt.seed,
                    scheduler=prompt.scheduler
                )

            elif prompt.pipe_type == StabilityPipelineType.IMG2UPSCALE:
                images = [PIL.Image.open(prompt.image_url).convert('RGB')]

            sd_batch_id = str(uuid.uuid4())
            sd_input_path = self.upload_path / sd_batch_id
            sd_input_path.mkdir(parents=True)
            codeformer_batch_id = str(uuid.uuid4())
            codeformer_output_path = self.upload_path / codeformer_batch_id

            # save images with generating random names by uuid
            for img in images:
                filename = str(uuid.uuid4()) + '.png'

                img.save(str(self.upload_path / sd_batch_id / filename))

            # use a codeformer upscaler
            if prompt.use_codeformer:
                self.text2img_pipeline.upscale(
                    os.environ['CODEFORMER_PATH'],
                    str(self.upload_path / sd_batch_id),
                    str(codeformer_output_path),
                    prompt.use_background_upscale,
                    prompt.use_face_upscale,
                    prompt.upscale_weight,
                    prompt.upscale_ratio,
                )

            # move files from codeformer_output_path to sd_batch_id
            for file in (codeformer_output_path / 'final_results').iterdir():
                file_name = 'codeformer_' + str(file.name)
                file.rename(self.upload_path / sd_batch_id / file_name)

            # recursively remove codeformer_output_path
            shutil.rmtree(str(codeformer_output_path))
            
            # catch codeformer filenames
            codeformer_filenames = [str(file) for file in (self.upload_path / sd_batch_id).iterdir() if file.name.startswith('codeformer_')]

            prompt.output_filenames = codeformer_filenames.copy()
            
            # create image grid
            self.img2img_pipeline.create_image_grid(
                [PIL.Image.open(file) for file in codeformer_filenames],
                str(self.upload_path / sd_batch_id),
                'grid',
                image_side_size=prompt.grid_size,
                rows=prompt.grid_rows,
                cols=prompt.grid_cols
            )

            # emit a success in the asyncio loop
            coro = self.callback_success_function['function'](
                interaction=interaction, prompt=prompt, image_path=str(self.upload_path / sd_batch_id / 'grid.png'),
                *self.callback_success_function['args'],
                **self.callback_success_function['kwargs']
            )

            self.client.loop.create_task(coro)


class OpenJourneyController:
    def __init__(self, client, upload_path: str, sd_model_ids: list[str], num_gpus: int = 1, num_workers_per_gpu: int = 2, *args, **kwargs):
        self.num_gpus = num_gpus
        self.num_workers_per_gpu = num_workers_per_gpu

        self.queue = Queue()
        self.workers = []
        self.client = client

        # create workers
        for i in range(self.num_gpus):
            for j in range(self.num_workers_per_gpu):
                worker = OpenJourneyWorker(
                    self.client, self.queue, upload_path, f'cuda:{i}', sd_model_ids, *args, **kwargs
                )

                self.workers.append(worker)
    
    def set_callback_success_function(self, function, *args, **kwargs):
        for worker in self.workers:
            worker.setup_success_callback(function, *args, **kwargs)

    def set_callback_failure_function(self, function, *args, **kwargs):
        for worker in self.workers:
            worker.setup_failure_callback(function, *args, **kwargs)

    def start(self):
        for worker in self.workers:
            worker.start()

    def add_job(self, prompt: Prompt, interaction: Interaction):
        self.queue.put(QueueItem(prompt, interaction))
