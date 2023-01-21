from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from diffusers import schedulers
from transformers import pipeline as tpl
import torch
import os

__all__ = ['create_text_pipeline', 'create_image_scheduler', 'create_codeformer_oipeline', 'create_imaginary_prompt_pipeline']


def create_text_pipeline(device: torch.device, model_id: str, 
    scheduler: schedulers.SchedulerMixin = None, safety_check: bool = False) -> StableDiffusionPipeline:
    """
    Creates a text pipeline from a model id and a scheduler

    :param device: The device to run the pipeline on
    :param model_id: The model id to load
    :param scheduler: The scheduler to use
    :param safety_check: Whether to run a safety check

    :return: The pipeline
    """
    kwargs = {}
    if safety_check is False:
        kwargs['safety_checker'] = None

    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        **kwargs
    )
    pipeline = pipeline.to(device)

    if scheduler is None:
        _scheduler = schedulers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    else:
        _scheduler = scheduler.from_config(pipeline.scheduler.config)

    pipeline.scheduler = _scheduler

    return pipeline


def create_image_pipeline(device: torch.device, model_id: str, 
    scheduler: schedulers.SchedulerMixin = None, safety_check: bool = False) -> StableDiffusionImg2ImgPipeline:
    """
    Creates an image pipeline from a model id and a scheduler

    :param device: The device to run the pipeline on
    :param model_id: The model id to load
    :param scheduler: The scheduler to use
    :param safety_check: Whether to run a safety check

    :return: The pipeline
    """
    kwargs = {}
    if safety_check is False:
        kwargs['safety_checker'] = None

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        **kwargs
    )
    pipeline = pipeline.to(device)

    if scheduler is None:
        _scheduler = schedulers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    else:
        _scheduler = scheduler.from_config(pipeline.scheduler.config)

    pipeline.scheduler = _scheduler

    return pipeline


def create_codeformer_pipeline(codeformer_inference_path: str, input_path: str, 
    output_path: str, bg_upscale: bool = False, face_upsample: bool = True, 
    w: float = 0.75, upscale_ratio: int = 2) -> list[str]:
    """
    Creates a codeformer pipeline from a model id and a scheduler

    :param codeformer_inference_path: The path to the codeformer inference script
    :param input_path: The path to the input video
    :param output_path: The path to the output video
    :param bg_upscale: Whether to upscale the background
    :param face_upsample: Whether to upscale the face
    :param w: The weight of the quality upsample strength
    :param upscale_ratio: The ratio to upscale by

    :return: The pipeline
    """

    # minmax scaler for upsample ratio (2, 4)
    upscale_ratio = max(2, min(4, upscale_ratio))

    # minmax scaler for w (0.5, 1)
    w = max(0.5, min(1, w))

    # get python home from current environment (conda-like, poetry-like & etc.)
    python_home = os.popen('which python').read().strip()

    arguments = [python_home, codeformer_inference_path, '--input_path', 
        input_path, '-o', output_path, '-w', str(w), '-s', str(upscale_ratio)]
    
    if bg_upscale:
        arguments.extend(['--bg_upsampler', 'realesrgan'])

    if face_upsample:
        arguments.extend(['--face_upsample'])

    return arguments


def create_imaginary_prompt_pipeline(prompt: str, device: torch.device, model_id: str = None, max_length: int = 77) -> tpl:
    """
    Creates an imaginary prompt pipeline from a model id
    
    :param prompt: The prompt to generate
    :param device: The device to run the pipeline on
    :param model_id: The model id to load
    :param max_length: The max length of the prompt
    
    :return: The pipeline
    """
    if model_id is None:
        model_id = 'Ar4ikov/gpt2-pt-2-stable-diffusion-prompt-generator'

    pipeline = tpl('text-generation', model=model_id, tokenizer=model_id, device=device)

    # generate prompt
    prompt = pipeline(prompt, max_length=max_length)[0]['generated_text']

    # split it by newlines and dot
    dot_split = prompt.split('.')
    newline_split = prompt.split('\n')

    if len(dot_split[0]) < len(newline_split[0]):
        prompt = dot_split[0]
    else:
        prompt = newline_split[0]

    # remove pipeline
    del pipeline

    # unload memory
    torch.cuda.empty_cache()

    return prompt
