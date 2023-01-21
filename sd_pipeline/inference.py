import argparse
from diffusers import schedulers
from stability_pipeline import StabilityPipeline
import random
import torch
import PIL
import pathlib
from uuid import uuid4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diffusers Stability Pipeline')
    parser.add_argument('--pipe_type', type=str, default='text2img', help='Type of pipeline to run')
    parser.add_argument('--prompt', type=str, help='Prompt to use for text2img pipeline')
    parser.add_argument('--imaginary_prompt', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Imaginary prompt to use for pipeline')
    parser.add_argument('--image_path', type=str, default=None, help='Path to image to use for img2img pipeline')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale to use for text2img pipeline')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of steps to run for')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples to run for')
    parser.add_argument('--strength', type=float, default=0.5, help='Strength to use for img2img pipeline')
    parser.add_argument('--negative_prompt', type=str, default=None, help='Negative prompt')
    parser.add_argument('--width', type=int, default=512, help='Width to use for text2img pipeline')
    parser.add_argument('--height', type=int, default=512, help='Height to use for text2img pipeline')
    parser.add_argument('--seed', type=int, default=None, help='Seed to generate')
    parser.add_argument('--sd_model_id', type=str, default='dreamlike-art/dreamlike-diffusion-1.0', help='SD Model id to use')
    parser.add_argument('--gpt_model_id', type=str, default='Ar4ikov/gpt2-pt-2-stable-diffusion-prompt-generator', help='GPT Model id to use')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--use_unload_memory', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use unload memory')
    parser.add_argument('--use_upscale', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Use upscale')
    parser.add_argument('--codeformer_inference_path', type=str, default='CodeFormer/inference_codeformer.py', help='Path to codeformer inference script')
    parser.add_argument('--input_path', type=str, default=None, help='Path to input text')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output directory')
    parser.add_argument('--bg_upscale', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Upscale background')
    parser.add_argument('--face_upsample', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Upsample face')
    parser.add_argument('--w', type=float, default=0.75, help='w value for codeformer')
    parser.add_argument('--upscale_ratio', type=int, default=2, help='Upscale ratio for codeformer')
    parser.add_argument('--cols', type=int, default=2, help='Number of columns in image grid')
    parser.add_argument('--rows', type=int, default=2, help='Number of rows in image grid')
    parser.add_argument('--filename', type=str, default=None, help='Filename for image grid')
    parser.add_argument('--nsfw-generate', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Enable NSFW generation in pipeline')

    args = parser.parse_args()

    # parse a scheduler
    schedulers = {
        'euler': schedulers.EulerAncestralDiscreteScheduler,
        'dmpsolver': schedulers.DPMSolverMultistepScheduler,
        'lms': schedulers.LMSDiscreteScheduler
    }

    scheduler = schedulers.get(args.scheduler)

    # create pipeline
    pipeline = StabilityPipeline(pipe_type=args.pipe_type, 
        scheduler=scheduler, device=args.device,
        sd_model_id=args.sd_model_id, gpt_model_id=args.gpt_model_id,
        safety_check=args.nsfw_generate)

    # min max scale for steps (10, 200)
    steps = max(min(args.num_steps, 200), 10)

    # min max scale for samples (1, 10)
    samples = max(min(args.num_samples, 10), 1)

    # min max scale for guidance scale (1, 40)
    guidance_scale = max(min(args.guidance_scale, 40), 1)

    # min max scale for strength (0.1, 1)
    strength = max(min(args.strength, 1), 0.1)

    # min max scale for width (128, 640)
    width = max(min(args.width, 640), 128)

    # min max scale for height (128, 640)
    height = max(min(args.height, 640), 128)

    if args.seed is None:
        seed = random.randint(0, 2 ** 32)
    else:
        seed = args.seed
    
    print(f'Using seed: {seed}')

    # create generator
    generator = torch.Generator(args.device).manual_seed(seed)

    # generate uuid for path or use input path
    if args.input_path is None:
        input_path = str(uuid4())
    else:
        input_path = args.input_path

    if args.output_path is None:
        output_path = str(uuid4())
    else:
        output_path = args.output_path

    if args.filename is None:
        filename = f'{str(uuid4())}.png'
    else:
        filename = args.filename

    # image prompt
    if args.imaginary_prompt:
        args.prompt = pipeline.imagine_prompt(
            args.prompt,
            device=args.device
        )

    print(f'Using prompt: {args.prompt}')
    print('Generating...')

    def diffusion_text_pipeline(df_pipeline):
        images = []
        
        for i in range(args.num_samples):
            image = df_pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            ).images[0]

            images.append(image)

        return images

    def diffusion_image_pipeline(df_pipeline):
        # load image first
        img = PIL.Image.open(args.image_path).convert('RGB')

        images = []

        for i in range(args.num_samples):
            image = df_pipeline(
                image_path=img,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                strength=strength,
                generator=generator
            ).images[0]

            images.append(image)

        return images

    if args.pipe_type == 'text2img':
        images = pipeline.create_job(diffusion_text_pipeline, device=args.device, use_unload_memory=args.use_unload_memory)

    elif args.pipe_type == 'img2img':
        images = pipeline.create_job(diffusion_image_pipeline, device=args.device, use_unload_memory=args.use_unload_memory)

    # create output directory
    basement = pathlib.Path(input_path)
    basement.mkdir(parents=True, exist_ok=True)

    # save images
    for image in images:
        # generate filenames
        f_name = f'{str(uuid4())}.png'
        image.save(basement / f_name)

    if args.use_upscale:
        pipeline.upscale(
            args.codeformer_inference_path, input_path, output_path, args.bg_upscale, 
            args.face_upsample, args.w, args.upscale_ratio 
        )

    # create image grid
    pipeline.create_image_grid(images, input_path, filename, args.cols, args.rows)

    # command 
    # python sd_pipeline/pipeline.py --prompt "A beautiful ginger woman in a firepit, by Makoto Shinkai, trending on artstation" 
    # --guidance_scale 14 --num_steps 40 --num_samples 4 --scheduler dpmsolver --device cuda:0 --use_upscale 
    # --codeformer_inference_path "/home/ar4ikov/Документы/StableDiffusionToMidjourney/CodeFormer/inference_codeformer.py" 
    # --bg_upscale --upscale_ratio 4 --use_memory_unload

    print('Done!')
