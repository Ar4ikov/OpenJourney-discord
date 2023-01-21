# StabilityPipeline

Это оптимизированный враппер для 🧨`diffusers`
Реализовано 2 use-case: 
* In-code реализация
* CLI реализация

## In-code реализация

```python
from sd_pipeline import StabilityPipeline

# Создаем экземпляр класса
# В качестве аргумента передаем название пайплайна
# text2img - для текста в картинку
# img2img - для картинки в картинку
pipeline = StabilityPipeline('text2img')

# Создадим рандомный промпт, укажем количество шагов генерации, высоту, ширину, CFG, все по классике
prompt = "Hello world"
steps = 50
width = 256
height = 256
cfg = 14

# Опишем поведение пайплайна
# Для этого создадим функцию, которая принимает на вход пайплайн

def diffusion_pipeline(pipeline, N=4):
    """
    Пример: генерация N независимых друг от друга картинок через цикл
    """

    # Добавляем в пайплайн 4 diffuser-а
    images = []

    for i in range(N):
        # 
        image = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=cfg
        ).images[0]

        images.append(image)

    return image


# Запускаем пайплайн с включенной оптимизацией
# В качестве аргумента передаем функцию, которая описывает поведение пайплайна

# получаем результат из работы поведения diffusion_pipeline

result = pipeline.create_job(
    diffusion_pipeline,
    device='cuda:0'
    use_unload_memory=True
    N=4,
)

# Сохраним результат
# Для генерации случайных имен файлов используем uuid
import uuid

for image in result:
    image.save(f'{uuid.uuid4()}.png')
```

## CLI реализация

Посмотрим на аргументы, которые принимает CLI

```bash
python sd_pipeline/inference.py -h
```

Вывод:

```bash
usage: inference.py [-h] [--pipe_type PIPE_TYPE] [--prompt PROMPT] [--imaginary_prompt | --no-imaginary_prompt] [--image_path IMAGE_PATH]
                    [--guidance_scale GUIDANCE_SCALE] [--num_steps NUM_STEPS] [--num_samples NUM_SAMPLES] [--strength STRENGTH]
                    [--negative_prompt NEGATIVE_PROMPT] [--width WIDTH] [--height HEIGHT] [--seed SEED] [--sd_model_id SD_MODEL_ID]
                    [--gpt_model_id GPT_MODEL_ID] [--scheduler SCHEDULER] [--device DEVICE] [--use_unload_memory | --no-use_unload_memory]
                    [--use_upscale | --no-use_upscale] [--codeformer_inference_path CODEFORMER_INFERENCE_PATH] [--input_path INPUT_PATH]
                    [--output_path OUTPUT_PATH] [--bg_upscale | --no-bg_upscale] [--face_upsample | --no-face_upsample] [--w W]
                    [--upscale_ratio UPSCALE_RATIO] [--cols COLS] [--rows ROWS] [--filename FILENAME] [--nsfw-generate | --no-nsfw-generate]

Diffusers Stability Pipeline

options:
  -h, --help            show this help message and exit
  --pipe_type PIPE_TYPE
                        Type of pipeline to run
  --prompt PROMPT       Prompt to use for text2img pipeline
  --imaginary_prompt, --no-imaginary_prompt
                        Imaginary prompt to use for pipeline (default: False)
  --image_path IMAGE_PATH
                        Path to image to use for img2img pipeline
  --guidance_scale GUIDANCE_SCALE
                        Guidance scale to use for text2img pipeline
  --num_steps NUM_STEPS
                        Number of steps to run for
  --num_samples NUM_SAMPLES
                        Number of samples to run for
  --strength STRENGTH   Strength to use for img2img pipeline
  --negative_prompt NEGATIVE_PROMPT
                        Negative prompt
  --width WIDTH         Width to use for text2img pipeline
  --height HEIGHT       Height to use for text2img pipeline
  --seed SEED           Seed to generate
  --sd_model_id SD_MODEL_ID
                        SD Model id to use
  --gpt_model_id GPT_MODEL_ID
                        GPT Model id to use
  --scheduler SCHEDULER
                        Scheduler to use
  --device DEVICE       Device to use
  --use_unload_memory, --no-use_unload_memory
                        Use unload memory (default: False)
  --use_upscale, --no-use_upscale
                        Use upscale (default: False)
  --codeformer_inference_path CODEFORMER_INFERENCE_PATH
                        Path to codeformer inference script
  --input_path INPUT_PATH
                        Path to input text
  --output_path OUTPUT_PATH
                        Path to output directory
  --bg_upscale, --no-bg_upscale
                        Upscale background (default: False)
  --face_upsample, --no-face_upsample
                        Upsample face (default: False)
  --w W                 w value for codeformer
  --upscale_ratio UPSCALE_RATIO
                        Upscale ratio for codeformer
  --cols COLS           Number of columns in image grid
  --rows ROWS           Number of rows in image grid
  --filename FILENAME   Filename for image grid
  --nsfw-generate, --no-nsfw-generate
                        Enable NSFW generation in pipeline (default: True)
```

### Пример генерации по тексту (без со-генерации промта)

```bash
python sd_pipeline/inference.py --pipe_type text2img --prompt "A Tokio town landscape, sunset" --guidance_scale 14 --num_steps 50 --num_samples 4 --width 512 --height 512 --device cuda:0 --use_unload_memory
```

### Пример генерации по тексту (с со-генерацией промта)

```bash
python sd_pipeline/inference.py --pipe_type text2img --prompt "A Tokio town landscape, sunset" --imaginary_prompt --guidance_scale 14 --num_steps 50 --num_samples 4 --width 512 --height 512 --device cuda:0 --use_unload_memory
```

### Пример генерации по тексту с использованием Codeformer (Upscale изображений 2X)

```bash
python sd_pipeline/inference.py --pipe_type text2img --prompt "A Tokio town landscape, sunset" --imaginary_prompt --guidance_scale 14 --num_steps 50 --num_samples 4 --width 512 --height 512 --device cuda:0 --use_unload_memory --use_upscale --codeformer_inference_path codeformer/inference.py --bg_upscale --face_upsample --w 0.75 --upscale_ratio 2
```

### Пример генерации по тексту с использованием Codeformer (Upscale изображений 4X)

```bash
python sd_pipeline/inference.py --pipe_type text2img --prompt "A Tokio town landscape, sunset" --imaginary_prompt --guidance_scale 14 --num_steps 50 --num_samples 4 --width 512 --height 512 --device cuda:0 --use_unload_memory --use_upscale --codeformer_inference_path codeformer/inference.py --bg_upscale --face_upsample --w 0.75 --upscale_ratio 2
```

### Отключение генерации NSFW контента

```bash
python sd_pipeline/inference.py --pipe_type text2img --prompt "A Tokio town landscape, sunset" --imaginary_prompt --guidance_scale 14 --num_steps 50 --num_samples 4 --width 512 --height 512 --device cuda:0 --use_unload_memory --use_upscale --codeformer_inference_path codeformer/inference.py --bg_upscale --face_upsample --w 0.75 --upscale_ratio 2 
--no-nsfw-generate
```
