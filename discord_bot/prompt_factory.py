from dataclasses import dataclass, asdict
from typing import Literal
from discord import Interaction

pipe_type = Literal['text2img', 'img2img', 'img2upscale']
schedulers = Literal['euler', 'lms', 'dpmsolver']

# TODO: Add options for sd_model_id and gpt_model_id

@dataclass
class Prompt:
    pipe_type: pipe_type
    prompt: str
    seed: int
    sd_model_id: str = None
    gpt_model_id: str = None
    scheduler: schedulers = 'dpmsolver'
    generated_prompt: str = None
    guidance_scale: float = 7.5
    steps: int = 50
    num_samples: int = 4
    negative_prompt: str = None
    image_url: str = None
    image_resize: int = 512
    strength: float = 0.75
    aspect_ratio_sizes: tuple[int] = (640, 640)
    use_codeformer: bool = True
    use_face_upscale: bool = True
    use_background_upscale: bool = True
    upscale_weight: float = 0.75
    upscale_ratio: int = 2
    output_filenames: list[str] = None
    grid_cols: int = 2
    grid_rows: int = 2
    grid_size: int = 1536

    def to_dict(self):
        return asdict(self)


@dataclass
class QueueItem:
    prompt: Prompt
    interaction: Interaction