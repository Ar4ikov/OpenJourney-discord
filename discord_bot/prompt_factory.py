from dataclasses import dataclass, asdict
from typing import Literal, TypeAlias
from discord import Interaction
from diffusers import schedulers
from sd_pipeline import StabilityPipelineType
from enum import Enum


class SchedulerType(Enum):
    EULER = schedulers.EulerAncestralDiscreteScheduler
    LMS = schedulers.LMSDiscreteScheduler
    DPMSOLVER = schedulers.DPMSolverMultistepScheduler


ASPECT_RATIO: TypeAlias = Literal["16:9", "4:3", "1:1-max", "1:1", "9:16", "3:4"]

# sizes for max side is 768 px
ASPECT_RATIO_SIZES = {
    "16:9": (768, 432),
    "4:3": (768, 576),
    "1:1-max": (640, 640),
    "1:1": (512, 512),
    "9:16": (432, 768),
    "3:4": (576, 768)
}


@dataclass
class Prompt:
    pipe_type: StabilityPipelineType
    prompt: str
    seed: int
    sd_model_id: str = None
    gpt_model_id: str = None
    scheduler: SchedulerType = SchedulerType.DPMSOLVER
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


def prompt_asdict_factory(prompt_data):
    def convert_value(obj):
        if isinstance(obj, StabilityPipelineType):
            return obj.value

        if isinstance(obj, SchedulerType):
            return obj.name.lower()

        return obj

    return dict((k, convert_value(v)) for k, v in prompt_data)


@dataclass
class QueueItem:
    prompt: Prompt
    interaction: Interaction