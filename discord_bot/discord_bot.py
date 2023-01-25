from discord import Client, Guild, Intents, Interaction, Object, File, ButtonStyle
from discord.app_commands import CommandTree, Command, describe
from discord.ui import View, Button, button
from diffusers import schedulers
import os
import random
import pathlib
import json
from dataclasses import replace, asdict
from typing import Literal, TypeAlias
from .prompt_factory import Prompt, prompt_asdict_factory, SchedulerType, ASPECT_RATIO, ASPECT_RATIO_SIZES
from .worker import OpenJourneyController
from sd_pipeline import StabilityPipelineType


SD_MODEL_IDS = []

# parse SD_MODEL_ID_{} from environ, starts from 1, if return None value - stop
for i in range(1, 100):
    model_id = os.environ.get(f'SD_MODEL_ID_{i}')
    if model_id is None:
        break
    SD_MODEL_IDS.append(model_id)


class OpenJourneyDialog(View):
    ACTION_TYPE: TypeAlias = Literal['variants', 'upscale', 'regenerate']

    def __init__(self, bot_class, interaction: Interaction, prompt: Prompt):
        super().__init__(timeout=99999999999)
        self.bot_class = bot_class
        self.interaction = interaction
        self.prompt = prompt
        self.do_imagine_prompt = True

    async def button_controller(self, interaction: Interaction, button: Button, _type: ACTION_TYPE, image_iter: int):
        prompt = replace(self.prompt)

        match _type:
            case 'variants':
                prompt.seed = random.randint(0, 2 ** 32)
                file_path = pathlib.Path(prompt.output_filenames[image_iter])
                prompt.image_url = str(file_path)
                # print(prompt.image_url)

                prompt.pipe_type = StabilityPipelineType.IMG2IMG
                prompt.output_filenames = None
                if self.do_imagine_prompt:
                    prompt.generated_prompt = None
                prompt.upscale_ratio = 2
                prompt.image_resize = 640
                prompt.steps = 120

                # and now we can generate the image
                await interaction.response.defer(thinking=True)

                # add job to the queue
                self.bot_class.controller.add_job(prompt, interaction)

            case 'upscale':
                file_path = pathlib.Path(prompt.output_filenames[image_iter])
                prompt.image_url = str(file_path)

                prompt.pipe_type = StabilityPipelineType.IMG2UPSCALE
                prompt.output_filenames = None
                prompt.prompt = prompt.generated_prompt
                prompt.generated_prompt = None
                prompt.upscale_ratio = 2
                prompt.num_samples = 1
                prompt.grid_cols = 1
                prompt.grid_rows = 1
                prompt.grid_size = 1024 * 10

                # and now we can generate the image
                await interaction.response.defer(thinking=True)

                # add job to the queue
                self.bot_class.controller.add_job(prompt, interaction)

            case 'regenerate':
                prompt.seed = random.randint(0, 2 ** 32)

                if self.do_imagine_prompt:
                    prompt.generated_prompt = None

                # and now we can generate the image
                await interaction.response.defer(thinking=True)

                # add job to the queue
                self.bot_class.controller.add_job(prompt, interaction)     
    
    @button(label="V1", style=ButtonStyle.secondary)
    async def v1(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'variants', 0)

    @button(label="V2", style=ButtonStyle.secondary)
    async def v2(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'variants', 1)

    @button(label="V3", style=ButtonStyle.secondary)
    async def v3(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'variants', 2)

    @button(label="V4", style=ButtonStyle.secondary)
    async def v4(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'variants', 3)

    @button(label="🔄", style=ButtonStyle.green)
    async def regenerate(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'regenerate', -1)

    @button(label="U1", style=ButtonStyle.primary)
    async def u1(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'upscale', 0)

    @button(label="U2", style=ButtonStyle.primary)
    async def u2(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'upscale', 1)

    @button(label="U3", style=ButtonStyle.primary)
    async def u3(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'upscale', 2)

    @button(label="U4", style=ButtonStyle.primary)
    async def u4(self, interaction: Interaction, button: Button):
        await self.button_controller(interaction, button, 'upscale', 3)

    @button(label="🔍 GP", style=ButtonStyle.green)
    async def switch_prompt_state(self, interaction: Interaction, button: Button):
        self.do_imagine_prompt = not self.do_imagine_prompt
        
        # change button style
        if self.do_imagine_prompt:
            button.style = ButtonStyle.green
            button.label = '🔎 GP'
        else:
            button.style = ButtonStyle.red
            button.label = '🔍 GP'

        await interaction.response.edit_message(view=self)


class OpenJourneyBot:
    def __init__(self):
        self.token = os.environ.get("DISCORD_TOKEN")
        self.guild_id = os.environ.get("GUILD_ID", -1)
        self.client = Client(intents=Intents.all())
        self.command_tree = CommandTree(self.client)

        if self.guild_id is None:
            self.guild = None
        else:
            self.guild = Object(id=int(self.guild_id))

        self.sd_model_ids = SD_MODEL_IDS.copy()

        self.gpt_model_id = os.environ['GPT_MODEL_ID']
        self.nsfw_generate = os.environ.get('NSFW_GENERATE', False).lower() in ['true', '1', 't', 'y', 'yes']
        self.controller = OpenJourneyController(self.client, os.environ['UPLOAD_PATH'], 
            sd_model_ids=self.sd_model_ids, gpt_model_id=self.gpt_model_id, nsfw_generate=self.nsfw_generate)

    def commands(self):
        self.command_tree.command(
            name="text_generate",
            description="Generate an image by text prompt",
            # guild=self.guild if self.guild_id != -1 else None
        )(self.text_generate)

        self.command_tree.command(
            name="image_generate",
            description="Generate an image by image prompt",
            # guild=self.guild if self.guild_id != -1 else None
        )(self.image_generate)

        self.command_tree.command(
            name='help',
            description='Get help',
            # guild=self.guild if self.guild_id != -1 else None
        )(self.help)
        
        self.client.event(self.on_ready)
        self.client.event(self.on_guild_join)

    async def on_ready(self):
        print('Syncing commands...')
        await self.command_tree.sync()
        print("Bot is ready!")

    async def on_guild_join(self, guild: Guild):
        print(f"Bot joined guild {guild.name}!")
        print("Syncing commands...")
        await self.command_tree.sync()

    async def success(self, interaction: Interaction, prompt: Prompt, image_path: str):
        # TODO: Fix Payload too large error
        attachments = [File(image_path, filename='grid.png')]

        # write out json file
        basement = pathlib.Path(image_path).parent
        json_path = basement / 'prompt.json'

        with open(json_path, 'w') as f:
            json.dump(asdict(prompt, dict_factory=prompt_asdict_factory), f, indent=4, ensure_ascii=False)

        # set up view
        if prompt.pipe_type != StabilityPipelineType.IMG2UPSCALE:
            view = OpenJourneyDialog(self, interaction=interaction, prompt=prompt)
        else:
            view = None

        content = f'Prompt: `{prompt.prompt}`\n' \
                    f'Generated prompt: `{prompt.generated_prompt}`\n' \
                    f'Steps: `{prompt.steps}`\n' \
                    f'Guidance scale: `{prompt.guidance_scale}`\n' \
                    f'Seed: `{prompt.seed}`\n' \
                    f'Scheduler: `{prompt.scheduler.name.lower()}`' \
                    '\n\nWARNING: This is the beta-test of new version, may not work properly. ' \
                    'Please report any bugs to @Ar4ikov#3805'

        if view is not None:
            await interaction.followup.send(
                content=content,
                files=attachments,
                view=view
            )
        else:
            await interaction.followup.send(
                content=content,
                files=attachments
            )

    async def failure(self, interaction: Interaction, error: str):
        await interaction.followup.send(
            content=f"ERROR: {error}"
        )

    @describe()
    async def help(self, interaction: Interaction):
        await interaction.response.send_message(
            content='https://button-aurora-7b8.notion.site/OpenJourney-Guide-05401d7a438e4e5cb2cbb241a15d6bdf'
        )

    @describe(
        prompt='Text to generate image', 
        steps='steps of the image (10 - 200)',
        guidance_scale='guidance scale of the image (1 - 40)',
        seed='seed of the image (0 - 2 ** 32)',
        negative_prompt='negative prompt',
        aspect_ratio='aspect ratio of the image',
        scheduler='scheduler of the image',
        do_imagine_prompt='let GPT2 to imagine the prompt for you'
    )
    async def text_generate(self, interaction: Interaction, 
        prompt: str, steps: int = 50, guidance_scale: int = 10, 
        seed: int = None, negative_prompt: str = None, aspect_ratio: ASPECT_RATIO = "1:1",
        scheduler: SchedulerType = SchedulerType.DPMSOLVER, do_imagine_prompt: bool = True):
        # set thinking status
        await interaction.response.defer(thinking=True)

        # check if valid values are passed
        try:
            assert steps >= 10 and steps <= 200, "steps should be between 10 and 200"
            assert guidance_scale >= 1 and guidance_scale <= 40, "guidance_scale should be between 1 and 40"
            assert seed is None or (seed >= 0 and seed <= 2 ** 32), "seed should be between 0 and 2 ** 32"
            assert aspect_ratio in ASPECT_RATIO_SIZES, "aspect_ratio should be one of the following: 1:1-max, 1:1, 4:3, 16:9"
            assert scheduler in SchedulerType, "scheduler should be one of the following: dpmsolver, lms, euler"
        except AssertionError as e:
            await interaction.followup.send(e)
            return

        if seed is None:
            seed = random.randint(0, 2 ** 32)

        # get ascpect ratio sizes
        aspect_sizes = ASPECT_RATIO_SIZES[aspect_ratio]

        # create prompt
        prompt = Prompt(
            pipe_type=StabilityPipelineType.TEXT2IMG,
            prompt=prompt,
            generated_prompt=None if do_imagine_prompt else prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            aspect_ratio_sizes=aspect_sizes,
            scheduler=scheduler,
            sd_model_id=SD_MODEL_IDS[0],
            gpt_model_id=os.environ.get('GPT_MODEL_ID')
        )

        # add job
        self.controller.add_job(prompt, interaction)

    @describe(
        image_url='your image url to prevent', 
        image_resize='resize image to custom side with same ratio (128 - 768)',
        prompt='Image to generate image', 
        strength='strength of the image (0.01 - 1.0)', 
        steps='steps of the image (10 - 200)',
        guidance_scale='generates image close to text prompt (1 - 40)',
        seed='seed of the image (0 - 2 ** 32)',
        negative_prompt='negative prompt',
        scheduler='scheduler of the image',
        do_imagine_prompt='let GPT2 to imagine the prompt for you'
    )
    async def image_generate(self, interaction: Interaction,
        image_url: str, prompt: str, image_resize: int = None,
        strength: float = 0.75, steps: int = 75, 
        guidance_scale: int = 10, seed: int = None, negative_prompt: str = None,
        scheduler: SchedulerType = SchedulerType.DPMSOLVER, do_imagine_prompt: bool = True):
        # set thinking status
        await interaction.response.defer(thinking=True)

        # check if valid values are passed
        try:
            assert strength >= 0.01 and strength <= 1.0, "strength should be between 0.01 and 1.0"
            assert steps >= 10 and steps <= 200, "steps should be between 10 and 200"
            assert guidance_scale >= 1 and guidance_scale <= 40, "guidance_scale should be between 1 and 40"
            assert seed is None or (seed >= 0 and seed <= 2 ** 32), "seed should be between 0 and 2 ** 32"
            assert image_resize is None or (image_resize >= 128 and image_resize <= 768), "image_resize should be between 128 and 768"
            assert scheduler in SchedulerType, "scheduler should be one of the following: dpmsolver, lms, euler"
        except AssertionError as e:
            await interaction.followup.send(e)
            return

        if seed is None:
            seed = random.randint(0, 2 ** 32)
            
        if image_resize is None:
            image_resize = 512

        # create prompt
        prompt = Prompt(
            pipe_type=StabilityPipelineType.IMG2IMG,
            prompt=prompt,
            generated_prompt=None if do_imagine_prompt else prompt,
            image_url=image_url,
            image_resize=image_resize,
            strength=strength,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            scheduler=scheduler,
            sd_model_id=SD_MODEL_IDS[0],
            gpt_model_id=os.environ.get('GPT_MODEL_ID')
        )

        # add job
        self.controller.add_job(prompt, interaction)

    def run(self):
        self.controller.set_callback_success_function(self.success)
        self.controller.set_callback_failure_function(self.failure)

        self.controller.start()

        self.commands()
        self.client.run(self.token)
