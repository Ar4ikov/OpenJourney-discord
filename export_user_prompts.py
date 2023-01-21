import json
import pathlib
import argparse
import pandas as pd


def get_prompt_from_json(prompt_uuid, json_path):
    with open(json_path, 'r') as file:
        print(json_path)
        data = json.loads(file.read())
    
    # make .get to avoid errors (keys from prompt_factory)
    pipe_type = data.get('pipe_type', None)
    prompt = data.get('prompt', None)
    seed = data.get('seed', None)
    sd_model_id = data.get('sd_model_id', None)
    gpt_model_id = data.get('gpt_model_id', None)
    scheduler = data.get('scheduler', None)
    negative_prompt = data.get('negative_prompt', None)
    generated_prompt = data.get('generated_prompt', None)
    image_url = data.get('image_url', None)
    image_resize = data.get('image_resize', None)
    num_samples = data.get('num_samples', None)
    steps = data.get('steps', None)
    guidance_scale = data.get('guidance_scale', None)
    strength = data.get('strength', None)
    grid_cols = data.get('grid_cols', None)
    grid_rows = data.get('grid_rows', None)
    grid_size = data.get('grid_size', None)
    aspect_ratio_sizes = data.get('aspect_ratio_sizes', None)
    use_codeformer = data.get('use_codeformer', None)
    use_face_upscale = data.get('use_face_upscale', None)
    use_background_upscale = data.get('use_background_upscale', None)
    output_filenames = data.get('output_filenames', None)
    upscale_weight = data.get('upscale_weight', None)
    upscale_ratio = data.get('upscale_ratio', None)

    return {
        'prompt_uuid': prompt_uuid,
        'pipe_type': pipe_type,
        'prompt': prompt,
        'seed': seed,
        'sd_model_id': sd_model_id,
        'gpt_model_id': gpt_model_id,
        'scheduler': scheduler,
        'negative_prompt': negative_prompt,
        'generated_prompt': generated_prompt,
        'image_url': image_url,
        'image_resize': image_resize,
        'num_samples': num_samples,
        'steps': steps,
        'guidance_scale': guidance_scale,
        'strength': strength,
        'grid_cols': grid_cols,
        'grid_rows': grid_rows,
        'grid_size': grid_size,
        'aspect_ratio_sizes': aspect_ratio_sizes,
        'use_codeformer': use_codeformer,
        'use_face_upscale': use_face_upscale,
        'use_background_upscale': use_background_upscale,
        'output_filenames': output_filenames,
        'upscale_weight': upscale_weight,
        'upscale_ratio': upscale_ratio
    }


def get_prompts_from_folder(folder_path):
    prompts = []
    for uuid_path in pathlib.Path(folder_path).iterdir():
        json_path = uuid_path / 'prompt.json'
        if json_path.exists():
            try:
                prompt = get_prompt_from_json(uuid_path.name, json_path)
            except json.decoder.JSONDecodeError:
                print(f'JSONDecodeError in {uuid_path}')
                continue
            else:
                prompts.append(prompt)
        else:
            print(f'No prompt.json in {uuid_path}')
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description='Diffusers Prompt Parser')
    parser.add_argument('--input_path', type=str, default=None, help='Path to folder with outputs')
    parser.add_argument('--output_path', type=str, default=None, help='Path to output file')

    args = parser.parse_args()

    prompts = get_prompts_from_folder(args.input_path)
    df = pd.DataFrame(prompts)
    df.to_csv(args.output_path, index=False, sep=';')


if __name__ == '__main__':
    main()
