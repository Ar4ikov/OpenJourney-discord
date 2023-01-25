# OpenJourney-discord

![build](/../../../OpenJourney-discord/actions/workflows/docker-image.yml/badge.svg)<br>
![image](/../../../../Ar4ikov/animated-spork/blob/main/spiral.gif)

This is the official repository for the OpenJourney Discord bot.

## What is OpenJourney?

OpenJourney is a Discord bot that allows you to create your own generated pictures by using [StableDiffusion](https://github.com/huggingface/diffusers)

## How to use?

To use OpenJourney, you need to have a Discord account. If you don't have one, you can create one [here](https://discord.com/register).

Once you have a Discord account, you can invite OpenJourney to your server by clicking [here](https://discord.com/oauth2/authorize?client_id=1057463364848209981&permissions=534723950656&scope=bot) (Works for all servers for 1 week from publishing)

## How to use the bot?

After adding bot just type `/help` to get started. <br>
Or just to official guide page in Notion: [OpenJourney Guide](https://button-aurora-7b8.notion.site/OpenJourney-Guide-05401d7a438e4e5cb2cbb241a15d6bdf)

## How to contribute?

If you want to contribute to OpenJourney, you can fork this repository and make a pull request. If you want to add a new feature, please open an issue first. Name your forked repository `OpenJourney-discord-<feature>`. For example, if you want to add a new command, name your forked repository `OpenJourney-discord-new-command`.

## Installation

### Index

1. [Create a Discord Application & Bot, Invite to your server](#0-create-a-discord-application--bot-invite-to-your-server)
2. [Clone the repository](#1-clone-the-repository)
3. [Install NVIDIA Runtime](#2-install-nvidia-runtime)
4. [Install Docker and docker-compose (optional)](#3-install-docker-and-docker-compose-optional)
5. [Install Nvidia Docker](#4-install-nvidia-docker)
6. [Setup the environment](#5-setup-the-environment)
7. [Build the image](#6-build-the-image)
8. [Run the container](#70-run-the-container)


### 0. Create a Discord Application & Bot, Invite to your server

1. Go to [Discord Developer Portal](https://discord.com/developers/applications) and create a new application
2. Go to the `Bot` tab and create a new bot
3. Go to the `OAuth2` tab and select `bot` scope
4. Select the permissions you want to give to the bot
5. Copy the link and paste it in your browser
6. Select the server you want to add the bot to

### 1. Clone the repository

Install git, curl, if you don't have them

```bash
sudo apt install git curl
```

```bash
git clone https://github.com/Ar4ikov/OpenJourney-discord.git
cd OpenJourney-discord
```

### 2. Install NVIDIA Runtime

Do it like you would do it for any other project (for Windows, Linux, MacOS) <br>
There is a simple example how to install it with Conda for Linux

```bash
conda install cudatoolkit=11.6 -c nvidia
```

### 3. Install Docker and docker-compose (optional)

```bash
sudo apt update && sudo apt install docker.io docker-compose
```

Or install docker and docker-ce that way:

```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
```

### 4. Install Nvidia Docker

Link to: [NVIDIA Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

1. Setup a package repository and the GPG key

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

2. Update the package repository

```bash
sudo apt-get update
```

3. Install the nvidia-docker-2

```bash
sudo apt-get install -y nvidia-docker2
```

4. Change the default runtime to nvidia [Link](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime)

```bash
sudo nano /etc/docker/daemon.json
```

It should looks like:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```

5. Restart the Docker daemon

```bash
sudo systemctl restart docker
```

At this point, you should be able to run the nvidia-docker2 container.

```bash
sudo docker run --rm --gpus all nvidia/cuda:11.6.0-base-ubuntu20.04 nvidia-smi
```

This should output the following:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.51.06    Driver Version: 450.51.06    CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### 5. Setup the environment

| Environment variable | Description | Default value |
| --- | --- | --- |
| `DISCORD_TOKEN` | Discord bot token | `None` |
| `GUILD_ID` | Discord server ID, -1 if sync commans globally | `-1` |
| `SD_MODEL_ID_1` | StableDiffusion model ID | `dreamlike-art/dreamlike-photoreal-2.0` |
| `GPT_MODEL_ID` | GPT-2 model ID for Magic Prompt generate | `Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator` |
| `NUM_GPUS` | Number of GPUs to use | `1` |
| `NUM_THREADS_PER_GPU` | Number of threads per GPU | `2` |
| `NSFW_GENERATE` | Allow NSFW generate images content | `True` |

You can use multiple models for StableDiffusion, just add `SD_MODEL_ID_2`, `SD_MODEL_ID_3` and so on

```bash
cp .env_example .env
nano .env
```

### 6. Build the image

```bash
source .env && docker-compose build
```

### 7.0. Run the container

```bash
source .env && docker-compose up -d
```

### 7.1. Stop the container

```bash
docker-compose down
```

## Technical details

For every GPU that use FP16, there are some calculations:
* ±16-20 GB of RAM (per 1 GPU & 2 threads)
* ±2 GB of VRAM **in background** (per 1 GPU & 2 threads)
* ±6 GB of VRAM **per thread in active stage** (per 1 GPU & 1 threads)
* ±12 GB of VRAM **per thread in active stage** (per 1 GPU & 2 threads)

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Credits

* [Dreamlike Art Model](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0)
* [StableDiffusion Prompt Generator](https://huggingface.co/Ar4ikov/gpt2-650k-stable-diffusion-prompt-generator)
* [CodeFormer Face Upsampler](https://github.com/sczhou/CodeFormer)
