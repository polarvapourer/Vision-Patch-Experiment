# Stapler patch similarity experiment

I used DINOv2-small to compare a photograph of a stapler with three
random-noise images. For each image, the program compares one patch near
the centre with every other patch and displays the scores as a heatmap.

In the photograph, a bright horizontal band roughly followed the stapler.
In the noise images, high scores spread across much of the frame. All four
maps use the same colour scale.

## Run on Windows

I ran the experiment with Python 3.13 and CPU PyTorch. Open a terminal in
the project folder and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install transformers pillow numpy matplotlib
.\.venv\Scripts\python.exe .\stapler_patch_experiment.py
```

The sample photo is included. The first run downloads the model.

Inspired by [Modern Self-Supervised Vision Transformers Are Pareidolic](https://arxiv.org/abs/2611.31238).