# @Author: Ibrahim Bashir
# @Date: 24 November 2026
# @Version: 1.0.0
# This program compares a stapler photograph with three noise images using DINOv2-small.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# Look for the public sample image in the same folder as this program.
PHOTO_PATH = Path(__file__).with_name("stapler_sample.png")
MODEL_NAME = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
IMAGE_SIZE = 224

# Counting from zero, this position falls on the blue body of the stapler.
REFERENCE_ROW = 8
REFERENCE_COLUMN = 8


def make_noise_image(seed):
    """Generate one noise image. Reusing the seed reproduces the same pixels."""
    generator = np.random.default_rng(seed)

    # Each pixel has three values: red, green and blue.
    # The upper limit is excluded, so 256 gives us values from 0 to 255.
    pixels = generator.integers(
        low=0,
        high=256,
        size=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=np.uint8
    )
    return Image.fromarray(pixels)


def make_similarity_map(image, processor, model):
    """Compare every patch with the reference patch from the same image."""
    # The image is already the correct size. Keeping its dimensions makes
    # positions on the heatmap line up with the original picture.
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False
    )

    with torch.inference_mode():
        output = model(**inputs)

        # Entry zero describes the whole image. The remaining entries
        # describe the individual patches, stored one row after another.
        patches = output.last_hidden_state[0, 1:, :]

        # A 224-pixel image with 14-pixel patches has 16 patches per row.
        grid_size = IMAGE_SIZE // model.config.patch_size
        reference_index = REFERENCE_ROW * grid_size + REFERENCE_COLUMN
        reference_patch = patches[reference_index]

        # A score close to 1 means the two patch descriptions are similar.
        scores = torch.nn.functional.cosine_similarity(
            patches,
            reference_patch.unsqueeze(0),
            dim=1
        )

    # Put the scores back into the same grid arrangement as the image.
    return scores.reshape(grid_size, grid_size).cpu().numpy()


def draw_heatmap(axis, similarity_map):
    """Use identical colour limits so all four maps can be compared."""
    return axis.imshow(
        similarity_map,
        cmap="viridis",
        vmin=-1,
        vmax=1,
        interpolation="nearest"
    )


def main():
    with Image.open(PHOTO_PATH) as original:
        photo = original.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    # Keep the photo first so it appears in the first column of the figure.
    images = [photo]
    titles = ["Stapler"]

    for seed in [1, 2, 3]:
        images.append(make_noise_image(seed))
        titles.append(f"Noise (seed {seed})")

    # Load the trained model once and reuse it for every image.
    print("Loading DINOv2-small...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION, use_safetensors=True)
    model.eval()

    similarity_maps = []
    for title, image in zip(titles, images):
        print(f"Processing {title}...")
        similarity_maps.append(make_similarity_map(image, processor, model))

    # Check the calculation: the reference patch should match itself.
    stapler_map = similarity_maps[0]
    reference_score = stapler_map[REFERENCE_ROW, REFERENCE_COLUMN]
    print(f"Reference patch compared with itself: {reference_score:.4f}")

    # First figure: inspect the stapler beside its own heatmap.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    axes[0].imshow(photo)
    axes[0].set_title("Stapler photo")
    plot = draw_heatmap(axes[1], stapler_map)
    axes[1].set_title("Similarity to reference patch")

    for axis in axes:
        axis.axis("off")

    fig.colorbar(plot, ax=axes[1], label="Cosine similarity")

    # Second figure: place each image directly above its own heatmap.
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")

    for column, image in enumerate(images):
        axes[0, column].imshow(image)
        axes[0, column].set_title(titles[column])
        axes[0, column].axis("off")

        plot = draw_heatmap(axes[1, column], similarity_maps[column])
        axes[1, column].axis("off")

    fig.colorbar(plot, ax=axes[1, :], label="Cosine similarity")
    plt.show()


# Start the experiment when this file is run directly.
if __name__ == "__main__":
    main()
