import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import random
from pathlib import Path

def load_image(image_path, resolution=256):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0)

def get_random_images_from_folder(folder_path, num_images=2):
    """Get random image paths from a folder."""
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(folder_path).rglob(f'*{ext}')))

    if len(image_files) < num_images:
        raise ValueError(f"Found only {len(image_files)} images in {folder_path}, need at least {num_images}")

    selected = random.sample(image_files, num_images)
    return [str(path) for path in selected]

@torch.no_grad()
def encode_image(vae, image, latents_scale, latents_bias):
    """Encode image to latent space."""
    d = vae.encode(image).latent_dist
    moments = torch.cat([d.mean, d.std], dim=1)

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = z * latents_scale + latents_bias
    return z

@torch.no_grad()
def decode_latent(vae, latent, latents_scale, latents_bias):
    """Decode latent to image."""
    latent = (latent - latents_bias) / latents_scale
    image = vae.decode(latent).sample
    return image

def interpolate_latents(latent1, latent2, alpha):
    """Interpolate between two latents with alpha."""
    return (1 - alpha) * latent1 + alpha * latent2

def save_image(tensor, path):
    """Save a tensor as an image."""
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0, 1)

    # Convert to numpy
    image = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)

    # Save
    Image.fromarray(image).save(path)

def create_interpolation_grid(images, output_path, num_steps):
    """Create a grid of interpolated images."""
    fig, axes = plt.subplots(1, num_steps + 2, figsize=(3 * (num_steps + 2), 3))

    for idx, img_tensor in enumerate(images):
        # Convert from [-1, 1] to [0, 1]
        img = (img_tensor + 1) / 2.0
        img = img.clamp(0, 1)
        img = img.cpu().squeeze(0).permute(1, 2, 0).numpy()

        axes[idx].imshow(img)
        axes[idx].axis('off')
        if idx == 0:
            axes[idx].set_title('Image 1', fontsize=12)
        elif idx == len(images) - 1:
            axes[idx].set_title('Image 2', fontsize=12)
        else:
            alpha = (idx) / (len(images) - 1)
            axes[idx].set_title(f'α={alpha:.2f}', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved interpolation grid to {output_path}")

def main(args):
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Setup latent scaling
    latents_scale = torch.tensor([0.18215, 0.18215, 0.18215, 0.18215]).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor([0., 0., 0., 0.]).view(1, 4, 1, 1).to(device)

    # Determine image paths
    if args.data_dir:
        print(f"Selecting random images from {args.data_dir}...")
        image_paths = get_random_images_from_folder(args.data_dir, num_images=2)
        image1_path = image_paths[0]
        image2_path = image_paths[1]
        print(f"Selected image 1: {image1_path}")
        print(f"Selected image 2: {image2_path}")
    else:
        if not args.image1 or not args.image2:
            raise ValueError("Either --data-dir or both --image1 and --image2 must be provided")
        image1_path = args.image1
        image2_path = args.image2

    # Load images
    print(f"Loading image 1: {image1_path}")
    img1 = load_image(image1_path, args.resolution).to(device)
    print(f"Loading image 2: {image2_path}")
    img2 = load_image(image2_path, args.resolution).to(device)

    # Encode images to latent space
    print("Encoding images to latent space...")
    latent1 = encode_image(vae, img1, latents_scale, latents_bias)
    latent2 = encode_image(vae, img2, latents_scale, latents_bias)

    print(f"Latent shape: {latent1.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Perform interpolation
    print(f"Interpolating with {args.num_steps} steps...")
    alphas = np.linspace(0, 1, args.num_steps + 2)

    interpolated_images = []

    for idx, alpha in enumerate(alphas):
        # Interpolate latents
        interpolated_latent = interpolate_latents(latent1, latent2, alpha)

        # Decode to image
        interpolated_img = decode_latent(vae, interpolated_latent, latents_scale, latents_bias)
        interpolated_images.append(interpolated_img)

        # Save individual image
        save_path = os.path.join(args.output_dir, f"interpolated_{idx:03d}_alpha_{alpha:.3f}.png")
        save_image(interpolated_img, save_path)
        print(f"  Saved: {save_path} (α={alpha:.3f})")

    # Create and save grid
    grid_path = os.path.join(args.output_dir, "interpolation_grid.png")
    create_interpolation_grid(interpolated_images, grid_path, args.num_steps)

    print("\nInterpolation completed!")
    print(f"Results saved to: {args.output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image Interpolation in VAE Latent Space")

    parser.add_argument("--image1", type=str, default=None,
                        help="Path to first image")
    parser.add_argument("--image2", type=str, default=None,
                        help="Path to second image")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory to randomly select 2 images from (alternative to --image1/--image2)")
    parser.add_argument("--output-dir", type=str, default="interpolation_results",
                        help="Directory to save results")
    parser.add_argument("--resolution", type=int, default=256,
                        help="Image resolution")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of interpolation steps (excluding start and end)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for image selection")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)



# CUDA_VISIBLE_DEVICES=6 python test_interpolate.py \
#   --image1 /v/mnt/GH/imagenet_data/val/n01494475/ILSVRC2012_val_00024015.JPEG \
#   --image2 /v/mnt/GH/imagenet_data/val/n01608432/ILSVRC2012_val_00015040.JPEG \
#   --num-steps 5 \
#   --resolution 256 \
#   --output-dir ./interpolation

## random 2 images from folder
# CUDA_VISIBLE_DEVICES=6 python test_interpolate.py \
#   --num-steps 5 \
#   --resolution 256 \
#   --output-dir ./interpolation
