import torch
import torchvision.utils as vutils
import os

# Load the trained Generator model
from CtScanGan import Generator  # Import from your GAN training script

# Hyperparameters (must match training settings)
nz = 100  # Latent vector size
ngf = 64  # Generator feature maps
nc = 1    # Number of channels (grayscale CT scans)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Generator and load trained weights
generator = Generator(nz, ngf, nc).to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Generate images
def generate_images(num_images=10, output_folder="generated_images"):
    os.makedirs(output_folder, exist_ok=True)
    noise = torch.randn(num_images, nz, 1, 1, device=device)  # Random noise
    with torch.no_grad():
        fake_images = generator(noise).cpu()
    
    # Save images
    for i in range(num_images):
        vutils.save_image(fake_images[i], os.path.join(output_folder, f"fake_ct_{i}.png"), normalize=True)
    
    print(f"Generated {num_images} images in {output_folder}")

# Run the function
generate_images(10)
