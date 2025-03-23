import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Load the trained generator
generator.load_state_dict(torch.load("saved_models/generator.pth"))
generator.eval()

# Generate noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise = torch.randn(16, 100, 1, 1, device=device)  # Generate 16 fake images

# Generate images
with torch.no_grad():
    fake_images = generator(noise).cpu()

# Plot images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated CT Scan Images")
plt.imshow(vutils.make_grid(fake_images, padding=2, normalize=True).permute(1, 2, 0))
plt.show()
