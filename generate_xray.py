import matplotlib.pyplot as plt
import numpy as np

# Generate a batch of images
num_samples = 5  # Change this to generate more images
z = torch.randn(num_samples, latent_dim).to(device)
fake_imgs = generator(z).detach().cpu().numpy()

# Plot the generated images
fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
for i in range(num_samples):
    img = np.squeeze(fake_imgs[i])  # Remove extra dimension
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")

plt.show()
