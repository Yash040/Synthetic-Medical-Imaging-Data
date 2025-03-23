import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class MRI_GAN:
    def __init__(self, input_shape=(128, 128, 128, 1), latent_dim=200):
        """
        Initialize the 3D MRI GAN
        
        Args:
            input_shape: Shape of the MRI volumes (default: 128x128x128 with 1 channel)
            latent_dim: Dimension of the latent space (default: 200)
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            metrics=['accuracy']
        )
        
        # Build the generator
        self.generator = self.build_generator()
        
        # The generator takes noise as input and generates MRI scans
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        
        # For the combined model, we only train the generator
        self.discriminator.trainable = False
        
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        
        # The combined model (stacked generator and discriminator)
        self.combined = models.Model(z, valid)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        )
        
    def build_generator(self):
        """
        Build the generator network
        
        Returns:
            A Keras Model representing the generator
        """
        noise = layers.Input(shape=(self.latent_dim,))
        
        # Initial size of feature maps
        init_depth = 64
        s = self.input_shape[0] // 16  # Starting size for each dimension
        
        # Foundation for 3D volume
        x = layers.Dense(s * s * s * init_depth * 8)(noise)
        x = layers.Reshape((s, s, s, init_depth * 8))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Transposed convolution layers to increase resolution
        # 8x8x8 -> 16x16x16
        x = layers.Conv3DTranspose(init_depth * 4, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # 16x16x16 -> 32x32x32
        x = layers.Conv3DTranspose(init_depth * 2, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # 32x32x32 -> 64x64x64
        x = layers.Conv3DTranspose(init_depth, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # 64x64x64 -> 128x128x128
        x = layers.Conv3DTranspose(init_depth // 2, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        
        # Output layer
        x = layers.Conv3D(1, (3, 3, 3), padding='same', activation='tanh')(x)
        
        return models.Model(noise, x)
    
    def build_discriminator(self):
        """
        Build the discriminator network
        
        Returns:
            A Keras Model representing the discriminator
        """
        img = layers.Input(shape=self.input_shape)
        
        # Feature extraction layers
        x = layers.Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same')(img)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv3D(256, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv3D(512, (4, 4, 4), strides=(2, 2, 2), padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Dropout(0.25)(x)
        
        # Output layer
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(img, x)
    
    def load_and_preprocess_data(self, data_dir, target_shape=None, max_samples=None):
        """
        Load and preprocess MRI data from .nii files
        
        Args:
            data_dir: Directory containing .nii files
            target_shape: Target shape for resizing (default: None, uses self.input_shape)
            max_samples: Maximum number of samples to load (default: None, loads all)
            
        Returns:
            Normalized and preprocessed MRI data as numpy array
        """
        if target_shape is None:
            target_shape = self.input_shape[:3]
            
        # Get all .nii files in the directory
        nii_files = glob(os.path.join(data_dir, '*.nii*'))
        if max_samples:
            nii_files = nii_files[:max_samples]
            
        print(f"Loading {len(nii_files)} .nii files...")
        
        mri_data = []
        for nii_file in tqdm(nii_files):
            try:
                # Load NIfTI file
                img = nib.load(nii_file)
                data = img.get_fdata()
                
                # Resize to target shape if needed
                if data.shape[:3] != target_shape:
                    # Simple resizing approach (you might want to use better methods)
                    from scipy.ndimage import zoom
                    zoom_factors = [t / s for t, s in zip(target_shape, data.shape[:3])]
                    data = zoom(data, zoom_factors + [1] * (len(data.shape) - 3))
                
                # Handle different dimensions
                if len(data.shape) == 3:
                    data = data[..., np.newaxis]  # Add channel dimension
                elif len(data.shape) > 4:
                    data = data[..., 0:1]  # Take first channel
                
                # Normalize to [-1, 1]
                data = (data - data.min()) / (data.max() - data.min()) * 2 - 1
                
                mri_data.append(data)
            except Exception as e:
                print(f"Error loading {nii_file}: {e}")
                
        return np.array(mri_data)
    
    def train(self, data, epochs, batch_size=4, save_interval=50, checkpoint_dir='checkpoints'):
        """
        Train the GAN
        
        Args:
            data: Training data
            epochs: Number of epochs
            batch_size: Batch size
            save_interval: Interval to save generated samples
            checkpoint_dir: Directory to save model checkpoints
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, 'samples'), exist_ok=True)
        
        # Labels for real and fake samples
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        for epoch in range(epochs):
            # ---------------------
            # Train Discriminator
            # ---------------------
            
            # Select a random batch of real MRIs
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_mris = data[idx]
            
            # Generate a batch of fake MRIs
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_mris = self.generator.predict(noise)
            
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_mris, real)
            d_loss_fake = self.discriminator.train_on_batch(fake_mris, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            # Train Generator
            # ---------------------
            
            # Generate new noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, real)
            
            # Print progress
            print(f"{epoch}/{epochs} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss}]")
            
            # Save generated samples and model at specified intervals
            if epoch % save_interval == 0:
                self.save_samples(epoch, checkpoint_dir)
                self.save_models(epoch, checkpoint_dir)
    
    def save_samples(self, epoch, checkpoint_dir):
        """
        Save generated samples
        
        Args:
            epoch: Current epoch
            checkpoint_dir: Directory to save samples
        """
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_mris = self.generator.predict(noise)
        
        # Rescale to [0, 1]
        gen_mris = 0.5 * gen_mris + 0.5
        
        fig, axs = plt.subplots(r, c, figsize=(12, 12))
        cnt = 0
        
        for i in range(r):
            for j in range(c):
                # Show middle slice of each volume in each dimension
                mri = gen_mris[cnt, :, :, :, 0]
                axs[i, j].imshow(mri[mri.shape[0]//2, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
                
        fig.savefig(os.path.join(checkpoint_dir, f'samples/mri_{epoch}.png'))
        plt.close()
        
        # Save a full volume as .nii
        sample_mri = gen_mris[0, :, :, :, 0]
        nii_img = nib.Nifti1Image(sample_mri, np.eye(4))
        nib.save(nii_img, os.path.join(checkpoint_dir, f'samples/mri_{epoch}.nii.gz'))
    
    def save_models(self, epoch, checkpoint_dir):
        """
        Save models
        
        Args:
            epoch: Current epoch
            checkpoint_dir: Directory to save models
        """
        self.generator.save(os.path.join(checkpoint_dir, f'generator_{epoch}.h5'))
        self.discriminator.save(os.path.join(checkpoint_dir, f'discriminator_{epoch}.h5'))
        
    def generate_samples(self, num_samples, output_dir='generated'):
        """
        Generate and save multiple synthetic MRI samples
        
        Args:
            num_samples: Number of samples to generate
            output_dir: Directory to save generated samples
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating {num_samples} synthetic MRI samples...")
        
        for i in tqdm(range(num_samples)):
            # Generate random noise
            noise = np.random.normal(0, 1, (1, self.latent_dim))
            
            # Generate fake MRI
            gen_mri = self.generator.predict(noise)[0]
            
            # Rescale from [-1, 1] to original intensity range
            gen_mri = 0.5 * gen_mri + 0.5
            
            # Create NIfTI image and save
            nii_img = nib.Nifti1Image(gen_mri[:, :, :, 0], np.eye(4))
            nib.save(nii_img, os.path.join(output_dir, f'synthetic_mri_{i}.nii.gz'))

# Usage example
if __name__ == "__main__":
    # Initialize the GAN
    input_shape = (128, 128, 128, 1)  # Adjust based on your MRI dimensions
    mri_gan = MRI_GAN(input_shape=input_shape, latent_dim=200)
    
    # Load and preprocess data
    data = mri_gan.load_and_preprocess_data(data_dir='/Users/Admin/Documents/Datasets/CirrMri600+/Cirrhosis_T2_3D/Grouped Scans/Mild/images', max_samples=100)
    
    # Train the GAN
    mri_gan.train(data, epochs=100, batch_size=4, save_interval=100)
    
    # Generate synthetic samples
    mri_gan.generate_samples(num_samples=50, output_dir='synthetic_mri_dataset')
