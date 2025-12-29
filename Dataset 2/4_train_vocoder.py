"""
Step 4: Train Vocoder (HiFi-GAN)
Converts mel spectrograms to audio waveforms
Simplified version optimized for Mac M4
FIXED: Handles variable mel spectrogram sizes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

CONFIG = {
    'base_path': '/Users/ajanshul02gmail.com/T-5/Project/New_version/fma_medium',
    'output_path': './trained_models',
    'sr': 22050,
    'n_mels': 128,
    'n_fft': 2048,
    'hop_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-4,
    'epochs': 50,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'save_every': 10,
    'sample_size': 1000,  # Number of files to use for vocoder training
    'fixed_mel_length': 130,  # Fixed length for mel spectrograms
}

class MelToWaveDataset(Dataset):
    """Dataset for mel spectrogram to waveform training"""
    def __init__(self, base_path, sr, n_mels, n_fft, hop_length, sample_size, fixed_mel_length):
        self.base_path = Path(base_path)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fixed_mel_length = fixed_mel_length
        
        # Get audio files
        audio_files = []
        for folder in sorted(self.base_path.iterdir()):
            if folder.is_dir() and folder.name.isdigit():
                mp3_files = list(folder.glob('*.mp3'))
                audio_files.extend(mp3_files)
        
        # Sample files
        sample_size = min(sample_size, len(audio_files))
        np.random.seed(42)  # For reproducibility
        self.audio_files = np.random.choice(audio_files, sample_size, replace=False)
        print(f"Using {len(self.audio_files)} files for vocoder training")
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        
        try:
            # Load audio with better error handling
            audio, _ = librosa.load(file_path, sr=self.sr, duration=3, res_type='kaiser_fast')
            
            # Fixed length
            target_length = self.sr * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sr, n_fft=self.n_fft,
                hop_length=self.hop_length, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Fix mel spectrogram length to exact size
            if mel_spec_db.shape[1] < self.fixed_mel_length:
                # Pad if too short
                pad_width = self.fixed_mel_length - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='edge')
            elif mel_spec_db.shape[1] > self.fixed_mel_length:
                # Trim if too long
                mel_spec_db = mel_spec_db[:, :self.fixed_mel_length]
            
            # Normalize using global statistics (assuming -80dB to 0dB range)
            mel_spec_db = np.clip(mel_spec_db, -80, 0)
            mel_spec_db = (mel_spec_db + 80) / 80
            
            return torch.FloatTensor(mel_spec_db), torch.FloatTensor(audio)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zeros if error with correct shapes
            return (torch.zeros(self.n_mels, self.fixed_mel_length), 
                    torch.zeros(self.sr * 3))

class ResBlock(nn.Module):
    """Residual block for generator"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)

class Generator(nn.Module):
    """Simplified HiFi-GAN Generator"""
    def __init__(self, n_mels=128):
        super().__init__()
        
        # Initial projection
        self.conv_pre = nn.Conv1d(n_mels, 512, kernel_size=7, padding=3)
        
        # Upsampling blocks
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
        ])
        
        # Residual blocks
        self.resblocks = nn.ModuleList([
            ResBlock(256),
            ResBlock(128),
            ResBlock(64),
            ResBlock(32),
        ])
        
        # Final projection to waveform
        self.conv_post = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
    def forward(self, mel):
        x = self.conv_pre(mel)
        
        for up, resblock in zip(self.ups, self.resblocks):
            x = torch.relu(up(x))
            x = resblock(x)
        
        x = torch.tanh(self.conv_post(x))
        return x.squeeze(1)

class Discriminator(nn.Module):
    """Simple discriminator for GAN training"""
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(128, 256, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 512, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.fc = nn.Linear(512, 1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

def train_vocoder(generator, discriminator, dataloader, 
                 opt_g, opt_d, device, epoch):
    """Train vocoder for one epoch"""
    generator.train()
    discriminator.train()
    
    g_losses = []
    d_losses = []
    
    for mel, audio in tqdm(dataloader, desc=f"Epoch {epoch}"):
        mel = mel.to(device)
        audio = audio.to(device)
        
        # Train Discriminator
        opt_d.zero_grad()
        
        # Generate fake audio
        fake_audio = generator(mel)
        
        # Discriminator on real and fake
        real_pred = discriminator(audio)
        fake_pred = discriminator(fake_audio.detach())
        
        # Discriminator loss
        d_loss = torch.mean((real_pred - 1)**2) + torch.mean(fake_pred**2)
        d_loss.backward()
        opt_d.step()
        
        # Train Generator
        opt_g.zero_grad()
        
        fake_audio = generator(mel)
        fake_pred = discriminator(fake_audio)
        
        # Generator loss (adversarial + L1 reconstruction)
        adv_loss = torch.mean((fake_pred - 1)**2)
        
        # Match lengths for L1 loss
        min_len = min(fake_audio.shape[-1], audio.shape[-1])
        l1_loss = nn.functional.l1_loss(fake_audio[..., :min_len], audio[..., :min_len])
        
        g_loss = adv_loss + 45 * l1_loss  # Weight L1 more
        g_loss.backward()
        opt_g.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
    
    return np.mean(g_losses), np.mean(d_losses)

def main():
    print("="*60)
    print("TRAINING VOCODER (HiFi-GAN)")
    print("="*60)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    output_path = Path(CONFIG['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = MelToWaveDataset(
        CONFIG['base_path'],
        CONFIG['sr'],
        CONFIG['n_mels'],
        CONFIG['n_fft'],
        CONFIG['hop_length'],
        CONFIG['sample_size'],
        CONFIG['fixed_mel_length']
    )
    
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=True, num_workers=0)
    
    # Initialize models
    print("\nInitializing models...")
    generator = Generator(n_mels=CONFIG['n_mels']).to(device)
    discriminator = Discriminator().to(device)
    
    opt_g = optim.AdamW(generator.parameters(), lr=CONFIG['learning_rate'], betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=CONFIG['learning_rate'], betas=(0.5, 0.999))
    
    # Training loop
    print("\nStarting training...")
    g_losses = []
    d_losses = []
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        g_loss, d_loss = train_vocoder(generator, discriminator, dataloader,
                                      opt_g, opt_d, device, epoch)
        g_losses.append(g_loss)
        d_losses.append(d_loss)
        
        print(f"Epoch {epoch}/{CONFIG['epochs']}, G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
        
        # Save checkpoint
        if epoch % CONFIG['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
            }, output_path / f'vocoder_checkpoint_epoch_{epoch}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch}")
    
    # Save final models
    torch.save(generator.state_dict(), output_path / 'final_vocoder.pt')
    print(f"\n✓ Saved final model: final_vocoder.pt")
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(g_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Generator Loss')
    plt.title('Generator Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(d_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Discriminator Loss')
    plt.title('Discriminator Training Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path / 'vocoder_training_losses.png', dpi=150)
    plt.close()
    print(f"✓ Saved training loss plots")
    
    print("\n" + "="*60)
    print("VOCODER TRAINING COMPLETE!")
    print(f"Model saved to: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()