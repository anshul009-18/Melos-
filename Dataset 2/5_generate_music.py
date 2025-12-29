"""
Step 5: Music Generation (Inference)
Generate music from emotion inputs using trained models
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import math

CONFIG = {
    'models_path': './trained_models',
    'output_path': './generated_music',
    'latent_dim': 128,
    'emotion_dim': 64,
    'n_emotions': 8,
    'timesteps': 1000,
    'sr': 22050,
    'n_mels': 128,
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
}

EMOTIONS = ['happy', 'sad', 'energetic', 'calm', 'dark', 'bright', 'aggressive', 'peaceful']

# ===== Model Architectures (copied from training scripts) =====

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DiffusionUNet(nn.Module):
    def __init__(self, latent_dim=128, emotion_dim=64, n_emotions=8, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.emotion_embed = nn.Embedding(n_emotions, emotion_dim)
        self.input_proj = nn.Linear(latent_dim + time_emb_dim + emotion_dim, 512)
        self.encoder = nn.Sequential(
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.1),
        )
        self.output_proj = nn.Linear(512, latent_dim)
        
    def forward(self, x, t, emotion):
        t_emb = self.time_mlp(t)
        e_emb = self.emotion_embed(emotion).squeeze(1)
        x = torch.cat([x, t_emb, e_emb], dim=-1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, output_length=1292):
        super(VAEDecoder, self).__init__()
        self.output_length = output_length
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512 * 16), nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 16)
        x = self.deconv_layers(x)
        x = torch.nn.functional.interpolate(x, size=self.output_length, mode='linear')
        x = x.unsqueeze(1)
        return x

class ResBlock(nn.Module):
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
    def __init__(self, n_mels=128):
        super().__init__()
        self.conv_pre = nn.Conv1d(n_mels, 512, kernel_size=7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
        ])
        self.resblocks = nn.ModuleList([
            ResBlock(256), ResBlock(128), ResBlock(64), ResBlock(32),
        ])
        self.conv_post = nn.Conv1d(32, 1, kernel_size=7, padding=3)
        
    def forward(self, mel):
        x = self.conv_pre(mel)
        for up, resblock in zip(self.ups, self.resblocks):
            x = torch.relu(up(x))
            x = resblock(x)
        x = torch.tanh(self.conv_post(x))
        return x.squeeze(1)

# ===== Diffusion Sampling =====

@torch.no_grad()
def p_sample(model, x, t, t_index, emotion, noise_schedule, device):
    """Sample from p(x_{t-1} | x_t)"""
    betas = noise_schedule['betas'].to(device)
    alphas = noise_schedule['alphas'].to(device)
    alphas_cumprod = noise_schedule['alphas_cumprod'].to(device)
    alphas_cumprod_prev = noise_schedule['alphas_cumprod_prev'].to(device)
    
    # Predict noise
    predicted_noise = model(x, t, emotion)
    
    # Compute coefficients
    alpha_t = alphas[t_index]
    alpha_t_cumprod = alphas_cumprod[t_index]
    alpha_t_cumprod_prev = alphas_cumprod_prev[t_index]
    beta_t = betas[t_index]
    
    # Compute mean
    model_mean = (1. / torch.sqrt(alpha_t)) * (
        x - (beta_t / torch.sqrt(1. - alpha_t_cumprod)) * predicted_noise
    )
    
    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        variance = beta_t * (1. - alpha_t_cumprod_prev) / (1. - alpha_t_cumprod)
        return model_mean + torch.sqrt(variance) * noise

@torch.no_grad()
def sample_music_latent(model, emotion_id, noise_schedule, device, 
                       latent_dim=128, timesteps=1000):
    """Generate latent vector using reverse diffusion"""
    model.eval()
    
    # Start from random noise
    x = torch.randn(1, latent_dim).to(device)
    emotion = torch.tensor([[emotion_id]]).to(device)
    
    # Reverse diffusion process
    for t_idx in reversed(range(timesteps)):
        t = torch.full((1,), t_idx, device=device, dtype=torch.long)
        x = p_sample(model, x, t, t_idx, emotion, noise_schedule, device)
    
    return x

# ===== Main Generation Pipeline =====

class MusicGenerator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.output_path = Path(config['output_path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        print("Loading models...")
        self.load_models()
        
    def load_models(self):
        """Load all trained models"""
        models_path = Path(self.config['models_path'])
        
        # Load diffusion model
        self.diffusion_model = DiffusionUNet(
            latent_dim=self.config['latent_dim'],
            emotion_dim=self.config['emotion_dim'],
            n_emotions=self.config['n_emotions']
        ).to(self.device)
        self.diffusion_model.load_state_dict(
            torch.load(models_path / 'final_diffusion.pt', map_location=self.device)
        )
        self.diffusion_model.eval()
        
        # Load noise schedule
        self.noise_schedule = torch.load(
            models_path / 'noise_schedule.pt', 
            map_location=self.device
        )
        
        # Load VAE decoder
        self.decoder = VAEDecoder(
            latent_dim=self.config['latent_dim'],
            output_length=1292
        ).to(self.device)
        self.decoder.load_state_dict(
            torch.load(models_path / 'final_decoder.pt', map_location=self.device)
        )
        self.decoder.eval()
        
        # Load vocoder
        self.vocoder = Generator(n_mels=self.config['n_mels']).to(self.device)
        self.vocoder.load_state_dict(
            torch.load(models_path / 'final_vocoder.pt', map_location=self.device)
        )
        self.vocoder.eval()
        
        print("All models loaded successfully!")
    
    def generate(self, emotion_name, filename=None):
        """Generate music for given emotion"""
        if emotion_name not in EMOTIONS:
            raise ValueError(f"Emotion must be one of {EMOTIONS}")
        
        emotion_id = EMOTIONS.index(emotion_name)
        print(f"\nGenerating music for emotion: {emotion_name}")
        
        # Step 1: Generate latent vector
        print("Step 1/3: Generating latent representation...")
        latent = sample_music_latent(
            self.diffusion_model,
            emotion_id,
            self.noise_schedule,
            self.device,
            self.config['latent_dim'],
            self.config['timesteps']
        )
        
        # Step 2: Decode to mel spectrogram
        print("Step 2/3: Decoding to mel spectrogram...")
        with torch.no_grad():
            mel_spec_vae = self.decoder(latent) # Output in [-1, 1]
        
        mel_spec_vae = mel_spec_vae.squeeze(0).squeeze(0)
        
        # Denormalize for Vocoder and Griffin-Lim
        # VAE output is [-1, 1]. Map to [0, 1] for Vocoder (if trained on [0, 1])
        mel_spec_norm = (mel_spec_vae + 1) / 2
        
        # Map to dB for Griffin-Lim [-80, 0]
        mel_spec_db = mel_spec_norm * 80 - 80
        mel_spec_db_np = mel_spec_db.cpu().numpy()
        
        # Convert to linear for Griffin-Lim
        mel_spec_linear = librosa.db_to_power(mel_spec_db_np)
        
        # Step 3: Generate waveform
        print("Step 3/3: Generating audio waveform...")
        
        # Method A: Neural Vocoder
        print("  > Using HiFi-GAN Vocoder...")
        with torch.no_grad():
            # Vocoder expects [0, 1] input based on our training
            waveform_vocoder = self.vocoder(mel_spec_norm.unsqueeze(0).unsqueeze(0))
        waveform_vocoder = waveform_vocoder.cpu().numpy().squeeze()
        
        # Method B: Griffin-Lim (Fallback)
        print("  > Using Griffin-Lim Algorithm (Fallback)...")
        waveform_gl = librosa.griffinlim(
            mel_spec_linear, 
            n_iter=32, 
            hop_length=512, 
            win_length=2048
        )
        
        # Save audio
        if filename is None:
            base_filename = f"{emotion_name}_music"
        else:
            base_filename = filename.replace('.wav', '')
        
        # Save Vocoder output
        output_file_voc = self.output_path / f"{base_filename}_vocoder.wav"
        sf.write(output_file_voc, waveform_vocoder, self.config['sr'])
        
        # Save Griffin-Lim output
        output_file_gl = self.output_path / f"{base_filename}_griffin_lim.wav"
        sf.write(output_file_gl, waveform_gl, self.config['sr'])
        
        print(f"✓ Saved Vocoder output: {output_file_voc}")
        print(f"✓ Saved Griffin-Lim output: {output_file_gl}")
        
        # Visualize mel spectrogram
        self.visualize_generation(mel_spec_db_np, emotion_name, base_filename)
        
        return waveform_vocoder, output_file_voc
    
    def visualize_generation(self, mel_spec_db, emotion_name, filename):
        """Visualize the generated mel spectrogram"""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        # mel_spec_db should be in dB scale now
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Generated Mel Spectrogram - {emotion_name}')
        plt.ylabel('Mel Frequency Bins')
        plt.xlabel('Time Frames')
        
        plt.subplot(1, 2, 2)
        plt.plot(mel_spec_db.mean(axis=0))
        plt.title('Average dB Energy Over Time')
        plt.xlabel('Time Frames')
        plt.ylabel('Average Magnitude (dB)')
        plt.grid(True)
        
        plt.tight_layout()
        vis_file = self.output_path / f"{filename}_visualization.png"
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualization saved to: {vis_file}")

def main():
    print("="*60)
    print("EMOTION-BASED MUSIC GENERATION")
    print("="*60)
    
    generator = MusicGenerator(CONFIG)
    
    print("\nAvailable emotions:", ", ".join(EMOTIONS))
    
    # Generate music for each emotion
    print("\nGenerating samples for all emotions...")
    for emotion in EMOTIONS:
        try:
            generator.generate(emotion)
        except Exception as e:
            print(f"Error generating {emotion}: {e}")
    
    print("\n" + "="*60)
    print("MUSIC GENERATION COMPLETE!")
    print(f"Generated files saved to: {CONFIG['output_path']}")
    print("="*60)
    
    # Interactive mode
    print("\n--- Interactive Mode ---")
    print("Enter an emotion to generate music (or 'quit' to exit)")
    while True:
        emotion = input(f"\nEmotion ({', '.join(EMOTIONS)}): ").strip().lower()
        if emotion == 'quit':
            break
        if emotion in EMOTIONS:
            try:
                generator.generate(emotion)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print(f"Invalid emotion. Choose from: {', '.join(EMOTIONS)}")

if __name__ == "__main__":
    main()