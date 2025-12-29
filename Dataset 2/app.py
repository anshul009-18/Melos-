import numpy as np
from scipy import signal
import soundfile as sf
from datetime import datetime
import os, warnings
warnings.filterwarnings("ignore")

class ProSynth:
    def __init__(self, sr=48000):
        self.sr = sr
    
    def adsr(self, n, a, d, s, r):
        a_n = max(1, int(a * self.sr))
        d_n = max(1, int(d * self.sr))
        r_n = max(1, int(r * self.sr))
        s_n = max(0, n - a_n - d_n - r_n)
        
        att = 1 - np.exp(-5 * np.linspace(0, 1, a_n))
        dec = s + (1-s) * np.exp(-4 * np.linspace(0, 1, d_n))
        sus = np.ones(s_n) * s
        rel = s * np.exp(-5 * np.linspace(0, 1, r_n))
        
        env = np.concatenate([att, dec, sus, rel])[:n]
        fade = min(50, n//20)
        env[:fade] *= np.linspace(0, 1, fade)
        env[-fade:] *= np.linspace(1, 0, fade)
        return env
    
    def filt(self, x, cutoff, res=0.5):
        nyq = self.sr / 2
        cutoff = min(cutoff, nyq * 0.98)
        sos = signal.butter(4, cutoff/nyq, 'low', output='sos')
        y = signal.sosfiltfilt(sos, x)
        if res > 0:
            sos_r = signal.butter(2, cutoff/nyq, 'low', output='sos')
            y += signal.sosfiltfilt(sos_r, x) * res * 0.12
        return y
    
    def piano(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        y = np.sin(2*np.pi*f*t)
        harms = [(2, 0.75, 8), (3, 0.55, 12), (4, 0.35, 16), (5, 0.22, 20), (6, 0.14, 24)]
        for h, amp, decay in harms:
            y += amp * np.sin(2*np.pi*f*h*t) * np.exp(-decay*t)
        
        y = self.filt(y, 4200 + v*25, 0.3)
        y *= self.adsr(n, 0.001, 0.1, 0.6, 0.25)
        return y * (v/127) * 0.42
    
    def strings(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        voices = []
        for det in [-0.006, -0.003, 0, 0.003, 0.006]:
            voice = sum(np.sin(2*np.pi*f*(1+det)*i*t)/i for i in range(1, 7))
            voices.append(voice)
        
        y = np.mean(voices, axis=0) / 2.5
        
        vib = 1 + 0.007 * np.sin(2*np.pi*5.5*t) * np.minimum(1, t*3)
        y *= vib
        
        y = self.filt(y, 6500, 0.5)
        y *= self.adsr(n, 0.12, 0.25, 0.85, 0.4)
        return y * (v/127) * 0.36
    
    def bass(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        y = (np.sin(2*np.pi*f*0.5*t) * 0.4 + 
             np.sin(2*np.pi*f*t) + 
             np.sin(2*np.pi*f*2*t) * 0.4 + 
             np.sin(2*np.pi*f*3*t) * 0.2)
        
        y = self.filt(y, 1500, 0.8)
        y *= self.adsr(n, 0.002, 0.06, 0.88, 0.12)
        return y * (v/127) * 0.58
    
    def pad(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        layers = []
        for det in [1.0, 1.005, 0.995, 1.01, 0.99]:
            layer = sum(np.sin(2*np.pi*f*det*i*t)/(i*0.8) for i in range(1, 5))
            layers.append(layer)
        
        y = np.mean(layers, axis=0) / 2
        
        lfo = (1 + 0.12*np.sin(2*np.pi*0.13*t)) * (1 + 0.08*np.sin(2*np.pi*0.21*t))
        y *= lfo
        
        y = self.filt(y, 4500, 0.6)
        y *= self.adsr(n, 0.3, 0.5, 0.8, 0.7)
        return y * (v/127) * 0.3
    
    def lead(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        y1 = sum(np.sin(2*np.pi*f*i*t)/i for i in range(1, 11))
        y2 = sum(np.sin(2*np.pi*f*1.003*i*t)/i for i in range(1, 11))
        y = (y1 + y2) / 2
        
        y = self.filt(y, 8000, 0.4)
        y *= self.adsr(n, 0.01, 0.12, 0.75, 0.2)
        return y * (v/127) * 0.4
    
    def arp(self, f, dur, v):
        n = int(dur * self.sr)
        t = np.linspace(0, dur, n, False)
        
        y = (np.sin(2*np.pi*f*t) + 
             np.sin(2*np.pi*f*2*t)*0.5 + 
             np.sin(2*np.pi*f*3*t)*0.3)
        
        y = self.filt(y, 9000, 0.3)
        y *= self.adsr(n, 0.001, 0.05, 0.3, 0.08)
        return y * (v/127) * 0.38
    
    def kick(self, v):
        n = int(0.6 * self.sr)
        t = np.linspace(0, 0.6, n, False)
        
        pitch = 180 * np.exp(-18*t) + 45
        phase = np.cumsum(pitch/self.sr) * 2*np.pi
        
        y = np.sin(phase) + np.sin(phase*0.5)*0.6
        y *= np.exp(-14*t)
        
        click_n = int(0.01*self.sr)
        click = np.random.randn(click_n) * 0.2 * np.exp(-120*np.linspace(0,0.01,click_n))
        y[:click_n] += click
        
        return y * (v/127) * 0.72
    
    def snare(self, v):
        n = int(0.2 * self.sr)
        t = np.linspace(0, 0.2, n, False)
        
        noise = np.random.randn(n) * 0.8
        tone = np.sin(2*np.pi*200*t)*0.4 + np.sin(2*np.pi*350*t)*0.3
        
        y = noise + tone
        sos = signal.butter(3, 8000/(self.sr/2), 'low', output='sos')
        y = signal.sosfilt(sos, y)
        y *= np.exp(-32*t)
        
        return y * (v/127) * 0.54
    
    def hihat(self, v):
        n = int(0.1 * self.sr)
        t = np.linspace(0, 0.1, n, False)
        
        noise = np.random.randn(n)
        sos1 = signal.butter(4, 7000/(self.sr/2), 'high', output='sos')
        sos2 = signal.butter(3, 15000/(self.sr/2), 'low', output='sos')
        
        y = signal.sosfilt(sos2, signal.sosfilt(sos1, noise))
        y *= np.exp(-50*t)
        
        return y * (v/127) * 0.36

def chord_notes(root, quality):
    chords = {
        'maj': [0,4,7], 'min': [0,3,7], 'maj7': [0,4,7,11],
        'min7': [0,3,7,10], 'dom7': [0,4,7,10], 'dim': [0,3,6]
    }
    return [root + i for i in chords.get(quality, [0,4,7])]

EMOTIONS = {
    'energetic': {
        'tempo': 128, 'scale': [0,2,4,5,7,9,11], 'density': 45, 'reverb': 0.18,
        'prog': [(0,'maj'), (5,'maj'), (7,'min'), (0,'maj')],
        'inst': {
            'piano': (1.0, (85,115), [60,72], 'arps'),
            'bass': (0.9, (95,120), [36,48], 'driving'),
            'lead': (0.95, (90,118), [72,84], 'melody'),
            'arp': (0.8, (75,105), [60,72], 'fast'),
            'kick': (1.0, (110,127), None, 'four'),
            'snare': (0.95, (95,120), None, 'back'),
            'hihat': (0.85, (65,90), None, 'eighths'),
        }
    },
    'melancholic': {
        'tempo': 68, 'scale': [0,2,3,5,7,8,10], 'density': 28, 'reverb': 0.48,
        'prog': [(0,'min7'), (5,'min7'), (3,'maj7'), (7,'maj')],
        'inst': {
            'piano': (1.0, (45,75), [48,60,72], 'flowing'),
            'strings': (0.95, (50,80), [48,60], 'sustained'),
            'pad': (0.85, (40,65), [36,48,60], 'ambient'),
            'bass': (0.55, (45,70), [24,36], 'minimal'),
        }
    },
    'aggressive': {
        'tempo': 150, 'scale': [0,2,3,5,7,8,10], 'density': 48, 'reverb': 0.08,
        'prog': [(0,'min'), (3,'min'), (5,'min'), (7,'min')],
        'inst': {
            'bass': (1.0, (110,127), [24,36], 'aggressive'),
            'lead': (0.98, (105,127), [60,72], 'aggressive'),
            'kick': (1.0, (118,127), None, 'double'),
            'snare': (0.98, (110,127), None, 'aggressive'),
            'hihat': (0.9, (80,110), None, 'sixteenths'),
        }
    },
    'mysterious': {
        'tempo': 70, 'scale': [0,1,3,5,7,8,10], 'density': 26, 'reverb': 0.55,
        'prog': [(0,'min7'), (2,'dim'), (7,'min7'), (5,'maj7')],
        'inst': {
            'pad': (1.0, (35,65), [36,48,60], 'ambient'),
            'strings': (0.88, (40,70), [48,60], 'sparse'),
            'piano': (0.72, (38,68), [60,72], 'sparse'),
            'bass': (0.55, (45,70), [24,36], 'minimal'),
        }
    },
    'epic': {
        'tempo': 110, 'scale': [0,2,4,5,7,9,11], 'density': 42, 'reverb': 0.35,
        'prog': [(0,'min'), (7,'maj'), (5,'maj'), (0,'min')],
        'inst': {
            'strings': (1.0, (80,115), [48,60,72], 'expressive'),
            'piano': (0.95, (75,110), [48,60,72], 'arps'),
            'lead': (0.92, (85,118), [72,84], 'melody'),
            'bass': (0.88, (85,112), [36,48], 'driving'),
            'kick': (0.85, (90,115), None, 'four'),
            'snare': (0.75, (80,105), None, 'back'),
        }
    },
}

def compose(emotion, duration):
    cfg = EMOTIONS[emotion]
    tempo = cfg['tempo']
    beat = 60.0 / tempo
    bar = beat * 4
    
    target = max(1500, int(cfg['density'] * duration))
    
    print(f"\n{'='*70}")
    print(f"COMPOSING: {emotion.upper()}")
    print(f"{'='*70}")
    print(f"Duration: {duration:.1f}s | Tempo: {tempo} BPM | Target: {target} notes")
    
    notes = []
    prog = cfg['prog']
    scale = cfg['scale']
    total_w = sum(w for w, _, _, _ in cfg['inst'].values())
    num_bars = int(np.ceil(duration / bar))
    
    for inst, (weight, vel_range, octaves, rhythm) in cfg['inst'].items():
        inst_target = int((weight / total_w) * target)
        
        if inst in ['kick', 'snare', 'hihat']:
            # Drums
            for bar_i in range(num_bars):
                bar_t = bar_i * bar
                
                if rhythm == 'four' and inst == 'kick':
                    for b in range(4):
                        t = bar_t + b * beat
                        if t < duration:
                            notes.append((inst, 36, t, 0.1, np.random.randint(*vel_range)))
                
                elif rhythm == 'double' and inst == 'kick':
                    for b in range(4):
                        t = bar_t + b * beat
                        if t < duration:
                            notes.append((inst, 36, t, 0.1, np.random.randint(*vel_range)))
                        if b % 2 == 0:
                            t2 = t + beat/2
                            if t2 < duration:
                                notes.append((inst, 36, t2, 0.1, np.random.randint(*vel_range)))
                
                elif rhythm == 'back' and inst == 'snare':
                    for b in [1, 3]:
                        t = bar_t + b * beat
                        if t < duration:
                            notes.append((inst, 38, t, 0.1, np.random.randint(*vel_range)))
                
                elif rhythm == 'aggressive' and inst == 'snare':
                    for b in [1, 3]:
                        t = bar_t + b * beat
                        if t < duration:
                            notes.append((inst, 38, t, 0.1, np.random.randint(*vel_range)))
                        if np.random.random() < 0.35:
                            t2 = t + beat/2
                            if t2 < duration:
                                notes.append((inst, 38, t2, 0.1, np.random.randint(*vel_range)))
                
                elif rhythm == 'eighths' and inst == 'hihat':
                    for e in range(8):
                        t = bar_t + e * (beat/2)
                        if t < duration:
                            notes.append((inst, 42, t, 0.05, np.random.randint(*vel_range)))
                
                elif rhythm == 'sixteenths' and inst == 'hihat':
                    for s in range(16):
                        if s % 2 == 0 or np.random.random() < 0.65:
                            t = bar_t + s * (beat/4)
                            if t < duration:
                                notes.append((inst, 42, t, 0.05, np.random.randint(*vel_range)))
        else:
            # Melodic
            notes_per_bar = max(4, inst_target // num_bars)
            
            for bar_i in range(num_bars):
                bar_t = bar_i * bar
                chord_i = bar_i % len(prog)
                c_root, c_qual = prog[chord_i]
                c_notes = chord_notes(60 + c_root, c_qual)
                c_tones = [n % 12 for n in c_notes]
                
                if rhythm == 'sustained':
                    for cn in c_notes[:3]:
                        oct = np.random.choice(octaves)
                        pitch = np.clip(oct + (cn % 12), 24, 96)
                        notes.append((inst, pitch, bar_t, bar*0.95, np.random.randint(*vel_range)))
                
                elif rhythm == 'ambient':
                    if bar_i % 2 == 0:
                        for cn in c_notes[:3]:
                            oct = np.random.choice(octaves)
                            pitch = np.clip(oct + (cn % 12), 24, 96)
                            notes.append((inst, pitch, bar_t, bar*2, np.random.randint(*vel_range)))
                
                elif rhythm == 'arps':
                    for i in range(8):
                        t = bar_t + i * (bar/8)
                        if t >= duration:
                            break
                        cn = c_notes[i % len(c_notes)]
                        oct = np.random.choice(octaves)
                        pitch = np.clip(oct + (cn % 12), 24, 96)
                        notes.append((inst, pitch, t, bar/8*0.8, np.random.randint(*vel_range)))
                
                elif rhythm == 'fast':
                    for i in range(16):
                        t = bar_t + i * (bar/16)
                        if t >= duration:
                            break
                        cn = c_notes[i % len(c_notes)]
                        oct = np.random.choice(octaves)
                        pitch = np.clip(oct + (cn % 12), 24, 96)
                        notes.append((inst, pitch, t, bar/16*0.7, np.random.randint(*vel_range)))
                
                else:  # melody, flowing, expressive, walking, driving, aggressive, minimal, sparse
                    for i in range(notes_per_bar):
                        t = bar_t + (i/notes_per_bar) * bar
                        if t >= duration:
                            break
                        
                        if np.random.random() < 0.72:
                            pc = np.random.choice(c_tones)
                        else:
                            pc = np.random.choice(scale)
                        
                        oct = np.random.choice(octaves)
                        pitch = np.clip(oct + pc, 24, 96)
                        
                        if rhythm in ['flowing', 'expressive', 'melody']:
                            dur = (bar/notes_per_bar) * np.random.uniform(0.7, 1.8)
                        elif rhythm == 'walking':
                            dur = (bar/notes_per_bar) * np.random.uniform(0.6, 0.9)
                        else:
                            dur = (bar/notes_per_bar) * np.random.uniform(0.5, 1.2)
                        
                        notes.append((inst, pitch, t, dur, np.random.randint(*vel_range)))
    
    notes.sort(key=lambda x: x[2])
    print(f"Generated: {len(notes)} notes ({len(notes)/target*100:.1f}%)\n")
    return notes

def render(notes, duration, emotion):
    sr = 48000
    samples = int(duration * sr) + sr
    cfg = EMOTIONS[emotion]
    synth = ProSynth(sr)
    tracks = {}
    
    print("="*70)
    print("RENDERING")
    print("="*70)
    
    for inst in set(n[0] for n in notes):
        inst_notes = [n for n in notes if n[0] == inst]
        print(f"  {inst:12s}: {len(inst_notes):4d} notes")
        
        track = np.zeros(samples)
        
        for i, (_, pitch, start, dur, vel) in enumerate(inst_notes):
            if (i+1) % 100 == 0:
                print(f"    {(i+1)/len(inst_notes)*100:.0f}%", end='\r')
            
            if inst == 'kick':
                audio = synth.kick(vel)
            elif inst == 'snare':
                audio = synth.snare(vel)
            elif inst == 'hihat':
                audio = synth.hihat(vel)
            else:
                freq = 440 * (2 ** ((pitch-69)/12))
                
                if inst == 'piano':
                    audio = synth.piano(freq, dur, vel)
                elif inst == 'strings':
                    audio = synth.strings(freq, dur, vel)
                elif inst == 'bass':
                    audio = synth.bass(freq, dur, vel)
                elif inst == 'pad':
                    audio = synth.pad(freq, dur, vel)
                elif inst == 'lead':
                    audio = synth.lead(freq, dur, vel)
                elif inst == 'arp':
                    audio = synth.arp(freq, dur, vel)
                else:
                    continue
            
            idx = int(start * sr)
            end = min(idx + len(audio), len(track))
            track[idx:end] += audio[:end-idx]
        
        tracks[inst] = track
        print()
    
    print("Mixing...")
    left = np.zeros(samples)
    right = np.zeros(samples)
    
    for inst, track in tracks.items():
        if np.max(np.abs(track)) < 1e-8:
            continue
        
        peak = np.max(np.abs(track))
        if peak > 0:
            track = track / peak * 0.55
        
        pan = 0.5 + np.random.uniform(-0.18, 0.18)
        left += track * np.sqrt(1-pan)
        right += track * np.sqrt(pan)
    
    # Reverb
    rev = cfg['reverb']
    if rev > 0.05:
        delay = int(0.03 * sr)
        for i in range(4):
            d = delay * (i+1)
            g = 0.16 / (i+1)
            if d < len(left):
                left[d:] += left[:-d] * g * rev
                right[d:] += right[:-d] * g * rev
    
    # Soft clipping
    left = np.tanh(left * 1.6) * 0.72
    right = np.tanh(right * 1.6) * 0.72
    
    # Master normalize
    peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if peak > 0:
        left = left / peak * 0.88
        right = right / peak * 0.88
    
    left = left[:int(duration*sr)]
    right = right[:int(duration*sr)]
    
    print("✓ Complete\n")
    return left, right

def generate(emotion='energetic', duration=45.0, filename=None):
    if emotion not in EMOTIONS:
        emotion = 'energetic'
    if duration < 45:
        duration = 45
    
    notes = compose(emotion, duration)
    left, right = render(notes, duration, emotion)
    
    stereo = np.column_stack((left, right))
    
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"music_{emotion}_{int(duration)}s_{ts}.wav"
    
    sf.write(filename, stereo, 48000, subtype='PCM_24')
    
    size = os.path.getsize(filename) / (1024*1024)
    
    print("="*70)
    print("✅ SAVED")
    print("="*70)
    print(f"File: {filename}")
    print(f"Notes: {len(notes)}")
    print(f"Size: {size:.2f} MB")
    print("="*70 + "\n")
    
    return filename

def main():
    print("\n" + "="*70)
    print("MUSIC GENERATOR v10.0 - STUDIO QUALITY")
    print("="*70)
    print("\nEmotions:")
    for i, e in enumerate(EMOTIONS.keys(), 1):
        print(f"  {i}. {e}")
    
    while True:
        choice = input("\nSelect (1-5) or 'q': ").strip()
        
        if choice.lower() == 'q':
            print("\n✅ Goodbye!\n")
            break
        
        try:
            idx = int(choice) - 1
            emo_list = list(EMOTIONS.keys())
            
            if 0 <= idx < len(emo_list):
                emotion = emo_list[idx]
                
                dur_in = input("Duration (min 45, default 45): ").strip()
                duration = float(dur_in) if dur_in else 45.0
                
                generate(emotion, duration)
            else:
                print("❌ Invalid")
        except Exception as e:
            print(f"❌ Error: {e}")

# if __name__ == "__main__":
#     main()

# =========================
# STREAMLIT APP
# =========================
import streamlit as st
import base64

st.set_page_config(
    page_title="🎵 Melos Studio",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional light theme with animations
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #fff5f7 0%, #e8d5ff 25%, #d5e8ff 50%, #e5f3ff 75%, #fff0f5 100%);
        animation: gradientShift 15s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Headers */
    h1 {
        color: #6b46c1;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 25%, #6366f1 50%, #06b6d4 75%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 200% 200%;
        animation: gradientText 5s ease infinite;
        letter-spacing: 2px;
        text-transform: uppercase;
        filter: drop-shadow(0 0 20px rgba(139, 92, 246, 0.3));
    }
    
    @keyframes gradientText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    h2 {
        color: #7c3aed;
        font-weight: 700;
        margin-top: 2rem;
        font-size: 1.5rem;
    }
    
    /* Subtitle text */
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.3rem;
        margin-bottom: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 3px;
    }
    
    /* Radio button styling */
    .stRadio > label {
        color: #7c3aed;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .stRadio > div {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        border: 3px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 10px 30px rgba(107, 70, 193, 0.15);
    }
    
    .stRadio [role="radiogroup"] {
        gap: 1rem;
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .stRadio [role="radiogroup"] label {
        background: linear-gradient(135deg, #ffffff 0%, #faf5ff 100%);
        padding: 1.2rem 2rem;
        border-radius: 15px;
        border: 3px solid #e9d5ff;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        font-weight: 700;
        color: #6b46c1;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .stRadio [role="radiogroup"] label::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.3), transparent);
        transition: left 0.5s;
    }
    
    .stRadio [role="radiogroup"] label:hover::before {
        left: 100%;
    }
    
    .stRadio [role="radiogroup"] label:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 35px rgba(139, 92, 246, 0.4);
        border-color: #8b5cf6;
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
    }
    
    .stRadio [role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    
    /* Slider styling */
    .stSlider > label {
        color: #7c3aed;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(107, 70, 193, 0.15);
    }
    
    .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(139, 92, 246, 0.1) 50%, rgba(6, 182, 212, 0.1) 100%);
        border: 3px solid;
        border-image: linear-gradient(135deg, #ec4899, #8b5cf6, #06b6d4) 1;
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 40px rgba(139, 92, 246, 0.25);
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(139, 92, 246, 0.1), transparent);
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .info-box-text {
        color: #6b46c1;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    /* Emotion description */
    .emotion-desc {
        text-align: center;
        color: #6b7280;
        margin: 1.5rem 0;
        font-size: 1.1rem;
        font-style: italic;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        border: 2px dashed #c4b5fd;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #6366f1 100%);
        color: white;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 12px 35px rgba(124, 58, 237, 0.4);
        transition: all 0.4s ease;
        margin-top: 2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 45px rgba(124, 58, 237, 0.6);
        background: linear-gradient(135deg, #9333ea 0%, #7c3aed 50%, #6366f1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1.2rem 2rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.6);
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
        border: 3px solid #6ee7b7;
        border-radius: 15px;
        color: #059669;
        padding: 1.5rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 8px 20px rgba(16, 185, 129, 0.2);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }
    
    /* Audio visualizer container */
    .audio-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(107, 70, 193, 0.15);
        margin: 2rem 0;
    }
    
    /* Visualizer bars */
    .visualizer {
        display: flex;
        align-items: flex-end;
        justify-content: space-around;
        height: 140px;
        margin: 2rem auto;
        max-width: 700px;
        gap: 6px;
        padding: 1.5rem;
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 50%, #ede9fe 100%);
        border-radius: 20px;
        border: 3px solid;
        border-image: linear-gradient(135deg, #ec4899, #8b5cf6, #06b6d4) 1;
        box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3);
    }
    
    .bar {
        width: 100%;
        background: linear-gradient(180deg, #ec4899 0%, #8b5cf6 50%, #06b6d4 100%);
        border-radius: 6px;
        animation: pulse 0.5s ease-in-out infinite alternate;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.5);
        transition: all 0.3s ease;
    }
    
    .bar:hover {
        transform: scaleY(1.1);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
    }
    
    .bar:nth-child(1) { animation-delay: 0.0s; }
    .bar:nth-child(2) { animation-delay: 0.1s; }
    .bar:nth-child(3) { animation-delay: 0.2s; }
    .bar:nth-child(4) { animation-delay: 0.3s; }
    .bar:nth-child(5) { animation-delay: 0.4s; }
    .bar:nth-child(6) { animation-delay: 0.3s; }
    .bar:nth-child(7) { animation-delay: 0.2s; }
    .bar:nth-child(8) { animation-delay: 0.1s; }
    .bar:nth-child(9) { animation-delay: 0.0s; }
    .bar:nth-child(10) { animation-delay: 0.1s; }
    .bar:nth-child(11) { animation-delay: 0.2s; }
    .bar:nth-child(12) { animation-delay: 0.3s; }
    .bar:nth-child(13) { animation-delay: 0.4s; }
    .bar:nth-child(14) { animation-delay: 0.3s; }
    .bar:nth-child(15) { animation-delay: 0.2s; }
    .bar:nth-child(16) { animation-delay: 0.1s; }
    .bar:nth-child(17) { animation-delay: 0.0s; }
    .bar:nth-child(18) { animation-delay: 0.1s; }
    .bar:nth-child(19) { animation-delay: 0.2s; }
    .bar:nth-child(20) { animation-delay: 0.3s; }
    
    @keyframes pulse {
        0% { height: 20%; opacity: 0.7; filter: brightness(1); }
        100% { height: 90%; opacity: 1; filter: brightness(1.3); }
    }
    
    /* Audio player custom */
    audio {
        width: 100%;
        margin: 1.5rem 0;
        border-radius: 15px;
        outline: none;
    }
    
    /* Section titles */
    .section-title {
        color: #7c3aed;
        font-weight: 700;
        font-size: 1.3rem;
        margin: 2rem 0 1rem 0;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.3), transparent);
        margin: 3rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 4rem;
        padding: 2rem;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Playing indicator */
    .playing-text {
        text-align: center;
        color: #7c3aed;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 1rem 0;
        animation: fadeInOut 2s ease-in-out infinite;
    }
    
    @keyframes fadeInOut {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🎵 Melos Studio")
st.markdown('<p class="subtitle">WHERE EMOTIONS BECOME MUSIC</p>', unsafe_allow_html=True)

# Emotion Selection Section
st.markdown("## 🎭 Select Your Emotion")

emotion = st.radio(
    "",
    list(EMOTIONS.keys()),
    horizontal=True,
    label_visibility="collapsed"
)

# Emotion descriptions
emotion_descriptions = {
    'energetic': '⚡ High-tempo, driving beats with vibrant melodies',
    'melancholic': '🌧️ Slow, emotional, and deeply expressive',
    'aggressive': '🔥 Intense, powerful, and relentless energy',
    'mysterious': '🌙 Dark, ambient, and enigmatic soundscapes',
    'epic': '⚔️ Cinematic, grand, and heroic compositions'
}

st.markdown(f'<div class="emotion-desc">{emotion_descriptions[emotion]}</div>', unsafe_allow_html=True)

# Duration Selection
st.markdown("## ⏱️ Set Duration")

duration = st.slider(
    "",
    min_value=45,
    max_value=240,
    value=45,
    step=15,
    format="%d seconds",
    label_visibility="collapsed"
)

# Info Box
tempo = EMOTIONS[emotion]['tempo']
density = EMOTIONS[emotion]['density']

st.markdown(f"""
<div class="info-box">
    <p class="info-box-text">
        🎵 <strong>{emotion.upper()}</strong> | ⏱️ {duration}s | 🎼 {tempo} BPM | 📊 Density: {density}
    </p>
</div>
""", unsafe_allow_html=True)

# Generate Button
if st.button("🎧 Generate Music", use_container_width=True):
    with st.spinner("🎼 Composing → 🎹 Synthesizing → 🎚️ Mixing → 🎛️ Mastering..."):
        filename = generate(emotion, duration)
    
    st.success("✅ Music Generated Successfully!")
    
    # Audio Visualizer and Player
    st.markdown('<div class="audio-container">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🎵 Now Playing</p>', unsafe_allow_html=True)
    
    # Animated visualizer bars
    st.markdown("""
    <div class="visualizer">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
    </div>
    <p class="playing-text">🎶 Press play to enjoy your track 🎶</p>
    """, unsafe_allow_html=True)
    
    # Audio Player
    st.audio(filename)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Button
    st.markdown('<p class="section-title">📥 Download</p>', unsafe_allow_html=True)
    with open(filename, "rb") as f:
        st.download_button(
            "⬇️ Download WAV File",
            f,
            file_name=filename,
            mime="audio/wav",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown('<div class="footer">🎵 Powered by Melos Engine v10.0 | Where Emotions Become Music 🎵</div>', unsafe_allow_html=True)