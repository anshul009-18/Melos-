import streamlit as st
import subprocess
import os
import json
import time

# Page config
st.set_page_config(
    page_title="🎬 Cinematic Piano Composer",
    page_icon="🎹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Inspired by Melos Studio
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        25% { background-position: 50% 100%; }
        50% { background-position: 100% 50%; }
        75% { background-position: 50% 0%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 900px;
    }
    
    /* Main Title */
    h1 {
        color: white;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3rem !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    
    /* Section containers */
    .section-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ee7752, #e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Selectbox styling */
    .stSelectbox > label {
        background: linear-gradient(135deg, #e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        border-radius: 14px;
        border: 2px solid #23a6d5;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(35, 166, 213, 0.2);
    }
    
    .stSelectbox > div > div:hover {
        border-color: #e73c7e;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(231, 60, 126, 0.3);
    }
    
    /* Slider */
    .stSlider > label {
        background: linear-gradient(135deg, #e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #ee7752 0%, #e73c7e 50%, #23a6d5 100%);
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 600;
        color: #374151;
    }
    
    /* Text input */
    .stTextInput > label {
        color: #667eea;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input {
        background: #f9fafb;
        border-radius: 12px;
        border: 2px solid #e5e7eb;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
    }
    
    /* Generate button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #23d5ab 0%, #23a6d5 50%, #e73c7e 100%);
        color: white;
        border: none;
        border-radius: 18px;
        padding: 1.3rem 2rem;
        font-size: 1.3rem;
        font-weight: 800;
        cursor: pointer;
        transition: all 0.4s;
        box-shadow: 0 12px 40px rgba(35, 213, 171, 0.4);
        margin-top: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 18px 50px rgba(231, 60, 126, 0.6);
        background: linear-gradient(135deg, #e73c7e 0%, #23a6d5 50%, #23d5ab 100%);
    }
    
    /* Success banner */
    .success-banner {
        background: linear-gradient(135deg, #23d5ab 0%, #23a6d5 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        font-weight: 800;
        font-size: 1.2rem;
        margin: 2rem 0;
        box-shadow: 0 12px 35px rgba(35, 213, 171, 0.4);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Visualizer container */
    .visualizer-container {
        background: linear-gradient(135deg, #ee7752 0%, #e73c7e 50%, #23a6d5 100%);
        padding: 3rem;
        border-radius: 24px;
        margin: 2rem 0;
        box-shadow: 0 15px 40px rgba(238, 119, 82, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .visualizer-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .visualizer-title {
        color: white;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
    }
    
    .wave-container {
        display: flex;
        align-items: flex-end;
        justify-content: center;
        gap: 6px;
        height: 130px;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        position: relative;
        z-index: 1;
    }
    
    .wave-bar {
        width: 12px;
        background: linear-gradient(180deg, #ffd700 0%, #ffed4e 50%, #fff 100%);
        border-radius: 6px;
        animation: pulse 1.2s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { height: 30%; opacity: 0.7; transform: scaleY(1); }
        50% { height: 90%; opacity: 1; transform: scaleY(1.05); }
    }
    
    .wave-bar:nth-child(1) { animation-delay: 0s; }
    .wave-bar:nth-child(2) { animation-delay: 0.1s; }
    .wave-bar:nth-child(3) { animation-delay: 0.2s; }
    .wave-bar:nth-child(4) { animation-delay: 0.3s; }
    .wave-bar:nth-child(5) { animation-delay: 0.4s; }
    .wave-bar:nth-child(6) { animation-delay: 0.5s; }
    .wave-bar:nth-child(7) { animation-delay: 0.6s; }
    .wave-bar:nth-child(8) { animation-delay: 0.5s; }
    .wave-bar:nth-child(9) { animation-delay: 0.4s; }
    .wave-bar:nth-child(10) { animation-delay: 0.3s; }
    .wave-bar:nth-child(11) { animation-delay: 0.2s; }
    .wave-bar:nth-child(12) { animation-delay: 0.1s; }
    .wave-bar:nth-child(13) { animation-delay: 0s; }
    .wave-bar:nth-child(14) { animation-delay: 0.1s; }
    .wave-bar:nth-child(15) { animation-delay: 0.2s; }
    .wave-bar:nth-child(16) { animation-delay: 0.3s; }
    .wave-bar:nth-child(17) { animation-delay: 0.4s; }
    .wave-bar:nth-child(18) { animation-delay: 0.3s; }
    .wave-bar:nth-child(19) { animation-delay: 0.2s; }
    .wave-bar:nth-child(20) { animation-delay: 0.1s; }
    
    /* Playing text */
    .playing-text {
        text-align: center;
        background: linear-gradient(135deg, #e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 1.2rem;
        margin: 1.5rem 0;
        animation: fadeInOut 2.5s ease-in-out infinite;
        letter-spacing: 1px;
    }
    
    @keyframes fadeInOut {
        0%, 100% { opacity: 0.5; transform: scale(0.98); }
        50% { opacity: 1; transform: scale(1); }
    }
    
    /* Stats cards */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        border: 2px solid #23d5ab;
        transition: all 0.4s;
        box-shadow: 0 4px 20px rgba(35, 213, 171, 0.2);
    }
    
    .stat-card:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 15px 35px rgba(231, 60, 126, 0.4);
        border-color: #e73c7e;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ee7752, #e73c7e, #23a6d5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Download section */
    .download-title {
        text-align: center;
        background: linear-gradient(135deg, #23d5ab, #23a6d5, #e73c7e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 2rem 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    
    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
    }
    
    .info-box p {
        color: #1e40af;
        margin: 0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


def execute_notebook_papermill(
    notebook_path,
    output_path,
    emotion,
    story_arc,
    key_root,
    tempo,
    clean_mode,
    dynamic_range,
    phrase_length,
    enable_layers
):
    """Execute notebook using Papermill"""
    try:
        import papermill as pm
        
        pm.execute_notebook(
            notebook_path,
            output_path,
            parameters={
                'emotion': emotion,
                'story_arc': story_arc,
                'key_root': key_root,
                'tempo': tempo,
                'clean_mode': clean_mode,
                'dynamic_range': dynamic_range,
                'phrase_length': phrase_length,
                'enable_layers': enable_layers
            },
            kernel_name='python3'
        )
        return True, "Notebook executed successfully!", ""
    except Exception as e:
        return False, "", str(e)


def execute_notebook_nbconvert(
    notebook_path,
    emotion,
    story_arc,
    key_root,
    tempo,
    clean_mode,
    dynamic_range,
    phrase_length,
    enable_layers
):
    """Execute notebook using NBConvert"""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        last_code_cell = None
        for cell in reversed(notebook['cells']):
            if cell['cell_type'] == 'code':
                last_code_cell = cell
                break
        
        if last_code_cell:
            param_code = f"""
emotion = "{emotion}"
story_arc = "{story_arc}"
key_root = {key_root}
tempo = {tempo}
clean_mode = {clean_mode}
dynamic_range = "{dynamic_range}"
phrase_length = "{phrase_length}"
enable_layers = {enable_layers}

create_cinematic_piano(
    midi_in="final_rich_piano.mid",
    midi_out="streamlit_output.mid",
    emotion=emotion,
    story_arc=story_arc,
    key_root=key_root,
    tempo=tempo,
    clean_mode=clean_mode,
    dynamic_range=dynamic_range,
    phrase_length=phrase_length,
    enable_layers=enable_layers
)
"""
            if isinstance(last_code_cell['source'], list):
                last_code_cell['source'].append(param_code)
            else:
                last_code_cell['source'] += "\n" + param_code
        
        temp_notebook = "temp_execution.ipynb"
        with open(temp_notebook, 'w') as f:
            json.dump(notebook, f)
        
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--output', 'executed_notebook.ipynb', temp_notebook],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if os.path.exists(temp_notebook):
            os.remove(temp_notebook)
        
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


# Header
st.markdown("# 🎬 CINEMATIC PIANO COMPOSER")
st.markdown('<p class="subtitle">Transform Emotions Into Music</p>', unsafe_allow_html=True)

# Musical Settings Section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🎭 Musical Settings</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    emotion = st.selectbox(
        "🎭 Emotion",
        options=["😢 Melancholy", "🌅 Hopeful", "🥀 Bittersweet", "🏆 Triumphant", 
                 "📸 Nostalgic", "💕 Romantic", "⚔️ Epic"],
        index=5
    )
    
    key_root = st.selectbox(
        "🎼 Musical Key",
        options=["C Major", "D Major", "E Major", "F Major", "G Major", 
                 "A Major", "A Minor", "G Minor", "B♭ Minor"],
        index=1
    )

with col2:
    story_arc = st.selectbox(
        "📖 Story Arc",
        options=["⚔️ Hero's Journey", "💕 Love Story", "✨ Loss & Hope", 
                 "🔍 Mystery", "🌅 Gentle Dawn"],
        index=1
    )
    
    dynamic_range = st.selectbox(
        "📊 Dynamic Range",
        options=["Low - Gentle", "Moderate - Balanced", "High - Dramatic"],
        index=1
    )

tempo = st.slider("⏱️ Tempo (BPM)", 45, 120, 70, 1)

st.markdown("</div>", unsafe_allow_html=True)

# Advanced Settings Section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ Advanced Settings</div>', unsafe_allow_html=True)

col3, col4 = st.columns([1, 2])

with col3:
    clean_mode = st.checkbox("🧹 Clean Mode", value=True)
    phrase_length = st.selectbox(
        "🎵 Phrase Length",
        options=["Short", "Medium", "Long"],
        index=1
    )

with col4:
    st.markdown("**🎼 Instrument Layers**")
    layer_cols = st.columns(5)
    with layer_cols[0]:
        layer_melody = st.checkbox("🎹", value=True, help="Melody")
    with layer_cols[1]:
        layer_harmony = st.checkbox("🎼", value=True, help="Harmony")
    with layer_cols[2]:
        layer_bass = st.checkbox("🎸", value=True, help="Bass")
    with layer_cols[3]:
        layer_strings = st.checkbox("🎻", value=True, help="Strings")
    with layer_cols[4]:
        layer_sparkle = st.checkbox("✨", value=True, help="Sparkle")

st.markdown("</div>", unsafe_allow_html=True)

# Notebook Path Section
st.markdown('<div class="section-container">', unsafe_allow_html=True)
notebook_path = st.text_input("📓 Notebook Path", value="version 4.ipynb")
st.markdown("</div>", unsafe_allow_html=True)

# Mappings
emotion_map = {
    "😢 Melancholy": "melancholy", "🌅 Hopeful": "hopeful", "🥀 Bittersweet": "bittersweet",
    "🏆 Triumphant": "triumphant", "📸 Nostalgic": "nostalgic", "💕 Romantic": "romantic",
    "⚔️ Epic": "epic"
}

story_map = {
    "⚔️ Hero's Journey": "hero_journey", "💕 Love Story": "love_story",
    "✨ Loss & Hope": "loss_and_hope", "🔍 Mystery": "mystery",
    "🌅 Gentle Dawn": "gentle_dawn"
}

key_values = {
    "C Major": 60, "D Major": 62, "E Major": 64, "F Major": 65, "G Major": 67,
    "A Major": 69, "A Minor": 57, "G Minor": 55, "B♭ Minor": 58
}

# Generate Button
if st.button("🎵 GENERATE MUSIC"):
    
    if not os.path.exists(notebook_path):
        st.error(f"❌ Notebook not found: {notebook_path}")
        st.info("💡 Make sure 'version 4.ipynb' is in the same directory as app.py")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🎼 Initializing...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        enable_layers = {
            "melody": layer_melody, "harmony": layer_harmony, "bass": layer_bass,
            "strings": layer_strings, "sparkle": layer_sparkle
        }
        
        emotion_value = emotion_map[emotion]
        story_value = story_map[story_arc]
        key_value = key_values[key_root]
        dynamic_value = dynamic_range.split(" - ")[0].lower()
        phrase_value = phrase_length.lower()
        
        status_text.text("🎹 Executing notebook cells...")
        progress_bar.progress(40)
        
        # Execute notebook
        success, stdout, stderr = execute_notebook_papermill(
            notebook_path=notebook_path,
            output_path="executed_output.ipynb",
            emotion=emotion_value,
            story_arc=story_value,
            key_root=key_value,
            tempo=tempo,
            clean_mode=clean_mode,
            dynamic_range=dynamic_value,
            phrase_length=phrase_value,
            enable_layers=enable_layers
        )
        
        progress_bar.progress(80)
        status_text.text("🎨 Finalizing...")
        time.sleep(0.3)
        
        progress_bar.progress(100)
        time.sleep(0.2)
        status_text.empty()
        progress_bar.empty()
        
        if success:
            # Success banner
            st.markdown("""
            <div class="success-banner">
                ✅ Music Generated Successfully!
            </div>
            """, unsafe_allow_html=True)
            
            # Find output files
            output_files = []
            possible_outputs = [
                "streamlit_output.mid", "romantic_clean.mid", "02_love_theme.mid",
                "cinematic_masterpiece.mid", "final_beautified_piano.mid",
                "final_emotional_piano_v3_1.mid"
            ]
            
            for file in possible_outputs:
                if os.path.exists(file):
                    output_files.append(file)
            
            if output_files:
                # Visualizer
                st.markdown("""
                <div class="visualizer-container">
                    <div class="visualizer-title">🎵 NOW PLAYING</div>
                    <div class="wave-container">
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                        <div class="wave-bar"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Playing text
                st.markdown('<p class="playing-text">🎶 Press play to enjoy your track 🎶</p>', unsafe_allow_html=True)
                
                # Note about audio playback
                st.info("💡 **Note:** Direct MIDI playback is not available in this interface. Download the MIDI file and play it with your favorite MIDI player or DAW (FL Studio, Ableton, Logic Pro, etc.)")
                
                # Stats
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="stats-container">
                    <div class="stat-card">
                        <div class="stat-value">{len(output_files)}</div>
                        <div class="stat-label">Files</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{tempo}</div>
                        <div class="stat-label">BPM</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{sum(enable_layers.values())}</div>
                        <div class="stat-label">Layers</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Download section
                st.markdown('<div class="download-title">💾 DOWNLOAD</div>', unsafe_allow_html=True)
                
                for midi_file in output_files:
                    with open(midi_file, 'rb') as f:
                        midi_data = f.read()
                    
                    st.download_button(
                        label=f"⬇️ {midi_file}",
                        data=midi_data,
                        file_name=midi_file,
                        mime="audio/midi",
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Notebook executed but no MIDI files found.")
        else:
            st.error("❌ Execution failed!")
            with st.expander("View Error Details"):
                st.code(stderr)
            
            st.markdown("""
            <div class="info-box">
                <p><strong>🔧 Troubleshooting:</strong></p>
                <ul>
                    <li>Install: <code>pip install jupyter nbconvert papermill</code></li>
                    <li>Make sure 'version 4.ipynb' exists in the same directory</li>
                    <li>Check all dependencies are installed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="info-box">
    <p><strong>📖 How it works:</strong> This app executes your entire Jupyter notebook with your selected parameters. All cells are run sequentially to generate your cinematic piano composition.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 2rem; color: white; font-size: 0.95rem; font-weight: 600;">
    🎵 Powered by Melos Engine | Where Emotions Become Music 🎵
</div>
""", unsafe_allow_html=True)