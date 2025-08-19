import os
import sys
import asyncio
import torch
torch.classes.__path__ = []

# Environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

# Fix event loop issues
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import streamlit as st
import os
import json
import time
import tempfile
import shutil
from datetime import datetime
import pandas as pd

# Import configuration
from config import config, get_app_config, get_processing_config

# Silent error handling
try:
    from streamlit_theme import st_theme
    HAS_THEME_COMPONENT = True
except ImportError:
    HAS_THEME_COMPONENT = False

# Import existing modules
from main import VideoCaptioning

# Configuration from centralized config
app_config = get_app_config()
processing_config = get_processing_config()

APP_TITLE = app_config.title
APP_DESCRIPTION = app_config.description
MAX_FILE_SIZE_MB = app_config.max_file_size_mb
SUPPORTED_VIDEO_FORMATS = app_config.supported_formats
DEMO_VIDEOS_PATH = app_config.demo_videos_path

# Ensure directories exist using config paths
os.makedirs(app_config.videos_dir, exist_ok=True)
os.makedirs(app_config.results_dir, exist_ok=True)
os.makedirs(app_config.temp_dir, exist_ok=True)
os.makedirs(DEMO_VIDEOS_PATH, exist_ok=True)

def get_demo_videos():
    """Get list of demo videos from the demo folder"""
    demo_videos = []
    if os.path.exists(DEMO_VIDEOS_PATH):
        for file in os.listdir(DEMO_VIDEOS_PATH):
            if file.lower().endswith(tuple(SUPPORTED_VIDEO_FORMATS)):
                demo_videos.append(file)
    return sorted(demo_videos)

class DemoVideoFile:
    """Mock uploaded file object for demo videos"""
    def __init__(self, file_path):
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        self.size = os.path.getsize(file_path)
        self.type = f"video/{os.path.splitext(self.name)[1][1:]}"
    
    def getbuffer(self):
        """Read file content as buffer"""
        with open(self.file_path, 'rb') as f:
            return f.read()

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=app_config.title.split()[0] + " " + app_config.title.split()[1],  # "AI Video"
        page_icon=app_config.page_icon,
        layout=app_config.layout,
        initial_sidebar_state=app_config.sidebar_state
    )

def add_custom_css():
    """Add custom CSS styling with st-theme component support and professional sidebar"""
    # Detect theme for text color
    is_dark = False
    if HAS_THEME_COMPONENT:
        try:
            current_theme = st_theme()
            if current_theme and isinstance(current_theme, dict):
                is_dark = current_theme.get('base') == 'dark'
        except Exception:
            is_dark = False

    if is_dark:
        text_color = "#ffffff"
        accent_color = "#a084e8"
        purple_color = "#7e3ff2"
    else:
        text_color = "#2c3e50"
        accent_color = "#667eea"
        purple_color = "#7e3ff2"

    st.markdown(f"""
    <style>
    /* Professional sidebar with statistics-matching background */
    section[data-testid="stSidebar"] > div:first-child {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 60%, rgba(102, 126, 234, 0.12) 100%) !important;
        color: {text_color} !important;
        min-height: 100vh;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: saturate(180%) blur(20px);
    }}
    
    /* Expander header styling with matching colors */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {{
        color: {accent_color} !important;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.8rem 1rem;
        background: rgba(102, 126, 234, 0.08);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }}
    
    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {{
        background: rgba(102, 126, 234, 0.12);
        border-color: #7e3ff2;
        box-shadow: 0 4px 12px rgba(126, 63, 242, 0.2);
        transform: translateY(-2px);
    }}
    
    /* Expander content with matching background */
    section[data-testid="stSidebar"] .streamlit-expanderContent {{
        background: rgba(102, 126, 234, 0.05);
        color: {text_color};
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        margin-bottom: 0.8rem;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
    }}
    
    /* Sidebar title with subtle styling */
    section[data-testid="stSidebar"] h1 {{
        color: {accent_color} !important;
        text-align: center;
        margin-bottom: 1.5rem;
        font-size: 1.6rem;
    }}
    /* Main container width limitation */
    .main .block-container {{
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }}
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .success-message {{
        color: #28a745;
        padding: 0.5rem 0;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: 500;
        font-size: 1rem;
        background: transparent;
    }}
    .processing-status {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }}
    .navigation-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
        padding: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
        width: 100%;
    }}
    .video-counter {{
        font-size: 1rem;
        font-weight: 600;
        color: #7e3ff2;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        background: transparent;
        border: 2px solid #7e3ff2;
        border-radius: 8px;
        padding: 0.5rem 2.5rem;
        margin: 0 0.5rem;
        height: 40px;
        min-width: 180px;
        flex: 0 0 auto;
        box-sizing: border-box;
    }}
    body.dark-theme .video-counter, .stApp[data-theme="dark"] .video-counter {{
        color: #ffffff !important;
    }}
    .video-preview {{
        max-width: 800px;
        margin: 1 auto;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }}
    /* Demo video selection styling */
    .demo-video-item {{
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        cursor: pointer;
    }}
    .demo-video-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }}
    /* Gradient underline for section-header and upload-header */
    .section-header, .upload-header {{
        font-size: 1.8rem;
        font-weight: 600;
        color: {text_color} !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.3rem;
        text-align: left;
        position: relative;
    }}
    .section-header::after, .upload-header::after {{
        content: "";
        display: block;
        position: absolute;
        left: 0;
        bottom: 0;
        width: 100%;
        height: 3px;
        border-radius: 2px;
        background: linear-gradient(90deg, #7e3ff2 0%, #764ba2 60%, #667eea 100%);
    }}
    .summary-container {{
        background: transparent !important;
        border: none !important;
        padding: 0;
        margin: 1rem 0 2rem 0;
        box-shadow: none !important;
    }}
    .summary-text {{
        font-size: 1.2rem;
        line-height: 1.8;
        color: {text_color} !important;
        text-align: justify;
        font-weight: 400;
        margin: 0 0 2rem 0;
        font-family: 'Georgia', serif;
    }}
    .stats-container {{
        background: transparent !important;
        border: none !important;
        padding: 0;
        margin: 1rem 0 2rem 0;
        box-shadow: none !important;
    }}
    .stat-item {{
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin: 0.3rem 0;
    }}
    .stat-item:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }}
    .stat-value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        display: block;
        margin-bottom: 0.3rem;
    }}
    .stat-label {{
        font-size: 0.85rem;
        color: {text_color} !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .scenes-container {{
        background: transparent !important;
        border: none !important;
        padding: 0;
        margin: 1rem 0 2rem 0;
        box-shadow: none !important;
    }}
    .scene-item {{
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0;
        margin: 0.3rem 0;
    }}
    .scene-header {{
        font-size: 1rem;
        font-weight: 600;
        color: {purple_color} !important;
        margin-bottom: 0.3rem;
    }}
    .scene-caption {{
        font-size: 1rem;
        line-height: 1.6;
        color: {text_color} !important;
        text-align: justify;
        font-style: italic;
        margin-left: 1rem;
    }}
    .downloads-container {{
        border: none !important;
        background: transparent !important;
        padding: 0;
        margin: 1rem 0 2rem 0;
    }}
    /* Compact metrics styling */
    .metric-compact {{
        text-align: center;
        padding: 0.3rem;
        margin: 0.1rem 0;
    }}
    .metric-compact .metric-label {{
        font-size: 0.8rem;
        color: {accent_color} !important;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }}
    .metric-compact .metric-value {{
        font-size: 0.9rem;
        color: {text_color} !important;
        font-weight: 500;
        margin: 0;
        padding: 0;
    }}
    /* Βελτιωμένο styling για τα κουμπιά επεξεργασίας βίντεο */
    .primary-action-button button[kind="primary"] {{
        background: linear-gradient(135deg, #5a4dbf 0%, #8a7fd1 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        padding: 1.25rem 1.75rem !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 18px rgba(90, 77, 191, 0.7) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }}
    .primary-action-button button[kind="primary"]:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(90, 77, 191, 0.9) !important;
    }}
    /* Target all primary buttons to ensure proper styling for demo button */
    div[data-testid="stButton"] > button[kind="primary"],
    button[kind="primary"][data-testid="baseButton-primary"],
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid transparent !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 1.5rem 1.5rem !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
        min-height: 1.5rem !important;
        height: auto !important;
        transition: all 0.3s ease !important;
        outline: none !important;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[kind="primary"][data-testid="baseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {{
        background: linear-gradient(135deg, #7e3ff2 0%, #a084e8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(126, 63, 242, 0.6) !important;
        border: 2px solid rgba(126, 63, 242, 0.3) !important;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:focus,
    button[kind="primary"][data-testid="baseButton-primary"]:focus,
    .stButton > button[kind="primary"]:focus {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        outline: none !important;
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4) !important;
    }}
    /* Hide default streamlit containers */
    .stContainer > div {{
        background: transparent !important;
    }}
    .element-container {{
        background: transparent !important;
    }}
    .stMarkdown {{
        background: transparent !important;
    }}
    /* Console warning suppression */
    .stApp {{
        /* Suppress some console warnings */
    }}
    </style>
    
    <script>
    // Suppress console warnings from st-theme component
    (function() {{
        const originalConsoleWarn = console.warn;
        const originalConsoleError = console.error;
        
        console.warn = function(...args) {{
            const message = args.join(' ');
            if (message.includes('Unrecognized feature') || 
                message.includes('iframe') ||
                message.includes('ComponentInstance')) {{
                return; // Suppress these warnings
            }}
            originalConsoleWarn.apply(console, args);
        }};
        
        console.error = function(...args) {{
            const message = args.join(' ');
            if (message.includes('ComponentInstance') ||
                message.includes('iframe')) {{
                return; // Suppress these errors
            }}
            originalConsoleError.apply(console, args);
        }};
    }})();
    </script>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'uploaded_videos' not in st.session_state:
        st.session_state.uploaded_videos = []
    if 'processed_videos' not in st.session_state:
        st.session_state.processed_videos = {}
    if 'current_video_index' not in st.session_state:
        st.session_state.current_video_index = 0
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False
    if 'batch_processing' not in st.session_state:
        st.session_state.batch_processing = False

def format_timestamp(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_srt_timestamp(seconds):
    """Format timestamp for SRT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def generate_srt_content(scene_captions):
    """Generate SRT subtitle content"""
    srt_content = ""
    for i, scene in enumerate(scene_captions):
        start_time = format_srt_timestamp(scene['start_time'])
        end_time = format_srt_timestamp(scene['end_time'])
        
        srt_content += f"{i+1}\n"
        srt_content += f"{start_time} --> {end_time}\n"
        srt_content += f"{scene['caption']}\n\n"
    
    return srt_content

def create_download_buttons(results, video_name, subtitled_video_path=None):
    """Create download buttons for different output formats"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 JSON Data",
            data=json_str,
            file_name=f"{video_name}_captions.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Subtitled video download
        if subtitled_video_path and os.path.exists(subtitled_video_path):
            with open(subtitled_video_path, "rb") as file:
                st.download_button(
                    label="🎥 Subtitled Video",
                    data=file.read(),
                    file_name=f"{video_name}_subtitled.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        else:
            st.button(
                label="🎥 No Subtitled Video",
                disabled=True,
                use_container_width=True
            )
    
    with col3:
        srt_content = generate_srt_content(results['scene_captions'])
        st.download_button(
            label="🎬 SRT Subtitles",
            data=srt_content,
            file_name=f"{video_name}_subtitles.srt",
            mime="text/plain",
            use_container_width=True
        )

def setup_sidebar():
    """Setup professional, clean, expandable sidebar with accurate model information"""
    import streamlit as st

    is_dark = False
    if HAS_THEME_COMPONENT:
        try:
            current_theme = st_theme()
            if current_theme and isinstance(current_theme, dict):
                is_dark = current_theme.get('base') == 'dark'
        except Exception:
            is_dark = False

    # Sidebar styling
    st.sidebar.markdown(
        """
        <style>
        /* Clean sidebar content styling */
        .sidebar-section-clean {
            margin-bottom: 1rem;
            padding: 0;
            color: inherit;
            font-size: 0.95rem;
        }
        .sidebar-section-clean h4 {
            color: #667eea;
            font-size: 1.1rem;
            margin: 0 0 0.8rem 0;
            font-weight: 600;
        }
        .sidebar-section-clean ul {
            margin: 0;
            padding-left: 1rem;
        }
        .sidebar-section-clean li {
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }
        .sidebar-section-clean p {
            margin: 0 0 0.5rem 0;
            line-height: 1.4;
        }
        .sidebar-section-clean strong {
            color: #667eea;
            font-weight: 600;
        }
        .sidebar-section-clean a {
            color: #7e3ff2;
            text-decoration: none;
        }
        .sidebar-section-clean a:hover {
            color: #667eea;
        }
        </style>
        """, unsafe_allow_html=True
    )

    with st.sidebar.expander("🚀 Quick Start", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>1. Upload</strong> video files</p>
            <p><strong>2. Start</strong> captioning</p>
            <p><strong>3. View</strong> results</p>
            <p><strong>4. Download</strong> outputs</p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("🤖 AI Architecture", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>BLIP Base (Salesforce)</strong><br>
            Employed for generating accurate, context-aware captions for individual video frames.</p>
            <p><strong>CLIP ViT-B/32 (OpenAI)</strong><br>
            Robust video frame embeddings and semantic alignment between visual content and captions.</p>
            <p><strong>FLAN-T5 Base (Google)</strong><br>
            Produces coherent, natural-language video summaries from scene captions.</p>
            <p><strong>SentenceTransformers</strong><br>
            Enables semantic similarity analysis for caption fusion and outlier filtering.</p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("🔄 Processing Pipeline", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>Frame Extraction</strong><br>
            Fixed-interval sampling at 1fps using OpenCV for consistent input processing</p>
            <p><strong>Scene Detection</strong><br>
            Hybrid perceptual hashing and optical flow analysis enabling accurate detection of scene transitions</p>
            <p><strong>Multi-Frame Caption Generation</strong><br>
            BLIP generates context-rich descriptions for start, middle, and end frames per scene</p>
            <p><strong>Motion-Aware Prioritization</strong><br>
            Dynamic content detection with keyword-based filtering for enhanced caption quality</p>
            <p><strong>Semantic Caption Selection</strong><br>
            SentenceTransformer embeddings fuse captions via cosine similarity and keyword uniqueness</p>
            <p><strong>CLIP-Guided Coherence</strong><br>
            Global video embedding validation ensures visual-semantic alignment consistency</p>
            <p><strong>Scene Merging</strong><br>
            Adjacent scene consolidation using cosine similarity with frequency-based caption selection and motion keyword prioritization</p>
            <p><strong>Caption Quality Control</strong><br>
            LOF and DBSCAN algorithms remove semantic outliers using cosine distance metrics and clustering techniques</p>
            <p><strong>Narrative Summarization</strong><br>
            FLAN-T5 generates coherent narratives with composite scoring (semantic centrality + CLIP alignment + duration weighting)</p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("📋 Technical Specifications", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>Supported Formats</strong><br>
            MP4, AVI, MOV, MKV, WebM</p>
            <p><strong>File Size Limit</strong><br>
            Up to 500MB per video</p>
            <p><strong>Export Options</strong><br>
            Multi-format output (JSON, SRT, subtitled video)</p>
            <p><strong>Processing Architecture</strong><br>
            GPU-accelerated with intelligent embedding caching</p>
            <p><strong>Caching System</strong><br>
            Semantic and CLIP embedding cache for computational efficiency</p>
            <p><strong>System Compatibility</strong><br>
            Adaptive CPU/GPU processing with automatic device detection</p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("🎓 Research Foundation", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>Training Dataset</strong><br>
            MSR-VTT Video-to-Text Corpus for video captioning benchmarking</p>
            <p><strong>Evaluation Metrics</strong><br>
            BLEU, METEOR, ROUGE-L for caption quality assessment</p>
            <p><strong>Model Architecture</strong><br>
            Multi-modal transformer ensemble with vision-language alignment</p>
            <p><strong>Research Context</strong><br>
            Deep Learning for Video Understanding and Natural Language Generation</p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("👨‍💻 Developer", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>Created By</strong><br>
            Antonis Tsiakiris</p>
            <p><strong>Research Focus</strong><br>
            Computer Vision, Natural Language Processing & Deep Learning</p>
            <p><strong>GitHub Repository</strong><br>
            <a href="https://github.com/AntonisXT/AI-Video-Captioning" target="_blank">github.com/AntonisXT//AI-Video-Captioning</a></p>
            <p><strong>Professional Network</strong><br>
            <a href="https://www.linkedin.com/in/antonis-tsiakiris-880114359" target="_blank">LinkedIn Profile</a></p>
            <p><strong>Contact</strong><br>
            <a href="mailto:tsiakiris.dev@gmail.com">tsiakiris.dev@gmail.com</a></p>
            </div>
            """, unsafe_allow_html=True
        )

    with st.sidebar.expander("📦 Version", expanded=False):
        st.markdown(
            """
            <div class="sidebar-section-clean">
            <p><strong>Current Version</strong><br>
            v1.0.0 - Thesis Implementation</p>
            <p><strong>Release Status</strong><br>
            Academic Research Project</p>
            <p><strong>Last Updated</strong><br>
            May 2025</p>
            <p><strong>License</strong><br>
            MIT License</p>
            </div>
            """, unsafe_allow_html=True
        )

    # Return config-based settings
    return {
        'generate_subtitles': True,
        'max_frames': processing_config.max_frames,
        'similarity_threshold': processing_config.similarity_threshold,
        'min_scene_duration': processing_config.min_scene_duration
    }

def save_temp_file(uploaded_file, temp_dir=None):
    """Save uploaded file to temporary directory"""
    temp_dir = temp_dir or app_config.temp_dir
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def process_single_video(video_file, settings=None):
    """Process a single video and store results in session state"""
    
    # Use config if settings not provided
    if settings is None:
        settings = {
            'generate_subtitles': True,
            'max_frames': processing_config.max_frames,
            'similarity_threshold': processing_config.similarity_threshold,
            'min_scene_duration': processing_config.min_scene_duration
        }
    
    progress_container = st.container()
    
    with progress_container:
        st.markdown(f'<div class="processing-status">🔄 Processing {video_file.name}... Please wait.</div>', unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("💾 Saving uploaded file...")
        progress_bar.progress(10)
        
        temp_path = save_temp_file(video_file, app_config.temp_dir)
        
        try:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_folder = f"session_{session_id}"
            
            status_text.text("🤖 Initializing AI models...")
            progress_bar.progress(20)
            
            video_captioner = VideoCaptioning(
                folder_name=session_folder,
                generate_subtitles=settings['generate_subtitles']
            )
            
            status_text.text("🎬 Analyzing video structure...")
            progress_bar.progress(30)
            
            video_name = video_file.name
            video_name_no_ext = os.path.splitext(video_name)[0]
            
            session_dir = os.path.join(app_config.videos_dir, session_folder)
            os.makedirs(session_dir, exist_ok=True)
            
            final_video_path = os.path.join(session_dir, video_name)
            shutil.move(temp_path, final_video_path)
            
            status_text.text("📝 Generating captions...")
            progress_bar.progress(60)
            
            video_captioner.process_video(final_video_path)
            
            if settings['generate_subtitles']:
                status_text.text("🎬 Creating subtitled video...")
                progress_bar.progress(85)
            
            status_text.text("📊 Creating summary...")
            progress_bar.progress(95)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            results_path = os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "captions", f"{video_name_no_ext}_captions.json")
            
            subtitled_video_path = None
            if settings['generate_subtitles']:
                possible_paths = [
                    os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", f"{video_name_no_ext}_subtitled.mp4"),
                    os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", f"{video_name_no_ext}.mp4"),
                    os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", video_name),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        subtitled_video_path = path
                        break
            
            if os.path.exists(results_path):
                with open(results_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                video_to_display = subtitled_video_path if subtitled_video_path and os.path.exists(subtitled_video_path) else final_video_path
                
                st.session_state.processed_videos[video_file.name] = {
                    'name': video_name,
                    'results': results,
                    'video_path': video_to_display,
                    'subtitled_video_path': subtitled_video_path,
                    'has_subtitles': settings['generate_subtitles'] and subtitled_video_path is not None
                }
                
                st.session_state.processing_status[video_file.name] = 'completed'
                
                time.sleep(1)
                progress_container.empty()
                
                return True
            else:
                st.error("❌ Results file not found.")
                st.session_state.processing_status[video_file.name] = 'error' 
                return False
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.session_state.processing_status[video_file.name] = 'error' 
            return False
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def process_all_videos(settings):
    """Process all uploaded or demo videos sequentially"""
    st.session_state.batch_processing = True
    total_videos = len(st.session_state.uploaded_videos)
    
    main_progress_container = st.container()
    
    with main_progress_container:
        st.markdown('<div class="processing-status">🚀 Processing All Videos... Please wait.</div>', unsafe_allow_html=True)
        
        overall_progress = st.progress(0)
        overall_status = st.empty()
        
        video_progress_container = st.container()
        
        for idx, video_file in enumerate(st.session_state.uploaded_videos):
            
            if st.session_state.processing_status.get(video_file.name) == 'completed':
                overall_progress.progress((idx + 1) / total_videos)
                overall_status.text(f"✅ Video {idx + 1}/{total_videos}: {video_file.name} (Already processed)")
                continue
            
            overall_status.text(f"🔄 Processing Video {idx + 1}/{total_videos}: {video_file.name}")
            st.session_state.processing_status[video_file.name] = 'processing'
            
            with video_progress_container:
                success = process_single_video_batch(video_file, idx, settings)
                if not success:
                    st.error(f"❌ Failed to process video: {video_file.name}")
                    st.session_state.processing_status[video_file.name] = 'error'
                    break
                else:
                    st.session_state.processing_status[video_file.name] = 'completed'
            
            overall_progress.progress((idx + 1) / total_videos)
        
        overall_status.text(f"✅ Batch processing completed! Processed {total_videos} videos.")
        time.sleep(2)
        main_progress_container.empty()
    
    st.session_state.batch_processing = False
    st.session_state.current_video_index = 0

def process_single_video_batch(video_file, video_index, settings):
    """Process a single video in batch mode (simplified progress display)"""
    try:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{video_index}"
        session_folder = f"session_{session_id}"
        
        video_captioner = VideoCaptioning(
            folder_name=session_folder,
            generate_subtitles=settings['generate_subtitles']
        )
        
        video_name = video_file.name
        video_name_no_ext = os.path.splitext(video_name)[0]
        
        session_dir = os.path.join(app_config.videos_dir, session_folder)
        os.makedirs(session_dir, exist_ok=True)
        
        temp_path = save_temp_file(video_file, app_config.temp_dir)
        final_video_path = os.path.join(session_dir, video_name)
        shutil.move(temp_path, final_video_path)
        
        video_captioner.process_video(final_video_path)
        
        results_path = os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "captions", f"{video_name_no_ext}_captions.json")
        
        subtitled_video_path = None
        if settings['generate_subtitles']:
            possible_paths = [
                os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", f"{video_name_no_ext}_subtitled.mp4"),
                os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", f"{video_name_no_ext}.mp4"),
                os.path.join(app_config.results_dir, session_folder, video_name_no_ext, "subtitled", video_name),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    subtitled_video_path = path
                    break
        
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            video_to_display = subtitled_video_path if subtitled_video_path and os.path.exists(subtitled_video_path) else final_video_path
            
            st.session_state.processed_videos[video_file.name] = {
                'name': video_name,
                'results': results,
                'video_path': video_to_display,
                'subtitled_video_path': subtitled_video_path,
                'has_subtitles': settings['generate_subtitles'] and subtitled_video_path is not None
            }
            
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing {video_file.name}: {str(e)}")
        return False
def load_all_demo_videos():
    """Load all demo videos into session state"""
    demo_videos = get_demo_videos()
    if demo_videos:
        demo_video_objects = []
        for video_name in demo_videos:
            video_path = os.path.join(DEMO_VIDEOS_PATH, video_name)
            demo_file = DemoVideoFile(video_path)
            demo_video_objects.append(demo_file)
        
        st.session_state.uploaded_videos = demo_video_objects
        st.session_state.demo_mode = True
        st.session_state.current_video_index = 0
        # Reset processing status for new videos
        st.session_state.processed_videos = {}
        st.session_state.processing_status = {}

def display_demo_videos():
    """Display demo videos selection"""
    demo_videos = get_demo_videos()
    
    if not demo_videos:
        st.warning("📁 No demo videos found in data/videos/app_demo/")
        st.info("💡 Add some video files to the demo folder to see them here!")
        return
    
    st.markdown('<div class="section-header">🎬 Try Demo Videos</div>', unsafe_allow_html=True)
    
    # Single button with full width to match drag and drop box
    if st.button(
        f"🎥 Caption Demo Videos ({len(demo_videos)} available)",
        key="load_all_demos",
        type="primary",
        use_container_width=True
    ):
        load_all_demo_videos()
        st.rerun()

def display_video_navigation():
    """Display enhanced video navigation with centered counter"""
    if not st.session_state.uploaded_videos:
        return
    
    total_videos = len(st.session_state.uploaded_videos)
    current_index = st.session_state.current_video_index
    
    st.markdown('<div class="navigation-container">', unsafe_allow_html=True)
    
    # Layout with buttons on sides and centered counter
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
    
    with nav_col1:
        if st.button("⏮️ First", disabled=(current_index == 0), key="first_btn", use_container_width=True):
            st.session_state.current_video_index = 0
            st.rerun()
    
    with nav_col2:
        if st.button("⬅️ Previous", disabled=(current_index == 0), key="prev_btn", use_container_width=True):
            st.session_state.current_video_index = max(0, current_index - 1)
            st.rerun()
    
    with nav_col3:
        # Centered video counter
        demo_indicator = " (Demo)" if st.session_state.demo_mode else ""
        st.markdown(f'''
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <div class="video-counter">Video {current_index + 1} of {total_videos}{demo_indicator}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with nav_col4:
        if st.button("➡️ Next", disabled=(current_index == total_videos - 1), key="next_btn", use_container_width=True):
            st.session_state.current_video_index = min(total_videos - 1, current_index + 1)
            st.rerun()
    
    with nav_col5:
        if st.button("⏭️ Last", disabled=(current_index == total_videos - 1), key="last_btn", use_container_width=True):
            st.session_state.current_video_index = total_videos - 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_current_video():
    """Display the current video and its information"""
    if not st.session_state.uploaded_videos:
        return
    
    current_index = st.session_state.current_video_index
    current_video = st.session_state.uploaded_videos[current_index]
    
    file_size_mb = current_video.size / (1024 * 1024)
    
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns([2, 1, 1, 1, 2])
    
    with metric_col2:
        st.markdown(f'''<div class="metric-compact"><div class="metric-label">📁 File Name</div><div class="metric-value">{current_video.name}</div></div>''', unsafe_allow_html=True)
    with metric_col3:
        st.markdown(f'''<div class="metric-compact"><div class="metric-label">📏 File Size</div><div class="metric-value">{file_size_mb:.2f} MB</div></div>''', unsafe_allow_html=True)
    with metric_col4:
        st.markdown(f'''<div class="metric-compact"><div class="metric-label">🎬 File Type</div><div class="metric-value">{current_video.type}</div></div>''', unsafe_allow_html=True)
    
    if current_video.name in st.session_state.processed_videos:
        processed_data = st.session_state.processed_videos[current_video.name]
        display_processed_video(processed_data)
    else:
        st.markdown('<div style="margin-top: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
        
        total_videos = len(st.session_state.uploaded_videos)
        # Recalculate unprocessed count based on new state structure
        unprocessed_count = sum(1 for v in st.session_state.uploaded_videos if st.session_state.processing_status.get(v.name) != 'completed')
        
        if total_videos > 1 and unprocessed_count > 1:
            button_col1, gap_col, button_col2 = st.columns([2, 0.3, 2])
            
            with button_col1:
                st.markdown('<div class="primary-action-button">', unsafe_allow_html=True)
                if st.button("🚀 Process Current Video", key=f"process_{current_index}", type="primary", use_container_width=True):
                    st.session_state.processing_status[current_video.name] = 'processing' 
                    settings = {'generate_subtitles': True, 'max_frames': processing_config.max_frames, 'similarity_threshold': processing_config.similarity_threshold, 'min_scene_duration': processing_config.min_scene_duration}
                    if process_single_video(current_video, settings): 
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with button_col2:
                st.markdown('<div class="primary-action-button">', unsafe_allow_html=True)
                if st.button(f"🎬 Process All Videos ({unprocessed_count} remaining)", key="process_all", type="primary", use_container_width=True):
                    settings = {'generate_subtitles': True, 'max_frames': processing_config.max_frames, 'similarity_threshold': processing_config.similarity_threshold, 'min_scene_duration': processing_config.min_scene_duration}
                    process_all_videos(settings)
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            button_col1, button_col2, button_col3 = st.columns([1.5, 3, 1.5])
            with button_col2:
                st.markdown('<div class="primary-action-button">', unsafe_allow_html=True)
                if st.button("🚀 Process Current Video", key=f"process_{current_index}", type="primary", use_container_width=True):
                    st.session_state.processing_status[current_video.name] = 'processing' 
                    settings = {'generate_subtitles': True, 'max_frames': processing_config.max_frames, 'similarity_threshold': processing_config.similarity_threshold, 'min_scene_duration': processing_config.min_scene_duration}
                    if process_single_video(current_video, settings): 
                        st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if current_video.name in st.session_state.processing_status:
            status = st.session_state.processing_status[current_video.name]
            if status == 'processing':
                st.warning("🔄 This video is currently being processed...")
            elif status == 'error':
                st.error("❌ Processing failed for this video.")
        
        if st.session_state.batch_processing:
            st.info("🎬 Batch processing in progress... Please wait for all videos to complete.")
        
        video_col1, video_col2, video_col3 = st.columns([0.2, 3.6, 0.2])
        with video_col2:
            st.markdown('<div style="text-align: center;"><h3>📺 Video Preview</h3></div>', unsafe_allow_html=True)
            st.markdown('<div class="video-preview">', unsafe_allow_html=True)
            if st.session_state.demo_mode:
                st.video(current_video.file_path)
            else:
                st.video(current_video)
            st.markdown('</div>', unsafe_allow_html=True)

def display_processed_video(processed_data):
    """Display processed video with results"""
    st.markdown('<div class="success-message">🎉 Video captioning completed successfully!</div>', unsafe_allow_html=True)
    
    # Video display
    video_col1, video_col2, video_col3 = st.columns([0.2, 3.6, 0.2])
    with video_col2:
        if processed_data['has_subtitles']:
            st.markdown('<div style="text-align: center;"><h3>📺 Subtitled Video</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center;"><h3>📺 Processed Video</h3></div>', unsafe_allow_html=True)
        
        if processed_data['video_path'] and os.path.exists(processed_data['video_path']):
            st.markdown('<div class="video-preview">', unsafe_allow_html=True)
            st.video(processed_data['video_path'])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Video file not found")
    
    results = processed_data['results']
    
    # Video Summary
    st.markdown('<div class="section-header">📋 Video Summary</div>', unsafe_allow_html=True)
    st.markdown(f'''
    <div class="summary-container">
        <div class="summary-text">{results["summary"]}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Video Statistics
    st.markdown('<div class="section-header">📊 Video Statistics</div>', unsafe_allow_html=True)
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{len(results["scene_captions"])}</span>
            <div class="stat-label">🎬 Total Scenes</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        total_duration = max([scene['end_time'] for scene in results['scene_captions']])
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{total_duration:.1f}s</span>
            <div class="stat-label">⏱️ Duration</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        avg_scene_length = sum([scene['end_time'] - scene['start_time'] for scene in results['scene_captions']]) / len(results['scene_captions'])
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{avg_scene_length:.1f}s</span>
            <div class="stat-label">📊 Avg Scene</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        total_words = sum([len(scene['caption'].split()) for scene in results['scene_captions']])
        st.markdown(f'''
        <div class="stat-item">
            <span class="stat-value">{total_words}</span>
            <div class="stat-label">📝 Total Words</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Scene Descriptions
    st.markdown('<div class="section-header">🎬 Scene Descriptions</div>', unsafe_allow_html=True)
    st.markdown('<div class="scenes-container">', unsafe_allow_html=True)
    
    for i, scene in enumerate(results['scene_captions']):
        clean_caption = scene['caption'].strip('"\'')
        
        st.markdown(f'''
        <div class="scene-item">
            <div class="scene-header">
                🎬 Scene {i+1}: {format_timestamp(scene['start_time'])} - {format_timestamp(scene['end_time'])} ({scene['end_time'] - scene['start_time']:.1f}s)
            </div>
            <div class="scene-caption">{clean_caption}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download Results
    st.markdown('<div class="section-header">📥 Download Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="downloads-container">', unsafe_allow_html=True)
    
    create_download_buttons(results, os.path.splitext(processed_data['name'])[0], processed_data['subtitled_video_path'])
    
    st.markdown('</div>', unsafe_allow_html=True)

def main_content_area():
    """Main content area"""
    is_processing = any(status == 'processing' for status in st.session_state.processing_status.values())

    if not is_processing and not st.session_state.batch_processing:
        settings = setup_sidebar()
    else:
        # Use config defaults when processing
        settings = {
            'generate_subtitles': True,
            'max_frames': processing_config.max_frames,
            'similarity_threshold': processing_config.similarity_threshold,
            'min_scene_duration': processing_config.min_scene_duration
        }
    
    st.markdown('<div class="upload-header">📁 Upload Videos</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose one or more video files",
        type=SUPPORTED_VIDEO_FORMATS,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}. Max size: {MAX_FILE_SIZE_MB}MB per file",
        key="video_uploader"  
    )

    new_file_names = {f.name for f in uploaded_files}
    old_file_names = {f.name for f in st.session_state.get('uploaded_videos', []) if not st.session_state.get('demo_mode', False)}

    if new_file_names != old_file_names:
        
        valid_files = [f for f in uploaded_files if f.size <= MAX_FILE_SIZE_MB * 1024 * 1024] 
        if len(valid_files) != len(uploaded_files):
            st.error(f"❌ Some files exceed {MAX_FILE_SIZE_MB}MB limit and were excluded.") 

        valid_file_names = {f.name for f in valid_files}

        old_processed = st.session_state.get('processed_videos', {})
        new_processed = {name: data for name, data in old_processed.items() if name in valid_file_names}
        
        old_status = st.session_state.get('processing_status', {})
        new_status = {name: status for name, status in old_status.items() if name in valid_file_names}
        
        st.session_state.uploaded_videos = valid_files 
        st.session_state.demo_mode = False
        st.session_state.processed_videos = new_processed
        st.session_state.processing_status = new_status
        
        st.session_state.current_video_index = 0
        
        st.rerun()

    if st.session_state.get('uploaded_videos'):
        display_video_navigation() 
        display_current_video() 
    else:
        display_demo_videos()
        
        st.info("👆 Upload your own videos or try the demo videos above to begin processing")
        
        st.subheader("🌟 Features")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **AI-Powered Analysis:**
            - 🎯 Automatic scene detection
            - 🖼️ Frame-by-frame visual analysis  
            - 🧠 Deep learning semantic processing
            - 📝 Natural language caption generation
            """) 
        
        with col2:
            st.markdown("""
            **Output Formats:**
            - 📄 JSON with scene captions & summary
            - 📊 Video statistics & analytics
            - 🎬 SRT subtitle files
            - 🎥 Subtitled video files
            """)

def main():
    """Main application function"""
    setup_page_config()
    add_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown(f'<h1 class="main-header">🎬 {APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #666;'>{APP_DESCRIPTION}</p>", unsafe_allow_html=True)
    
    # Main content
    main_content_area()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit, PyTorch, and Transformers</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()