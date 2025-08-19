import streamlit as st
import os
import tempfile
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="AI Video Captioning",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """Add custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .processing-status {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def display_video_info(uploaded_file):
    """Display video file information"""
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type)

def create_timeline_visualization(scene_data):
    """Create interactive timeline visualization of scenes"""
    if not scene_data:
        return None
    
    df = pd.DataFrame([
        {
            'Scene': f"Scene {i+1}",
            'Start': scene['start_time'],
            'End': scene['end_time'],
            'Duration': scene['end_time'] - scene['start_time'],
            'Caption': scene['caption'][:50] + "..." if len(scene['caption']) > 50 else scene['caption']
        }
        for i, scene in enumerate(scene_data)
    ])
    
    fig = px.timeline(
        df, 
        x_start="Start", 
        x_end="End", 
        y="Scene",
        color="Duration",
        hover_data=["Caption"],
        title="Video Scene Timeline"
    )
    
    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Scenes",
        height=400
    )
    
    return fig

def save_temp_file(uploaded_file, temp_dir):
    """Save uploaded file to temporary directory"""
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def format_timestamp(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def create_download_buttons(results, video_name):
    """Create download buttons for different output formats"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON download
        json_str = json.dumps(results, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"{video_name}_captions.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Text summary download
        summary_text = f"Video Summary:\n{results['summary']}\n\nScene Descriptions:\n"
        for i, scene in enumerate(results['scene_captions']):
            summary_text += f"\nScene {i+1} ({format_timestamp(scene['start_time'])} - {format_timestamp(scene['end_time'])}):\n{scene['caption']}\n"
        
        st.download_button(
            label="📝 Download Text",
            data=summary_text,
            file_name=f"{video_name}_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # SRT download (if available)
        srt_content = generate_srt_content(results['scene_captions'])
        st.download_button(
            label="🎬 Download SRT",
            data=srt_content,
            file_name=f"{video_name}_subtitles.srt",
            mime="text/plain",
            use_container_width=True
        )

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

def format_srt_timestamp(seconds):
    """Format timestamp for SRT format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
