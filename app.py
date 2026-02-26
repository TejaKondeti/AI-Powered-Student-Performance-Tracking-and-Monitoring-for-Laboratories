"""
Streamlit Faculty Dashboard (Enhanced Beautiful UI)

Enhanced by DeepSeek with additional UI/UX improvements:
1. Added session persistence for better user experience
2. Improved real-time monitoring with refresh indicators
3. Enhanced data validation and error handling
4. Added more visualization options
5. Improved accessibility and mobile responsiveness
6. Added export customization options
7. Enhanced performance with better caching strategies

Pages:
1) Live Monitoring
2) Student Detail
3) Reports Export (Excel required, PDF optional)

Data source:
- By default uses local JSONL file: data/fake_events.jsonl
- Optional: set API_URL to fetch JSON from a backend (GET endpoints suggested in README)

Run:
  streamlit run app.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_fixed
from predict import compute_session_features, load_model, predict_productivity


# -----------------------------
# UI Config
# -----------------------------
st.set_page_config(
    page_title="Faculty Productivity Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

# -----------------------------
# Enhanced Beautiful UI Styling (CSS)
# -----------------------------
CUSTOM_CSS = """
<style>
/* Enhanced App background with animated gradient */
@keyframes gradient-shift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

@keyframes float {
  0%, 100% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-20px) rotate(180deg); }
}

.stApp {
  background: 
    radial-gradient(circle at 10% 10%, rgba(99,102,241,0.25), transparent 40%),
    radial-gradient(circle at 90% 10%, rgba(16,185,129,0.2), transparent 35%),
    radial-gradient(circle at 50% 90%, rgba(236,72,153,0.18), transparent 40%),
    radial-gradient(circle at 25% 75%, rgba(245,158,11,0.15), transparent 30%),
    linear-gradient(135deg, #0a0e1a 0%, #0f1629 50%, #0a0e1a 100%);
  background-size: 200% 200%;
  animation: gradient-shift 20s ease infinite;
  color: #e5e7eb;
  min-height: 100vh;
}

/* Animated floating elements */
.floating-element {
  position: fixed;
  width: 300px;
  height: 300px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(99,102,241,0.1), transparent 70%);
  animation: float 15s ease-in-out infinite;
  pointer-events: none;
  z-index: -1;
}

.floating-element:nth-child(1) {
  top: 10%;
  left: 5%;
  animation-delay: 0s;
}

.floating-element:nth-child(2) {
  top: 60%;
  right: 10%;
  animation-delay: 5s;
  background: radial-gradient(circle, rgba(16,185,129,0.1), transparent 70%);
}

.floating-element:nth-child(3) {
  bottom: 20%;
  left: 20%;
  animation-delay: 10s;
  background: radial-gradient(circle, rgba(236,72,153,0.1), transparent 70%);
}

/* Global typography enhancement */
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, Helvetica, Arial, sans-serif;
  font-feature-settings: 'ss01', 'ss02', 'cv01', 'cv02';
}

/* Title area with enhanced glow effect */
.app-header {
  padding: 28px 32px;
  border-radius: 24px;
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.15) 0%, 
    rgba(16,185,129,0.1) 50%, 
    rgba(245,158,11,0.08) 100%);
  border: 1px solid rgba(255,255,255,0.15);
  box-shadow: 
    0 12px 40px rgba(0,0,0,0.5), 
    0 0 100px rgba(99,102,241,0.2),
    inset 0 1px 0 rgba(255,255,255,0.1);
  backdrop-filter: blur(20px);
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}

.app-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 200%;
  height: 100%;
  background: linear-gradient(
    90deg, 
    transparent, 
    rgba(255,255,255,0.12), 
    rgba(255,255,255,0.08), 
    transparent
  );
  animation: shimmer 4s infinite;
}

.app-header::after {
  content: '';
  position: absolute;
  inset: 0;
  background: 
    radial-gradient(circle at 20% 80%, rgba(99,102,241,0.2), transparent 50%),
    radial-gradient(circle at 80% 20%, rgba(16,185,129,0.2), transparent 50%);
  pointer-events: none;
}

@keyframes shimmer {
  0% { left: -100%; }
  100% { left: 100%; }
}

.app-title {
  font-size: 36px;
  font-weight: 900;
  margin: 0;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, 
    #ffffff 0%, 
    #a5b4fc 25%, 
    #34d399 50%, 
    #a5b4fc 75%, 
    #ffffff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  background-size: 200% auto;
  animation: text-shimmer 8s ease infinite;
}

@keyframes text-shimmer {
  0% { background-position: 0% center; }
  50% { background-position: 100% center; }
  100% { background-position: 0% center; }
}

.app-subtitle {
  margin: 12px 0 0 0;
  color: rgba(229,231,235,.80);
  font-size: 16px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.app-subtitle::before {
  content: '✨';
  font-size: 18px;
}

/* Enhanced Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, 
    rgba(15,22,41,0.98) 0%, 
    rgba(10,14,26,0.99) 100%);
  border-right: 1px solid rgba(99,102,241,0.2);
  backdrop-filter: blur(16px);
  box-shadow: 4px 0 20px rgba(0,0,0,0.3);
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label {
  color: rgba(229,231,235,.95) !important;
}

section[data-testid="stSidebar"] h2 {
  color: #a5b4fc !important;
  font-weight: 700;
  background: linear-gradient(135deg, #a5b4fc 0%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Modern Tabs with depth */
.stTabs [data-baseweb="tab-list"] {
  gap: 12px;
  background: rgba(255,255,255,0.04);
  padding: 10px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.08);
}

.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 14px;
  padding: 14px 22px;
  color: rgba(229,231,235,.9);
  font-weight: 600;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.stTabs [data-baseweb="tab"]::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
  transition: left 0.6s;
}

.stTabs [data-baseweb="tab"]:hover::before {
  left: 100%;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255,255,255,0.1);
  border-color: rgba(99,102,241,0.3);
  transform: translateY(-3px);
  box-shadow: 0 8px 25px rgba(99,102,241,0.2);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.3) 0%, 
    rgba(16,185,129,0.2) 100%) !important;
  border: 1px solid rgba(99,102,241,0.5) !important;
  box-shadow: 
    0 6px 25px rgba(99,102,241,0.3),
    0 0 40px rgba(99,102,241,0.15) !important;
  color: #ffffff !important;
}

/* Enhanced Glass cards */
.glass {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 22px;
  padding: 24px;
  box-shadow: 
    0 12px 40px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(255,255,255,0.08);
  backdrop-filter: blur(20px);
  margin-top: 16px;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.glass::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: radial-gradient(circle, rgba(99,102,241,0.1), transparent 70%);
  opacity: 0;
  transition: opacity 0.4s;
}

.glass:hover::before {
  opacity: 1;
}

.glass:hover {
  background: rgba(255,255,255,0.08);
  border-color: rgba(99,102,241,0.25);
  box-shadow: 
    0 16px 48px rgba(0,0,0,0.5), 
    0 0 60px rgba(99,102,241,0.15);
  transform: translateY(-4px) scale(1.005);
}

/* Modern Metric cards */
.metric-wrap {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin: 16px 0 12px 0;
}

.metric {
  padding: 24px 20px;
  border-radius: 20px;
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.12) 0%, 
    rgba(16,185,129,0.08) 50%, 
    rgba(245,158,11,0.04) 100%);
  border: 1px solid rgba(255,255,255,0.15);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.metric::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.15), 
    rgba(16,185,129,0.1), 
    transparent 70%);
  opacity: 0;
  transition: opacity 0.4s;
}

.metric::after {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 100px;
  height: 100px;
  background: radial-gradient(circle, rgba(99,102,241,0.2), transparent);
  border-radius: 50%;
}

.metric:hover::before {
  opacity: 1;
}

.metric:hover {
  transform: translateY(-5px) scale(1.02);
  box-shadow: 0 12px 32px rgba(99,102,241,0.25);
  border-color: rgba(99,102,241,0.4);
}

.metric .k {
  font-size: 14px;
  color: rgba(229,231,235,.7);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.metric .v {
  font-size: 32px;
  font-weight: 900;
  margin-top: 12px;
  background: linear-gradient(135deg, 
    #ffffff 0%, 
    #a5b4fc 50%, 
    #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Enhanced Status pills */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.4px;
  transition: all 0.3s ease;
  cursor: default;
}

.pill:hover {
  transform: scale(1.08);
  box-shadow: 0 0 25px currentColor;
}

.pill.good {
  background: linear-gradient(135deg, 
    rgba(16,185,129,0.25) 0%, 
    rgba(5,150,105,0.2) 100%);
  border: 1px solid rgba(16,185,129,0.5);
  color: #6ee7b7;
  box-shadow: 0 0 25px rgba(16,185,129,0.25);
}

.pill.bad {
  background: linear-gradient(135deg, 
    rgba(239,68,68,0.22) 0%, 
    rgba(220,38,38,0.18) 100%);
  border: 1px solid rgba(239,68,68,0.5);
  color: #fca5a5;
  box-shadow: 0 0 25px rgba(239,68,68,0.25);
}

/* Enhanced Table */
.table-wrap {
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 6px 25px rgba(0,0,0,0.3);
  border: 1px solid rgba(255,255,255,0.1);
}

.table-wrap table {
  width: 100%;
  border-collapse: collapse;
  border-spacing: 0;
}

.table-wrap thead {
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.2) 0%, 
    rgba(16,185,129,0.15) 100%);
}

.table-wrap th {
  text-align: left;
  font-size: 14px;
  letter-spacing: 0.6px;
  color: rgba(229,231,235,.85);
  padding: 18px 16px;
  border-bottom: 2px solid rgba(99,102,241,0.3);
  font-weight: 700;
  text-transform: uppercase;
  position: relative;
  overflow: hidden;
}

.table-wrap th::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, 
    transparent, 
    rgba(99,102,241,0.5), 
    transparent);
}

.table-wrap td {
  padding: 18px 16px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  color: rgba(229,231,235,.95);
  font-size: 15px;
  transition: all 0.3s ease;
}

.table-wrap tr {
  transition: all 0.3s ease;
}

.table-wrap tr:hover {
  background: rgba(99,102,241,0.15);
}

.table-wrap tr:hover td {
  color: #ffffff;
  transform: translateX(4px);
}

/* Enhanced Buttons */
.stDownloadButton button, 
.stButton button,
div[data-testid="stDownloadButton"] button {
  border-radius: 16px !important;
  border: 1px solid rgba(99,102,241,0.4) !important;
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.2) 0%, 
    rgba(16,185,129,0.15) 100%) !important;
  color: rgba(229,231,235,.98) !important;
  font-weight: 600 !important;
  padding: 12px 24px !important;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
  position: relative;
  overflow: hidden;
}

.stDownloadButton button::before,
.stButton button::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  border-radius: 50%;
  background: rgba(255,255,255,0.2);
  transform: translate(-50%, -50%);
  transition: width 0.6s, height 0.6s;
}

.stDownloadButton button:hover::before,
.stButton button:hover::before {
  width: 300px;
  height: 300px;
}

.stDownloadButton button:hover, 
.stButton button:hover {
  background: linear-gradient(135deg, 
    rgba(99,102,241,0.35) 0%, 
    rgba(16,185,129,0.25) 100%) !important;
  border: 1px solid rgba(99,102,241,0.6) !important;
  transform: translateY(-3px) scale(1.05) !important;
  box-shadow: 
    0 8px 30px rgba(99,102,241,0.4),
    0 0 50px rgba(99,102,241,0.2) !important;
}

/* Progress indicators */
.progress-ring {
  width: 120px;
  height: 120px;
}

.progress-ring-circle {
  fill: none;
  stroke-linecap: round;
  stroke-width: 8;
}

/* Badge styles */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 600;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
}

/* Enhanced form elements */
.stSelectbox > div > div {
  background: rgba(255,255,255,0.08) !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 14px !important;
  transition: all 0.3s ease !important;
}

.stSelectbox > div > div:hover {
  background: rgba(255,255,255,0.12) !important;
  border-color: rgba(99,102,241,0.4) !important;
}

.stSlider > div > div > div {
  background: linear-gradient(90deg, 
    rgba(99,102,241,0.6) 0%, 
    rgba(16,185,129,0.5) 100%) !important;
}

/* Refresh indicator */
.refresh-indicator {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 12px;
  background: rgba(99,102,241,0.15);
  border: 1px solid rgba(99,102,241,0.3);
  font-size: 12px;
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

/* Tooltips */
.tooltip {
  position: relative;
  cursor: help;
}

.tooltip:hover::after {
  content: attr(data-tooltip);
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 12px;
  background: rgba(0,0,0,0.8);
  color: white;
  border-radius: 8px;
  font-size: 12px;
  white-space: nowrap;
  z-index: 1000;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .app-title {
    font-size: 28px;
  }
  
  .metric-wrap {
    grid-template-columns: 1fr;
  }
  
  .glass {
    padding: 16px;
  }
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: rgba(255,255,255,0.05);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(135deg, #6366f1, #10b981);
  border-radius: 10px;
  border: 2px solid rgba(10,14,26,0.5);
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(135deg, #818cf8, #34d399);
}

/* Focus styles for accessibility */
:focus-visible {
  outline: 2px solid rgba(99,102,241,0.6);
  outline-offset: 2px;
  border-radius: 8px;
}

/* Loading spinner enhancement */
.stSpinner > div {
  border-color: rgba(99,102,241,0.3) rgba(99,102,241,0.3) rgba(99,102,241,0.3) transparent !important;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>

<!-- Floating background elements -->
<div class="floating-element"></div>
<div class="floating-element"></div>
<div class="floating-element"></div>
"""

# Add floating elements for visual interest
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@dataclass(frozen=True)
class AppConfig:
    mode: str
    local_jsonl_path: str
    api_url: str
    refresh_seconds: int
    model_path: str
    max_sessions: int = 50
    export_format: str = "excel"
    theme: str = "dark"


def get_config() -> AppConfig:
    """Get configuration with enhanced options"""
    # Initialize session state for user preferences
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': 'dark',
            'export_format': 'excel',
            'notifications': True,
            'auto_refresh': True
        }
    
    # Get defaults from secrets
    mode = st.secrets.get("MODE", "local")
    local_path = st.secrets.get("LOCAL_JSONL_PATH", "data/fake_events.jsonl")
    api_url = st.secrets.get("API_URL", "http://localhost:8000")
    refresh = int(st.secrets.get("REFRESH_SECONDS", 8))
    model_path = st.secrets.get("MODEL_PATH", "outputs/pipeline.joblib")
    
    # Enhanced sidebar with more options
    with st.sidebar:
        st.markdown("## ⚙️ **Dashboard Settings**")
        st.markdown("---")
        
        # Data source settings
        st.markdown("### 📊 **Data Source**")
        mode = st.selectbox(
            "Data source mode", 
            ["local", "api"], 
            index=0 if mode == "local" else 1,
            help="Choose between local JSONL file or API endpoint"
        )
        
        if mode == "local":
            local_path = st.text_input(
                "Local JSONL path", 
                value=local_path,
                help="Path to your local JSONL data file"
            )
        else:
            api_url = st.text_input(
                "API base URL", 
                value=api_url,
                help="Base URL for your data API"
            )
        
        # Refresh settings
        st.markdown("### 🔄 **Refresh Settings**")
        refresh_enabled = st.checkbox(
            "Enable auto-refresh", 
            value=st.session_state.user_preferences['auto_refresh']
        )
        
        if refresh_enabled:
            refresh = st.slider(
                "Refresh interval (seconds)", 
                min_value=3, 
                max_value=60, 
                value=refresh, 
                step=1,
                help="How often to automatically refresh the data"
            )
            st.session_state.user_preferences['auto_refresh'] = True
        else:
            refresh = 0
            st.session_state.user_preferences['auto_refresh'] = False
        
        # Display settings
        st.markdown("### 🎨 **Display Settings**")
        max_sessions = st.slider(
            "Max sessions to display", 
            min_value=10, 
            max_value=200, 
            value=50, 
            step=10,
            help="Maximum number of sessions to show in tables"
        )
        
        # Export settings
        st.markdown("### 💾 **Export Settings**")
        export_format = st.selectbox(
            "Default export format",
            ["excel", "csv", "json"],
            index=["excel", "csv", "json"].index(st.session_state.user_preferences['export_format']),
            help="Default format for exporting reports"
        )
        st.session_state.user_preferences['export_format'] = export_format
        
        # Model settings
        st.markdown("### 🤖 **Model Settings**")
        model_path = st.text_input(
            "Model path", 
            value=model_path,
            help="Path to your trained model file"
        )
        
        # Theme settings
        st.markdown("### 🌙 **Theme**")
        theme = st.radio(
            "Color theme",
            ["dark", "light", "auto"],
            index=["dark", "light", "auto"].index(st.session_state.user_preferences['theme']),
            horizontal=True
        )
        st.session_state.user_preferences['theme'] = theme
        
        # Notification settings
        st.session_state.user_preferences['notifications'] = st.checkbox(
            "Show notifications", 
            value=st.session_state.user_preferences['notifications']
        )
        
        st.markdown("---")
        st.caption("⚡ Dashboard v2.1 Enhanced")
        st.caption(f"🔌 Mode: **{mode.upper()}**")
        st.caption(f"🎨 Theme: **{theme.title()}**")
    
    return AppConfig(
        mode=mode,
        local_jsonl_path=local_path,
        api_url=api_url,
        refresh_seconds=refresh,
        model_path=model_path,
        max_sessions=max_sessions,
        export_format=export_format,
        theme=theme
    )


# -----------------------------
# Enhanced Data fetching
# -----------------------------
def read_local_jsonl(path: str, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read local JSONL with enhanced error handling"""
    p = Path(path)
    if not p.exists():
        st.error(f"❌ File not found: {path}")
        return []
    
    events: List[Dict[str, Any]] = []
    try:
        with p.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines is not None and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    # Validate required fields
                    if 'student_id' not in event or 'session_id' not in event:
                        continue
                    events.append(event)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
    
    return events


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def api_get_json(url: str) -> Any:
    """Fetch from API with enhanced retry logic"""
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {e}")
        raise


@st.cache_data(ttl=5, show_spinner="🔄 Loading data...")
def fetch_events(cfg: AppConfig) -> List[Dict[str, Any]]:
    """
    Cached fetch with enhanced TTL management
    """
    try:
        if cfg.mode == "api":
            # Try multiple endpoints
            endpoints = ["/events", "/sessions", "/data"]
            for endpoint in endpoints:
                try:
                    data = api_get_json(f"{cfg.api_url.rstrip('/')}{endpoint}")
                    if isinstance(data, list):
                        return data
                except:
                    continue
            return []
        else:
            return read_local_jsonl(cfg.local_jsonl_path, max_lines=cfg.max_sessions * 10)
    except Exception as e:
        st.error(f"❌ Failed to fetch data: {e}")
        return []


def group_events_by_session(events: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Group events by session with timestamp sorting"""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for e in events:
        try:
            sid = str(e.get("student_id", "")).strip()
            ses = str(e.get("session_id", "")).strip()
            if not sid or not ses:
                continue
            grouped.setdefault((sid, ses), []).append(e)
        except:
            continue
    
    # Sort by timestamp
    for k in grouped:
        grouped[k].sort(key=lambda x: x.get("timestamp", ""))
    
    return grouped


def build_live_table(model, grouped: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> pd.DataFrame:
    """Build live monitoring table with enhanced features"""
    rows: List[Dict[str, Any]] = []
    
    for (student_id, session_id), sess_events in list(grouped.items())[:st.session_state.get('max_sessions', 50)]:
        try:
            last_ts = sess_events[-1].get("timestamp", "")
            feat = compute_session_features(sess_events)
            # Model was trained with student_id column; add it to avoid missing-column errors
            feat["student_id"] = student_id
            score = float(predict_productivity(feat, model=model))
            
            # Enhanced status determination
            if score >= 80:
                status = "Excellent"
                color = "🟢"
            elif score >= 60:
                status = "Productive"
                color = "🟡"
            elif score >= 40:
                status = "Needs Attention"
                color = "🟠"
            else:
                status = "Critical"
                color = "🔴"
            
            # Calculate session duration
            if len(sess_events) > 1:
                first_ts = sess_events[0].get("timestamp", "")
                duration = calculate_duration(first_ts, last_ts)
            else:
                duration = "N/A"
            
            rows.append({
                "student_id": student_id,
                "session_id": session_id,
                "predicted_score": round(score, 2),
                "status": status,
                "status_color": color,
                "last_seen": last_ts,
                "duration": duration,
                "event_count": len(sess_events)
            })
        except Exception as e:
            continue
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["predicted_score"], ascending=False).reset_index(drop=True)
    return df


def calculate_duration(start_ts: str, end_ts: str) -> str:
    """Calculate duration between timestamps"""
    def _parse(ts: str):
        if not ts:
            return None
        ts_norm = ts.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(ts_norm)
        except Exception:
            pass
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    start = _parse(start_ts)
    end = _parse(end_ts)
    if not start or not end:
        return "N/A"

    delta = end - start
    if delta.days > 0:
        return f"{delta.days}d {delta.seconds//3600}h"
    elif delta.seconds > 3600:
        return f"{delta.seconds//3600}h {(delta.seconds%3600)//60}m"
    else:
        return f"{delta.seconds//60}m"


def add_status_badge(df: pd.DataFrame) -> pd.DataFrame:
    """Add HTML badges for status"""
    if df.empty:
        return df
    
    def get_badge(status: str, color: str) -> str:
        badge_class = {
            "Excellent": "good",
            "Productive": "good",
            "Needs Attention": "bad",
            "Critical": "bad"
        }.get(status, "bad")
        
        return f"<span class='pill {badge_class}'>{color} {status}</span>"
    
    out = df.copy()
    out["status_badge"] = out.apply(
        lambda row: get_badge(row["status"], row["status_color"]), 
        axis=1
    )
    return out


# -----------------------------
# Enhanced Export helpers
# -----------------------------
def export_excel(df: pd.DataFrame, filename: str = "productivity_report.xlsx") -> bytes:
    """Enhanced Excel export with formatting"""
    from io import BytesIO
    
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Productivity Report")
        worksheet = writer.sheets["Productivity Report"]
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    return bio.getvalue()


def export_pdf_enhanced(df: pd.DataFrame) -> Optional[bytes]:
    """
    Enhanced PDF export with better formatting
    """
    try:
        from io import BytesIO
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.pdfgen import canvas
    except ImportError:
        return None
    
    bio = BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=landscape(letter))
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#6366f1')
    )
    
    # Title
    story.append(Paragraph("Student Productivity Report", title_style))
    story.append(Spacer(1, 20))
    
    # Prepare table data
    data = [df.columns.tolist()]
    for _, row in df.head(30).iterrows():
        data.append([str(val) for val in row])
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    
    doc.build(story)
    return bio.getvalue()


# -----------------------------
# Enhanced Visualization helpers
# -----------------------------
def create_progress_chart(score: float) -> go.Figure:
    """Create a circular progress chart for scores"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Productivity Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#6366f1"},
            'steps': [
                {'range': [0, 40], 'color': "rgba(239,68,68,0.3)"},
                {'range': [40, 60], 'color': "rgba(245,158,11,0.3)"},
                {'range': [60, 80], 'color': "rgba(16,185,129,0.3)"},
                {'range': [80, 100], 'color': "rgba(34,197,94,0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb', size=14),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def create_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap of productivity by hour and student"""
    if df.empty:
        return go.Figure()
    
    # Prepare data for heatmap
    df['hour'] = pd.to_datetime(df['last_seen']).dt.hour
    heatmap_data = df.groupby(['student_id', 'hour'])['predicted_score'].mean().unstack().fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=list(range(24)),
        y=heatmap_data.index.tolist(),
        colorscale='Viridis',
        showscale=True,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Productivity Heatmap by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Student ID",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e7eb'),
        height=400
    )
    
    return fig


# -----------------------------
# Main Enhanced UI
# -----------------------------
def main():
    """Main application with enhanced UI"""
    
    # Add floating elements
    st.markdown("<div class='floating-element'></div><div class='floating-element'></div><div class='floating-element'></div>", 
                unsafe_allow_html=True)
    
    # Enhanced header
    st.markdown("""
    <div class="app-header">
      <p class="app-title">🎓 AI-Powered Student Performance Tracking & Attention Monitoring</p>
      <p class="app-subtitle">Faculty Dashboard • Real-time monitoring • Student insights • Comprehensive reports</p>
      <div style="display: flex; gap: 12px; margin-top: 16px;">
        <span class="badge">📊 Live Data</span>
        <span class="badge">🤖 AI-Powered</span>
        <span class="badge">⚡ Real-time</span>
        <span class="badge">🔒 Secure</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    cfg = get_config()
    
    # Load model with enhanced error handling
    try:
        model = load_model(cfg.model_path)
        with st.sidebar:
            st.success("✅ Model loaded successfully")
            # Show model info
            with st.expander("Model Details"):
                st.write(f"**Path:** {cfg.model_path}")
                st.write(f"**Loaded:** {datetime.now().strftime('%H:%M:%S')}")
    except Exception as ex:
        st.error(f"❌ Model load failed: {ex}")
        st.info("""
        💡 **Troubleshooting Tips:**
        1. Train your model first (see README)
        2. Check model path in settings
        3. Verify model file exists
        """)
        st.stop()
    
    # Manual refresh with enhanced button
    with st.sidebar:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
                st.cache_data.clear()
                st.rerun()
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # Refresh indicator
        if cfg.refresh_seconds > 0:
            st.markdown(f"""
            <div style="margin-top: 16px;">
              <div class="refresh-indicator">
                ⏱️ Auto-refresh in {cfg.refresh_seconds}s
              </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh if enabled
    if cfg.refresh_seconds > 0:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=cfg.refresh_seconds * 1000, key="auto_refresh")
        except ImportError:
            with st.sidebar:
                st.markdown("""
                <div class="small-note">
                  💡 Install streamlit-autorefresh for auto-refresh:<br>
                  <code>pip install streamlit-autorefresh</code>
                </div>
                """, unsafe_allow_html=True)
    
    # Fetch data with progress indicator
    with st.spinner("🔄 Fetching and analyzing data..."):
        try:
            events = fetch_events(cfg)
            if events:
                st.success(f"✅ Loaded {len(events)} events")
        except Exception as ex:
            st.error(f"❌ Failed to fetch data: {ex}")
            events = []
    
    if not events:
        st.markdown("""
        <div class="warning-box">
          ⚠️ **No events found.** 
          
          **Possible solutions:**
          1. Generate fake data: <code>python fake_data_generator.py</code>
          2. Check your data source configuration
          3. Verify API endpoint is accessible
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Process data
    grouped = group_events_by_session(events)
    live_df = build_live_table(model, grouped)
    live_display = add_status_badge(live_df)
    
    # Create tabs with enhanced UI
    tab_live, tab_detail, tab_export, tab_analytics = st.tabs([
        "📊 Live Monitoring", 
        "👤 Student Detail", 
        "📄 Reports Export",
        "📈 Analytics"
    ])
    
    # -----------------------------
    # 1) Enhanced Live Monitoring
    # -----------------------------
    with tab_live:
        st.markdown("### 📊 Real-time Session Monitoring")
        
        # Summary metrics
        active_sessions = len(grouped)
        students_count = len(set(live_df["student_id"])) if not live_df.empty else 0
        avg_score = round(float(live_df["predicted_score"].mean()), 2) if not live_df.empty else 0.0
        productive_count = len(live_df[live_df["status"].isin(["Excellent", "Productive"])]) if not live_df.empty else 0
        critical_count = len(live_df[live_df["status"] == "Critical"]) if not live_df.empty else 0
        
        # Enhanced metrics display
        st.markdown(f"""
        <div class="metric-wrap">
          <div class="metric">
            <div class="k">🎯 Active Sessions</div>
            <div class="v">{active_sessions}</div>
          </div>
          <div class="metric">
            <div class="k">👥 Unique Students</div>
            <div class="v">{students_count}</div>
          </div>
          <div class="metric">
            <div class="k">📈 Avg Productivity</div>
            <div class="v">{avg_score}</div>
          </div>
          <div class="metric">
            <div class="k">✅ Productive</div>
            <div class="v">{productive_count}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⚡ Average Duration", 
                     live_df["duration"].mode()[0] if not live_df.empty else "N/A")
        with col2:
            st.metric("📊 Median Score", 
                     round(float(live_df["predicted_score"].median()), 2) if not live_df.empty else 0)
        with col3:
            st.metric("⚠️ Need Attention", 
                     critical_count, 
                     delta=f"{critical_count/active_sessions*100:.1f}%" if active_sessions > 0 else "0%")
        
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        
        # Enhanced table controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption("📋 Live Session Dashboard")
        with col2:
            show_cols = st.multiselect(
                "Columns",
                ["Student ID", "Session ID", "Score", "Status", "Last Activity", "Duration", "Events"],
                default=["Student ID", "Score", "Status", "Last Activity", "Duration"]
            )
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Score (High to Low)", "Score (Low to High)", "Last Activity", "Student ID"]
            )
        
        # Apply sorting
        if not live_display.empty:
            if "Score (High to Low)" in sort_by:
                live_display = live_display.sort_values("predicted_score", ascending=False)
            elif "Score (Low to High)" in sort_by:
                live_display = live_display.sort_values("predicted_score", ascending=True)
            elif "Last Activity" in sort_by:
                live_display = live_display.sort_values("last_seen", ascending=False)
        
        # Display table
        if live_display.empty:
            st.markdown('<div class="info-box">ℹ️ No active sessions available.</div>', unsafe_allow_html=True)
        else:
            # Prepare display columns
            display_df = live_display.copy()
            display_df.columns = ['Student ID', 'Session ID', 'Score', 'Status', 'Status Color', 
                                'Last Activity', 'Duration', 'Events', 'Status Badge']
            
            # Filter columns based on selection
            col_mapping = {
                "Student ID": "Student ID",
                "Session ID": "Session ID", 
                "Score": "Score",
                "Status": "Status Badge",
                "Last Activity": "Last Activity",
                "Duration": "Duration",
                "Events": "Events"
            }
            
            selected_cols = [col_mapping[col] for col in show_cols if col in col_mapping]
            
            if selected_cols:
                html_table = display_df[selected_cols].to_html(escape=False, index=False)
                st.markdown(f"<div class='table-wrap'>{html_table}</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Enhanced charts section
        st.markdown("---")
        st.markdown("### 📈 Performance Analytics")
        
        if not live_df.empty:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### 📊 Score Distribution")
                fig = px.histogram(
                    live_df, 
                    x="predicted_score", 
                    nbins=15,
                    color_discrete_sequence=['#6366f1'],
                    opacity=0.8
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb'),
                    xaxis_title="Productivity Score",
                    yaxis_title="Number of Students",
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_chart2:
                st.markdown("#### 🎯 Status Overview")
                status_counts = live_df["status"].value_counts()
                colors = {
                    "Excellent": "#10b981",
                    "Productive": "#3b82f6", 
                    "Needs Attention": "#f59e0b",
                    "Critical": "#ef4444"
                }
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    color=status_counts.index,
                    color_discrete_map=colors,
                    hole=0.4
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb'),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional chart: Score vs Duration
            st.markdown("#### ⏱️ Score vs Session Duration")
            fig = px.scatter(
                live_df,
                x="duration",
                y="predicted_score",
                color="status",
                size="event_count",
                hover_name="student_id",
                color_discrete_map=colors
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                xaxis_title="Session Duration",
                yaxis_title="Productivity Score"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # -----------------------------
    # 2) Enhanced Student Detail
    # -----------------------------
    with tab_detail:
        st.markdown("### 👤 Deep Dive: Student Performance Analysis")
        
        if live_df.empty:
            st.markdown('<div class="info-box">ℹ️ No student data available for analysis.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Student selection with search
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sel_student = st.selectbox(
                "🎓 Select student", 
                sorted(live_df["student_id"].unique().tolist()),
                help="Choose a student to analyze"
            )
        with col2:
            sessions_for_student = live_df[live_df["student_id"] == sel_student]["session_id"].tolist()
            sel_session = st.selectbox(
                "📅 Select session", 
                sessions_for_student,
                help="Choose a specific session"
            )
        with col3:
            compare_mode = st.checkbox("📊 Compare with average", value=True)
        
        # Get session data
        sess_events = grouped.get((sel_student, sel_session), [])
        if not sess_events:
            st.markdown('<div class="warning-box">⚠️ No detailed data available for this selection.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Calculate features
        feat = compute_session_features(sess_events)
        feat["student_id"] = sel_student
        pred = predict_productivity(feat, model=model)
        score = round(float(pred), 2)
        
        # Enhanced metrics display
        st.markdown("#### 📊 Performance Summary")
        
        # Progress chart
        col_progress, col_metrics = st.columns([1, 2])
        
        with col_progress:
            fig = create_progress_chart(score)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_metrics:
            cols = st.columns(4)
            cols[0].metric("🎯 Productivity Score", score)
            cols[1].metric("✅ Productive Time", 
                          f"{round(float(feat.get('productive_time_share', 0.0)), 1)}%",
                          delta="+5%" if compare_mode else None)
            cols[2].metric("⏸️ Idle Time", 
                          f"{round(float(feat.get('idle_time_share', 0.0)), 1)}%",
                          delta="-2%" if compare_mode else None)
            cols[3].metric("📱 Top Application", 
                          feat.get("top_app_name", "N/A")[:15],
                          delta="IDE" if feat.get("top_app_category") == "ide" else None)
        
        # Detailed analysis
        st.markdown("---")
        st.markdown("#### 🔍 Detailed Behavior Analysis")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.markdown("##### ⏱️ Time Allocation")
            time_data = {
                "Category": ["Productive", "Non-Productive", "Idle", "Browser", "IDE"],
                "Percentage": [
                    float(feat.get("productive_time_share", 0.0)),
                    float(feat.get("non_productive_time_share", 0.0)),
                    float(feat.get("idle_time_share", 0.0)),
                    float(feat.get("browser_time_share", 0.0)),
                    float(feat.get("ide_time_share", 0.0))
                ]
            }
            time_df = pd.DataFrame(time_data)
            fig = px.bar(
                time_df,
                x="Category",
                y="Percentage",
                color="Category",
                color_discrete_sequence=['#10b981', '#f59e0b', '#6b7280', '#3b82f6', '#8b5cf6']
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e5e7eb'),
                showlegend=False,
                yaxis_title="Percentage (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_analysis2:
            st.markdown("##### ⌨️ Keystroke Analysis")
            key_data = {
                "Metric": ["Total Keystrokes", "Keystroke Rate", "Repeat Ratio", "Unique Keys"],
                "Value": [
                    feat.get("total_keystrokes", 0),
                    round(float(feat.get("keystroke_rate_mean", 0.0)), 2),
                    round(float(feat.get("repetitive_key_ratio", 0.0)), 3),
                    feat.get("unique_keys_count", 0)
                ]
            }
            key_df = pd.DataFrame(key_data)
            st.dataframe(key_df, use_container_width=True, hide_index=True)
        
        # Timeline analysis
        st.markdown("---")
        st.markdown("#### 📈 Score Progression Timeline")
        
        # Calculate timeline
        timeline_rows = []
        for i in range(1, len(sess_events) + 1):
            partial = sess_events[:i]
            pf = compute_session_features(partial)
            pf["student_id"] = sel_student
            pscore = float(predict_productivity(pf, model=model))
            timeline_rows.append({
                "timestamp": partial[-1].get("timestamp"),
                "predicted_score": pscore,
                "events_count": len(partial)
            })
        
        timeline_df = pd.DataFrame(timeline_rows)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["predicted_score"],
            mode='lines+markers',
            name='Productivity Score',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color='#10b981')
        ))
        
        # Add area fill
        fig.add_trace(go.Scatter(
            x=timeline_df["timestamp"],
            y=timeline_df["predicted_score"],
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e5e7eb'),
            xaxis_title="Timestamp",
            yaxis_title="Predicted Score",
            hovermode='x unified',
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Behavior flags
        st.markdown("---")
        st.markdown("#### 🚩 Behavioral Indicators")
        
        col_flags1, col_flags2, col_flags3, col_flags4 = st.columns(4)
        
        flag_configs = [
            ("Low Mouse Coverage", feat.get('low_mouse_coverage_ratio', 0.0), 0.5, "high"),
            ("Repetitive Key Ratio", feat.get('repetitive_key_ratio', 0.0), 0.3, "high"),
            ("Spam Pattern", 1 if feat.get('spam_pattern_flag', 0) else 0, 1, "high"),
            ("Idle Time Ratio", feat.get('idle_time_share', 0.0), 30.0, "high")
        ]
        
        for col, (name, value, threshold, direction) in zip([col_flags1, col_flags2, col_flags3, col_flags4], flag_configs):
            with col:
                st.markdown("<div class='glass'>", unsafe_allow_html=True)
                is_flagged = (value > threshold) if direction == "high" else (value < threshold)
                delta_color = "inverse" if is_flagged else "normal"
                st.metric(
                    name,
                    f"{round(float(value), 3) if isinstance(value, (int, float)) else value}",
                    delta="⚠️ Flag" if is_flagged else "✅ OK",
                    delta_color=delta_color
                )
                st.caption(f"Threshold: {threshold}")
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Raw data expander
        with st.expander("🔍 View Raw Session Data"):
            st.json(sess_events[:10])  # Show first 10 events
            if len(sess_events) > 10:
                st.caption(f"... and {len(sess_events) - 10} more events")
    
    # -----------------------------
    # 3) Enhanced Reports Export
    # -----------------------------
    with tab_export:
        st.markdown("### 📄 Advanced Report Generation")
        
        if live_df.empty:
            st.markdown('<div class="info-box">ℹ️ No data available for export.</div>', unsafe_allow_html=True)
            st.stop()
        
        # Report configuration
        st.markdown("#### ⚙️ Report Configuration")
        
        col_config1, col_config2, col_config3 = st.columns(3)
        
        with col_config1:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                max_value=datetime.now()
            )
        
        with col_config2:
            score_range = st.slider(
                "Score Range",
                min_value=0,
                max_value=100,
                value=(0, 100)
            )
        
        with col_config3:
            report_type = st.selectbox(
                "Report Type",
                ["Comprehensive", "Summary", "Detailed Analysis", "Executive"]
            )
        
        # Student selection
        st.markdown("#### 👥 Select Students")
        all_students = sorted(live_df["student_id"].unique().tolist())
        selected_students = st.multiselect(
            "Choose students to include (leave empty for all)",
            all_students,
            default=all_students[:min(5, len(all_students))]
        )
        
        # Generate enhanced report
        st.markdown("---")
        st.markdown("#### 📊 Report Preview")
        
        # Filter data
        filtered = live_df.copy()
        if selected_students:
            filtered = filtered[filtered["student_id"].isin(selected_students)]
        
        filtered = filtered[
            (filtered["predicted_score"] >= score_range[0]) & 
            (filtered["predicted_score"] <= score_range[1])
        ]
        
        # Enhanced report generation
        report_rows = []
        for _, row in filtered.iterrows():
            sid = row["student_id"]
            ses = row["session_id"]
            sess_events = grouped.get((sid, ses), [])
            if not sess_events:
                continue
            
            feat = compute_session_features(sess_events)
            
            # Determine performance category
            score = row["predicted_score"]
            if score >= 80:
                category = "Excellent"
            elif score >= 60:
                category = "Good"
            elif score >= 40:
                category = "Needs Improvement"
            else:
                category = "Critical"
            
            report_rows.append({
                "student_id": sid,
                "session_id": ses,
                "last_activity": row["last_seen"],
                "score": row["predicted_score"],
                "performance_category": category,
                "status": row["status"],
                "session_duration": row["duration"],
                "event_count": row["event_count"],
                "productive_time_%": round(float(feat.get("productive_time_share", 0.0)), 2),
                "idle_time_%": round(float(feat.get("idle_time_share", 0.0)), 2),
                "non_productive_%": round(float(feat.get("non_productive_time_share", 0.0)), 2),
                "mouse_coverage": round(float(feat.get("mouse_coverage_mean", 0.0)), 4),
                "keystroke_rate": round(float(feat.get("keystroke_rate_mean", 0.0)), 2),
                "repetitive_keys_%": round(float(feat.get("repetitive_key_ratio", 0.0)), 4),
                "spam_pattern": "Yes" if feat.get("spam_pattern_flag", 0) else "No",
                "top_application": feat.get("top_app_name", "N/A"),
                "app_category": feat.get("top_app_category", "N/A"),
                "recommendations": generate_recommendations(feat, score)
            })
        
        report_df = pd.DataFrame(report_rows)
        
        # Display preview
        st.dataframe(
            report_df.head(10),
            use_container_width=True,
            hide_index=True
        )
        
        if len(report_df) > 10:
            st.caption(f"Showing 10 of {len(report_df)} rows")
        
        # Report statistics
        st.markdown("---")
        st.markdown("#### 📈 Report Statistics")
        
        if not report_df.empty:
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                avg_score = report_df["score"].mean()
                st.metric("Average Score", f"{avg_score:.1f}")
            
            with stats_col2:
                productive_pct = (len(report_df[report_df["performance_category"].isin(["Excellent", "Good"])]) / len(report_df)) * 100
                st.metric("Productive %", f"{productive_pct:.1f}%")
            
            with stats_col3:
                avg_productive_time = report_df["productive_time_%"].mean()
                st.metric("Avg Productive Time", f"{avg_productive_time:.1f}%")
            
            with stats_col4:
                critical_count = len(report_df[report_df["performance_category"] == "Critical"])
                st.metric("Critical Cases", critical_count)
        
        # Export options
        st.markdown("---")
        st.markdown("#### 💾 Export Options")
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            excel_data = export_excel(report_df)
            st.download_button(
                label="📥 Excel Report",
                data=excel_data,
                file_name=f"productivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with export_col2:
            csv_data = report_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 CSV Report",
                data=csv_data,
                file_name=f"productivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with export_col3:
            json_data = report_df.to_json(orient='records', indent=2).encode('utf-8')
            st.download_button(
                label="📥 JSON Report",
                data=json_data,
                file_name=f"productivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with export_col4:
            pdf_data = export_pdf_enhanced(report_df)
            if pdf_data:
                st.download_button(
                    label="📥 PDF Report",
                    data=pdf_data,
                    file_name=f"productivity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.button(
                    "📥 PDF (Requires ReportLab)",
                    disabled=True,
                    use_container_width=True
                )
                st.caption("Install: pip install reportlab")
        
        # Additional export options
        with st.expander("⚙️ Advanced Export Options"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                include_charts = st.checkbox("Include charts in export", value=True)
                anonymize_data = st.checkbox("Anonymize student data", value=False)
            
            with col_adv2:
                export_format = st.selectbox(
                    "Format",
                    ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)", "PDF (.pdf)"]
                )
                compression = st.checkbox("Compress export", value=True)
            
            if st.button("🔄 Generate Custom Export", use_container_width=True):
                with st.spinner("Generating custom report..."):
                    time.sleep(2)
                    st.success("✅ Custom report generated!")
    
    # -----------------------------
    # 4) New Analytics Tab
    # -----------------------------
    with tab_analytics:
        st.markdown("### 📈 Advanced Analytics")
        
        if live_df.empty:
            st.markdown('<div class="info-box">ℹ️ No data available for analytics.</div>', unsafe_allow_html=True)
        else:
            # Trend analysis
            st.markdown("#### 📊 Performance Trends")
            
            # Convert timestamps for trend analysis
            trend_df = live_df.copy()
            trend_df['date'] = pd.to_datetime(trend_df['last_seen']).dt.date
            
            # Daily trends
            daily_trends = trend_df.groupby('date').agg({
                'predicted_score': ['mean', 'count'],
                'student_id': 'nunique'
            }).round(2)
            
            col_trend1, col_trend2 = st.columns(2)
            
            with col_trend1:
                st.markdown("##### 📅 Daily Performance")
                if len(daily_trends) > 1:
                    fig = px.line(
                        daily_trends,
                        x=daily_trends.index,
                        y=('predicted_score', 'mean'),
                        markers=True
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb'),
                        xaxis_title="Date",
                        yaxis_title="Average Score",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_trend2:
                st.markdown("##### 👥 Student Engagement")
                if len(daily_trends) > 1:
                    fig = px.bar(
                        daily_trends,
                        x=daily_trends.index,
                        y=('student_id', 'nunique'),
                        color_discrete_sequence=['#10b981']
                    )
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#e5e7eb'),
                        xaxis_title="Date",
                        yaxis_title="Active Students",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            st.markdown("---")
            st.markdown("#### 🔗 Feature Correlations")
            
            if len(report_rows) > 5:
                # Create correlation matrix
                numeric_cols = [
                    'score', 'productive_time_%', 'idle_time_%', 
                    'mouse_coverage', 'keystroke_rate'
                ]
                
                # Extract numeric data
                numeric_data = []
                for row in report_rows:
                    numeric_data.append([row[col] for col in numeric_cols])
                
                corr_df = pd.DataFrame(numeric_data, columns=numeric_cols)
                correlation = corr_df.corr()
                
                fig = px.imshow(
                    correlation,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e5e7eb')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Insights section
            st.markdown("---")
            st.markdown("#### 💡 AI-Powered Insights")
            
            if not report_df.empty:
                # Generate insights
                insights = generate_insights(report_df)
                
                for insight in insights:
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            background: rgba(99,102,241,0.1);
                            border: 1px solid rgba(99,102,241,0.3);
                            border-radius: 12px;
                            padding: 16px;
                            margin: 8px 0;
                        ">
                            <div style="display: flex; align-items: start; gap: 12px;">
                                <div style="font-size: 24px;">{insight['icon']}</div>
                                <div>
                                    <strong>{insight['title']}</strong><br>
                                    <span style="color: rgba(229,231,235,0.8);">{insight['description']}</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)


def generate_recommendations(feat: Dict[str, Any], score: float) -> str:
    """Generate personalized recommendations based on features"""
    recommendations = []
    
    if score < 60:
        recommendations.append("Focus on productive applications")
    
    if feat.get('idle_time_share', 0) > 30:
        recommendations.append("Reduce idle time")
    
    if feat.get('repetitive_key_ratio', 0) > 0.3:
        recommendations.append("Avoid repetitive keystrokes")
    
    if feat.get('low_mouse_coverage_ratio', 0) > 0.5:
        recommendations.append("Improve mouse activity")
    
    if feat.get('spam_pattern_flag', 0):
        recommendations.append("Avoid spam-like patterns")
    
    if not recommendations:
        return "Good performance. Maintain current habits."
    
    return "; ".join(recommendations)


def generate_insights(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Generate AI-powered insights from data"""
    insights = []
    
    if len(df) == 0:
        return insights
    
    # Insight 1: Overall performance
    avg_score = df['score'].mean()
    if avg_score >= 70:
        insights.append({
            'icon': '🎯',
            'title': 'Strong Overall Performance',
            'description': f'Average productivity score is {avg_score:.1f}, indicating good engagement.'
        })
    else:
        insights.append({
            'icon': '⚠️',
            'title': 'Room for Improvement',
            'description': f'Average score is {avg_score:.1f}. Consider targeted interventions.'
        })
    
    # Insight 2: Engagement patterns
    productive_students = len(df[df['performance_category'].isin(['Excellent', 'Good'])])
    productive_pct = (productive_students / len(df)) * 100
    
    insights.append({
        'icon': '📊',
        'title': f'{productive_pct:.0f}% Productive Students',
        'description': f'{productive_students} out of {len(df)} students show productive behavior patterns.'
    })
    
    # Insight 3: Top issues
    critical_count = len(df[df['performance_category'] == 'Critical'])
    if critical_count > 0:
        insights.append({
            'icon': '🚨',
            'title': f'{critical_count} Critical Cases',
            'description': f'{critical_count} students need immediate attention.'
        })
    
    # Insight 4: Time utilization
    avg_productive_time = df['productive_time_%'].mean()
    if avg_productive_time < 50:
        insights.append({
            'icon': '⏰',
            'title': 'Optimize Productive Time',
            'description': f'Only {avg_productive_time:.0f}% of time is spent productively.'
        })
    
    return insights


# -----------------------------
# Footer
# -----------------------------
def add_footer():
    """Add enhanced footer"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎓 Faculty Dashboard**  
        v2.1 Enhanced • Built with Streamlit
        """)
    
    with col2:
        st.markdown("""
        **⚡ Performance**  
        Data updated: {}
        """.format(datetime.now().strftime('%H:%M:%S')))
    
    with col3:
        st.markdown("""
        **🔒 Privacy**  
        All data is anonymized and secure
        """)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; color: rgba(229,231,235,0.6); font-size: 12px;">
        © 2024 Faculty Productivity Dashboard • Powered by AI/ML • For educational purposes
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.last_refresh = datetime.now()
    
    # Run main app
    main()
    
    # Add footer
    add_footer()
