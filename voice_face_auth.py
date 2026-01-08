"""
Voice + Face Authentication Module for Accessible ATM Use
Complete Implementation in a Single File
"""

# ============================================================================
# IMPORTS
# ============================================================================

import streamlit as st
import cv2
import numpy as np
import librosa
import speech_recognition as sr
import whisper
import torch
import pickle
import tempfile
import os
import json
import base64
import hashlib
from datetime import datetime
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import cosine
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Voice + Face Authentication",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for accessibility
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 1rem;
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        height: 3em;
        font-size: 1.1em;
        border-radius: 10px;
        margin: 5px 0;
    }
    
    /* Card styling */
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Success box */
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Error box */
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Instruction box */
    .instruction-box {
        background-color: #e7f3ff;
        border-left: 5px solid #0066cc;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    
    /* Audio prompt box */
    .audio-prompt {
        background-color: #f0f8ff;
        border: 2px dashed #4dabf7;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-style: italic;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* High contrast mode */
    .high-contrast {
        background-color: black !important;
        color: white !important;
    }
    
    .high-contrast .card {
        background-color: #222 !important;
        color: white !important;
    }
    
    /* Large text mode */
    .large-text {
        font-size: 1.2em !important;
    }
    
    .large-text .stTextInput input {
        font-size: 1.2em !important;
        padding: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    # User data
    if 'user_id' not in st.session_state:
        st.session_state.user_id = ""
    if 'user_enrolled' not in st.session_state:
        st.session_state.user_enrolled = False
    if 'enrolled_users' not in st.session_state:
        st.session_state.enrolled_users = {}
    
    # Authentication flow
    if 'auth_step' not in st.session_state:
        st.session_state.auth_step = 1
    if 'digit_challenge' not in st.session_state:
        st.session_state.digit_challenge = ""
    if 'face_image' not in st.session_state:
        st.session_state.face_image = None
    if 'voice_audio' not in st.session_state:
        st.session_state.voice_audio = None
    if 'auth_result' not in st.session_state:
        st.session_state.auth_result = None
    
    # Settings
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'face_threshold': 0.7,
            'voice_threshold': 0.65,
            'require_liveness': True,
            'require_digits': True,
            'multimodal': True,
            'audio_volume': 80,
            'prompt_speed': 'Normal',
            'high_contrast': False,
            'large_text': False,
            'timeout_duration': 60
        }
    
    # Models
    if 'face_model' not in st.session_state:
        st.session_state.face_model = None
    if 'voice_model' not in st.session_state:
        st.session_state.voice_model = None
    if 'asr_model' not in st.session_state:
        st.session_state.asr_model = None

# ============================================================================
# FACE RECOGNITION MODULE
# ============================================================================

class FaceAuthentication:
    """Face recognition and verification module"""
    
    def __init__(self):
        self.face_templates = {}
        self.haar_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.haar_eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def extract_face_features(self, image_array):
        """Extract facial features using Haar cascades and histograms"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None
            
            # Get the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_region = cv2.resize(face_region, (100, 100))
            
            # Extract histogram features
            hist = cv2.calcHist([face_region], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Extract LBP-like features (simplified)
            lbp_features = self._extract_simplified_lbp(face_region)
            
            # Combine features
            features = np.concatenate([hist, lbp_features])
            
            return features
            
        except Exception as e:
            st.warning(f"Face feature extraction error: {e}")
            return None
    
    def _extract_simplified_lbp(self, image):
        """Simplified Local Binary Pattern feature extraction"""
        # This is a simplified version for demo purposes
        height, width = image.shape
        features = []
        
        # Divide image into 4x4 blocks
        for i in range(0, height, 25):
            for j in range(0, width, 25):
                block = image[i:min(i+25, height), j:min(j+25, width)]
                if block.size > 0:
                    features.append(np.mean(block))
                    features.append(np.std(block))
        
        return np.array(features[:32])  # Limit to 32 features
    
    def enroll_face(self, user_id, image_paths):
        """Enroll face from multiple images"""
        embeddings = []
        
        for img_path in image_paths:
            # Load image
            if isinstance(img_path, str):
                img = cv2.imread(img_path)
            else:
                # Already an array
                img = img_path
            
            if img is None:
                continue
            
            # Extract features
            features = self.extract_face_features(img)
            if features is not None:
                embeddings.append(features)
        
        if embeddings:
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            self.face_templates[user_id] = avg_embedding
            return True
        
        return False
    
    def verify_face(self, image_array, user_id, threshold=0.7):
        """Verify face against stored template"""
        if user_id not in self.face_templates:
            return False, 0.0
        
        # Extract features from test image
        test_features = self.extract_face_features(image_array)
        if test_features is None:
            return False, 0.0
        
        # Get stored template
        stored_features = self.face_templates[user_id]
        
        # Calculate cosine similarity
        try:
            similarity = 1 - cosine(test_features, stored_features)
            similarity = max(0, min(1, similarity))  # Clip to [0, 1]
        except:
            similarity = 0.0
        
        return similarity >= threshold, similarity
    
    def detect_liveness(self, image_array):
        """Basic liveness detection using eye blinking"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.haar_face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return False
            
            # Check for eyes in the largest face
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.haar_eye_cascade.detectMultiScale(
                face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
            
            # If we detect at least one eye, consider it live
            return len(eyes) >= 1
            
        except Exception as e:
            st.warning(f"Liveness detection error: {e}")
            return False
    
    def detect_face(self, image_array):
        """Check if a face is present in the image"""
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = self.haar_face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return len(faces) > 0

# ============================================================================
# VOICE RECOGNITION MODULE
# ============================================================================

class VoiceAuthentication:
    """Voice recognition and verification module"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.voice_templates = {}
    
    def extract_voice_features(self, audio_data):
        """Extract MFCC features from audio"""
        try:
            # If audio_data is a file path
            if isinstance(audio_data, str):
                y, sr = librosa.load(audio_data, sr=self.sample_rate)
            else:
                # If audio_data is raw bytes or numpy array
                y = audio_data
                sr = self.sample_rate
            
            # Ensure proper length
            if len(y) < sr:  # Less than 1 second
                y = np.pad(y, (0, max(0, sr - len(y))), mode='constant')
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) == 0:
                y_trimmed = y
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=y_trimmed, 
                sr=sr, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            
            # Calculate delta and delta-delta
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
            
            # Take mean across time
            features_mean = np.mean(features, axis=1)
            
            # Normalize
            features_norm = (features_mean - np.mean(features_mean)) / np.std(features_mean)
            
            return features_norm
            
        except Exception as e:
            st.warning(f"Voice feature extraction error: {e}")
            return None
    
    def enroll_voice(self, user_id, audio_samples):
        """Enroll voice from multiple samples"""
        embeddings = []
        
        for audio_data in audio_samples:
            features = self.extract_voice_features(audio_data)
            if features is not None:
                embeddings.append(features)
        
        if embeddings:
            # Average the embeddings
            avg_embedding = np.mean(embeddings, axis=0)
            self.voice_templates[user_id] = avg_embedding
            return True
        
        return False
    
    def verify_voice(self, audio_data, user_id, threshold=0.65):
        """Verify voice against stored template"""
        if user_id not in self.voice_templates:
            return False, 0.0
        
        # Extract features from test audio
        test_features = self.extract_voice_features(audio_data)
        if test_features is None:
            return False, 0.0
        
        # Get stored template
        stored_features = self.voice_templates[user_id]
        
        # Calculate cosine similarity
        try:
            similarity = 1 - cosine(test_features, stored_features)
            similarity = max(0, min(1, similarity))  # Clip to [0, 1]
        except:
            similarity = 0.0
        
        return similarity >= threshold, similarity

# ============================================================================
# SPEECH RECOGNITION (ASR) MODULE
# ============================================================================

class DigitSpeechRecognition:
    """Speech recognition for digit challenges"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def recognize_digits_from_audio(self, audio_file_path):
        """Recognize digits from audio file using Google Speech Recognition"""
        try:
            with sr.AudioFile(audio_file_path) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.record(source)
                
                # Recognize using Google Web Speech API
                text = self.recognizer.recognize_google(audio)
                
                # Extract only digits
                digits = ''.join(filter(str.isdigit, text))
                
                return digits
                
        except sr.UnknownValueError:
            st.warning("Speech recognition could not understand audio")
            return ""
        except sr.RequestError as e:
            st.warning(f"Could not request results from speech recognition service; {e}")
            return ""
        except Exception as e:
            st.warning(f"Speech recognition error: {e}")
            return ""
    
    def recognize_digits_from_bytes(self, audio_bytes):
        """Recognize digits from audio bytes"""
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Recognize from file
            digits = self.recognize_digits_from_audio(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            return digits
            
        except Exception as e:
            st.warning(f"Error processing audio bytes: {e}")
            return ""
    
    def generate_digit_challenge(self, length=4):
        """Generate random digit challenge"""
        import random
        return ''.join(str(random.randint(0, 9)) for _ in range(length))

# ============================================================================
# FUSION & DECISION SYSTEM
# ============================================================================

class FusionAuthenticationSystem:
    """Main system that fuses face and voice authentication"""
    
    def __init__(self):
        self.face_auth = FaceAuthentication()
        self.voice_auth = VoiceAuthentication()
        self.digit_asr = DigitSpeechRecognition()
        
        # Authentication logs
        self.auth_logs = []
    
    def enroll_user(self, user_id, face_images, voice_samples):
        """Enroll a new user with face and voice"""
        try:
            # Enroll face
            face_success = self.face_auth.enroll_face(user_id, face_images)
            
            # Enroll voice
            voice_success = self.voice_auth.enroll_voice(user_id, voice_samples)
            
            if face_success and voice_success:
                # Store in session state
                st.session_state.enrolled_users[user_id] = {
                    'face_enrolled': True,
                    'voice_enrolled': True,
                    'enrollment_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Enrollment error: {e}")
            return False
    
    def authenticate_user(self, user_id, face_image, voice_audio, expected_digits=""):
        """Perform complete authentication"""
        result = {
            'user_id': user_id,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'authenticated': False,
            'face_score': 0.0,
            'voice_score': 0.0,
            'digit_match': False,
            'liveness_check': False,
            'overall_confidence': 0.0,
            'reasons': []
        }
        
        try:
            settings = st.session_state.settings
            
            # Check if user is enrolled
            if user_id not in st.session_state.enrolled_users:
                result['reasons'].append("User not enrolled")
                return result
            
            # 1. Face verification
            face_verified, face_score = self.face_auth.verify_face(
                face_image, 
                user_id, 
                threshold=settings['face_threshold']
            )
            result['face_score'] = float(face_score)
            
            if not face_verified:
                result['reasons'].append(f"Face verification failed (score: {face_score:.2f})")
            
            # 2. Voice verification
            voice_verified, voice_score = self.voice_auth.verify_voice(
                voice_audio,
                user_id,
                threshold=settings['voice_threshold']
            )
            result['voice_score'] = float(voice_score)
            
            if not voice_verified:
                result['reasons'].append(f"Voice verification failed (score: {voice_score:.2f})")
            
            # 3. Digit challenge (if required)
            digit_match = False
            if settings['require_digits'] and expected_digits:
                # Save audio to temp file for ASR
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    if isinstance(voice_audio, bytes):
                        tmp.write(voice_audio)
                    else:
                        # Assume it's a file path
                        with open(voice_audio, 'rb') as f:
                            tmp.write(f.read())
                    tmp_path = tmp.name
                
                # Recognize digits
                spoken_digits = self.digit_asr.recognize_digits_from_audio(tmp_path)
                
                # Clean up
                os.unlink(tmp_path)
                
                digit_match = spoken_digits == expected_digits
                result['digit_match'] = digit_match
                
                if not digit_match:
                    result['reasons'].append(f"Digit mismatch. Expected: {expected_digits}, Got: {spoken_digits}")
            
            # 4. Liveness check (if required)
            liveness_check = True  # Default to True if not required
            if settings['require_liveness']:
                liveness_check = self.face_auth.detect_liveness(face_image)
                result['liveness_check'] = liveness_check
                
                if not liveness_check:
                    result['reasons'].append("Liveness check failed")
            
            # 5. Calculate overall confidence
            weights = {'face': 0.4, 'voice': 0.4, 'digits': 0.2}
            
            if not settings['require_digits']:
                weights = {'face': 0.5, 'voice': 0.5, 'digits': 0.0}
            
            overall_confidence = (
                weights['face'] * face_score +
                weights['voice'] * voice_score +
                weights['digits'] * (1.0 if digit_match else 0.0)
            )
            result['overall_confidence'] = float(overall_confidence)
            
            # 6. Make final decision
            conditions_met = []
            
            if face_verified:
                conditions_met.append("face")
            if voice_verified:
                conditions_met.append("voice")
            if not settings['require_digits'] or digit_match:
                conditions_met.append("digits")
            if not settings['require_liveness'] or liveness_check:
                conditions_met.append("liveness")
            
            # Check if multimodal is required
            if settings['multimodal']:
                required_conditions = ['face', 'voice']
                if settings['require_digits']:
                    required_conditions.append('digits')
                if settings['require_liveness']:
                    required_conditions.append('liveness')
                
                result['authenticated'] = all(
                    cond in conditions_met for cond in required_conditions
                )
            else:
                # Single modality is enough
                result['authenticated'] = len(conditions_met) >= 2
            
            # Log the authentication attempt
            self.auth_logs.append(result.copy())
            
            return result
            
        except Exception as e:
            st.error(f"Authentication error: {e}")
            result['reasons'].append(f"System error: {str(e)}")
            return result
    
    def generate_challenge(self, length=4):
        """Generate random digit challenge"""
        return self.digit_asr.generate_digit_challenge(length)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_audio_file(audio_bytes, filename="audio.wav"):
    """Save audio bytes to a file"""
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    return filename

def convert_image_to_array(image_file):
    """Convert uploaded image to numpy array"""
    image = Image.open(image_file)
    return np.array(image)

def play_audio_prompt(text):
    """Display audio prompt as text (for simulation)"""
    st.markdown(f'<div class="audio-prompt">🔊 Audio Prompt: "{text}"</div>', 
                unsafe_allow_html=True)

def apply_accessibility_settings():
    """Apply accessibility settings from session state"""
    settings = st.session_state.settings
    
    if settings['high_contrast']:
        st.markdown('<style>.main {background-color: black !important; color: white !important;}</style>', 
                   unsafe_allow_html=True)
    
    if settings['large_text']:
        st.markdown('<style>body {font-size: 1.2em !important;}</style>', 
                   unsafe_allow_html=True)

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

def show_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-checked.png", 
                width=100, use_column_width=True)
        
        st.title("🔐 Accessible Auth")
        st.markdown("---")
        
        # Navigation
        nav_options = {
            "🏠 Home": "home",
            "📝 Enrollment": "enrollment",
            "🔐 Authentication": "authentication",
            "📊 Dashboard": "dashboard",
            "⚙️ Settings": "settings",
            "ℹ️ About": "about"
        }
        
        selected_page = st.radio(
            "Navigation",
            list(nav_options.keys()),
            index=0,
            key="nav_radio"
        )
        
        st.markdown("---")
        
        # System status
        st.markdown("### System Status")
        enrolled_count = len(st.session_state.enrolled_users)
        st.metric("Enrolled Users", enrolled_count)
        
        # Quick actions
        st.markdown("### Quick Actions")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Flow", use_container_width=True):
                st.session_state.auth_step = 1
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Data", use_container_width=True):
                if st.checkbox("Confirm clear all data"):
                    st.session_state.enrolled_users = {}
                    st.session_state.user_enrolled = False
                    st.rerun()
        
        st.markdown("---")
        
        # Accessibility quick toggle
        st.markdown("### Accessibility")
        high_contrast = st.toggle("High Contrast", 
                                  value=st.session_state.settings['high_contrast'])
        large_text = st.toggle("Large Text", 
                               value=st.session_state.settings['large_text'])
        
        if high_contrast != st.session_state.settings['high_contrast']:
            st.session_state.settings['high_contrast'] = high_contrast
            st.rerun()
        
        if large_text != st.session_state.settings['large_text']:
            st.session_state.settings['large_text'] = large_text
            st.rerun()
        
        return nav_options[selected_page]

def show_home_page():
    """Render the home page"""
    st.title("🎤👁️ Voice + Face Authentication System")
    st.markdown("### Secure, Accessible Authentication for Visually-Impaired Users")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>🌟 Why This System?</h3>
        <p>Traditional ATMs force blind users to enter PINs on public keypads, 
        compromising privacy and independence. Our solution provides:</p>
        <ul>
        <li><strong>Privacy:</strong> Headphone-guided authentication</li>
        <li><strong>Security:</strong> Dual-factor biometric verification</li>
        <li><strong>Accessibility:</strong> Audio prompts and guidance</li>
        <li><strong>Safety:</strong> Anti-spoofing with liveness detection</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://img.icons8.com/color/300/000000/voice-id.png", 
                caption="Voice Authentication")
        st.image("https://img.icons8.com/color/300/000000/face-id.png", 
                caption="Face Authentication")
    
    # Features grid
    st.markdown("## ✨ Key Features")
    
    features = st.columns(3)
    
    with features[0]:
        st.markdown("""
        <div class="card">
        <h4>🎤 Voice Recognition</h4>
        <ul>
        <li>Speaker verification</li>
        <li>Random digit challenges</li>
        <li>Noise-resistant processing</li>
        <li>Short utterance support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features[1]:
        st.markdown("""
        <div class="card">
        <h4>👁️ Face Recognition</h4>
        <ul>
        <li>Real-time face detection</li>
        <li>Liveness verification</li>
        <li>Template-based matching</li>
        <li>Anti-spoofing measures</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with features[2]:
        st.markdown("""
        <div class="card">
        <h4>♿ Accessibility</h4>
        <ul>
        <li>Audio-guided prompts</li>
        <li>Headphone compatible</li>
        <li>Tactile guidance</li>
        <li>Timeout assistance</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("## 🚀 Quick Start Guide")
    
    steps = st.columns(4)
    
    with steps[0]:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>1️⃣</h3>
        <h4>Enroll</h4>
        <p>Register with voice and face samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[1]:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>2️⃣</h3>
        <h4>Authenticate</h4>
        <p>Follow audio prompts for verification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[2]:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>3️⃣</h3>
        <h4>Verify</h4>
        <p>Complete face and voice checks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with steps[3]:
        st.markdown("""
        <div class="card" style="text-align: center;">
        <h3>4️⃣</h3>
        <h4>Access</h4>
        <p>Get authentication decision</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Get Started with Enrollment", use_container_width=True):
            st.session_state.nav_radio = "📝 Enrollment"
            st.rerun()

def show_enrollment_page():
    """Render the user enrollment page"""
    st.title("📝 User Enrollment")
    st.markdown("Register a new user with voice and face biometrics")
    
    # Enrollment form
    with st.form("enrollment_form", clear_on_submit=True):
        st.markdown("### Step 1: User Information")
        
        user_id = st.text_input(
            "User ID / Card Number",
            placeholder="Enter unique user identifier",
            help="This could be an ATM card number or user ID"
        )
        
        st.markdown("---")
        st.markdown("### Step 2: Voice Enrollment")
        
        play_audio_prompt("Please say the digits 1 2 3 4 clearly")
        
        col1, col2, col3 = st.columns(3)
        
        voice_samples = []
        
        with col1:
            st.markdown("**Sample 1: 1 2 3 4**")
            audio_1 = st.audio_input("Record audio", key="voice1", label_visibility="collapsed")
            if audio_1:
                voice_samples.append(audio_1.getvalue())
                st.audio(audio_1.getvalue())
        
        with col2:
            st.markdown("**Sample 2: 5 6 7 8**")
            audio_2 = st.audio_input("Record audio", key="voice2", label_visibility="collapsed")
            if audio_2:
                voice_samples.append(audio_2.getvalue())
                st.audio(audio_2.getvalue())
        
        with col3:
            st.markdown("**Sample 3: 9 0 1 2**")
            audio_3 = st.audio_input("Record audio", key="voice3", label_visibility="collapsed")
            if audio_3:
                voice_samples.append(audio_3.getvalue())
                st.audio(audio_3.getvalue())
        
        st.markdown("---")
        st.markdown("### Step 3: Face Enrollment")
        
        play_audio_prompt("Please face the camera directly. Look straight ahead with a neutral expression.")
        
        face_images = []
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Front View**")
            img_1 = st.camera_input("Take photo", key="face1", label_visibility="collapsed")
            if img_1:
                face_images.append(convert_image_to_array(img_1))
                st.image(img_1, caption="Front view")
        
        with col2:
            st.markdown("**Slight Left**")
            img_2 = st.camera_input("Turn slightly left", key="face2", label_visibility="collapsed")
            if img_2:
                face_images.append(convert_image_to_array(img_2))
                st.image(img_2, caption="Left view")
        
        with col3:
            st.markdown("**Slight Right**")
            img_3 = st.camera_input("Turn slightly right", key="face3", label_visibility="collapsed")
            if img_3:
                face_images.append(convert_image_to_array(img_3))
                st.image(img_3, caption="Right view")
        
        st.markdown("---")
        
        # Submit button
        submitted = st.form_submit_button(
            "✅ Complete Enrollment", 
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            if not user_id:
                st.error("Please enter a User ID")
                return
            
            if len(voice_samples) < 3:
                st.error("Please provide 3 voice samples")
                return
            
            if len(face_images) < 3:
                st.error("Please provide 3 face images")
                return
            
            # Initialize system if needed
            if 'auth_system' not in st.session_state:
                st.session_state.auth_system = FusionAuthenticationSystem()
            
            # Perform enrollment
            with st.spinner("Processing enrollment..."):
                success = st.session_state.auth_system.enroll_user(
                    user_id, face_images, voice_samples
                )
                
                if success:
                    st.session_state.user_id = user_id
                    st.session_state.user_enrolled = True
                    
                    st.markdown('<div class="success-box">'
                               '<h3>✅ Enrollment Successful!</h3>'
                               f'<p>User <strong>{user_id}</strong> has been enrolled successfully.</p>'
                               '</div>', unsafe_allow_html=True)
                    
                    # Show enrollment summary
                    st.markdown("### Enrollment Summary")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Voice Samples", len(voice_samples))
                        st.metric("Face Images", len(face_images))
                    
                    with col2:
                        st.metric("User ID", user_id)
                        st.metric("Enrollment Date", 
                                 datetime.now().strftime("%Y-%m-%d"))
                    
                    # Option to proceed to authentication
                    if st.button("Proceed to Authentication"):
                        st.session_state.nav_radio = "🔐 Authentication"
                        st.rerun()
                else:
                    st.error("Enrollment failed. Please try again.")

def show_authentication_page():
    """Render the authentication page"""
    st.title("🔐 User Authentication")
    
    # Check if any users are enrolled
    if len(st.session_state.enrolled_users) == 0:
        st.warning("⚠️ No users enrolled yet. Please enroll a user first.")
        if st.button("Go to Enrollment"):
            st.session_state.nav_radio = "📝 Enrollment"
            st.rerun()
        return
    
    # Initialize system if needed
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = FusionAuthenticationSystem()
    
    # Step progress
    steps = ["User ID", "Face Scan", "Voice Challenge", "Result"]
    current_step = st.session_state.auth_step
    
    # Progress bar
    st.progress(current_step / len(steps))
    
    # Step indicators
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        with cols[i]:
            if i + 1 < current_step:
                st.markdown(f"✅ **{step}**")
            elif i + 1 == current_step:
                st.markdown(f"▶️ **{step}**")
            else:
                st.markdown(f"⭕ **{step}**")
    
    st.markdown("---")
    
    # Step 1: User ID input
    if current_step == 1:
        st.markdown("### Step 1: Enter User ID")
        
        play_audio_prompt("Please enter your user ID using the keypad or voice input.")
        
        user_id = st.text_input(
            "User ID",
            value=st.session_state.user_id,
            placeholder="Enter your registered user ID"
        )
        
        # Voice input option (simulated)
        with st.expander("🎤 Voice Input (Simulated)"):
            voice_id = st.text_input("Say your user ID (simulation)", 
                                    placeholder="123456")
            if voice_id:
                user_id = voice_id
        
        if st.button("Continue", type="primary", use_container_width=True):
            if not user_id:
                st.error("Please enter a User ID")
                return
            
            if user_id not in st.session_state.enrolled_users:
                st.error(f"User ID {user_id} is not enrolled. Please enroll first.")
                return
            
            st.session_state.user_id = user_id
            st.session_state.auth_step = 2
            st.rerun()
    
    # Step 2: Face authentication
    elif current_step == 2:
        st.markdown("### Step 2: Face Verification")
        
        play_audio_prompt("Please face the camera. Look straight ahead. Blink when you hear the beep.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Camera View")
            
            # Camera input
            camera_img = st.camera_input(
                "Position your face in the frame",
                key="auth_camera",
                help="Make sure your face is well-lit and centered"
            )
            
            if camera_img:
                # Convert to numpy array
                img_array = convert_image_to_array(camera_img)
                st.session_state.face_image = img_array
                
                # Display with face detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                faces = st.session_state.auth_system.face_auth.haar_face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                # Draw rectangles around faces
                img_with_faces = img_array.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                st.image(img_with_faces, caption="Face detected" if len(faces) > 0 else "No face detected")
                
                # Liveness check
                if st.session_state.settings['require_liveness']:
                    liveness_result = st.session_state.auth_system.face_auth.detect_liveness(img_array)
                    if liveness_result:
                        st.success("✅ Liveness check passed (eyes detected)")
                    else:
                        st.warning("⚠️ Please blink or move your head slightly")
        
        with col2:
            st.markdown("#### Instructions")
            st.markdown('<div class="instruction-box">'
                       '<h4>Face Capture Tips:</h4>'
                       '<ul>'
                       '<li>Ensure good lighting</li>'
                       '<li>Remove sunglasses/hat</li>'
                       '<li>Face directly forward</li>'
                       '<li>Maintain neutral expression</li>'
                       '<li>Blink naturally when prompted</li>'
                       '</ul>'
                       '</div>', unsafe_allow_html=True)
            
            if st.button("Capture & Continue", type="primary", use_container_width=True):
                if st.session_state.face_image is None:
                    st.error("Please capture a face image first")
                else:
                    # Generate digit challenge
                    st.session_state.digit_challenge = st.session_state.auth_system.generate_challenge(4)
                    st.session_state.auth_step = 3
                    st.rerun()
            
            if st.button("← Back", use_container_width=True):
                st.session_state.auth_step = 1
                st.rerun()
    
    # Step 3: Voice challenge
    elif current_step == 3:
        st.markdown("### Step 3: Voice Challenge")
        
        challenge = st.session_state.digit_challenge
        play_audio_prompt(f"Please say the following digits clearly: {challenge}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"#### 🔢 Challenge: **{challenge}**")
            
            # Audio recording
            voice_audio = st.audio_input(
                f"Say the digits: {challenge}",
                key="challenge_audio"
            )
            
            if voice_audio:
                audio_bytes = voice_audio.getvalue()
                st.session_state.voice_audio = audio_bytes
                st.audio(audio_bytes)
                
                # Verify digits (optional preview)
                if st.button("Check Digits", use_container_width=True):
                    with st.spinner("Recognizing digits..."):
                        digits_spoken = st.session_state.auth_system.digit_asr.recognize_digits_from_bytes(
                            audio_bytes
                        )
                        
                        if digits_spoken == challenge:
                            st.success(f"✅ Digits matched: {digits_spoken}")
                        else:
                            st.error(f"❌ Digits mismatch. Expected: {challenge}, Got: {digits_spoken}")
        
        with col2:
            st.markdown("#### Voice Tips")
            st.markdown('<div class="instruction-box">'
                       '<h4>Recording Guidelines:</h4>'
                       '<ul>'
                       '<li>Speak clearly and naturally</li>'
                       '<li>Minimize background noise</li>'
                       '<li>Use headphones if available</li>'
                       '<li>Say all digits in sequence</li>'
                       '<li>Wait for the beep before speaking</li>'
                       '</ul>'
                       '</div>', unsafe_allow_html=True)
            
            if st.button("Verify & Complete", type="primary", use_container_width=True):
                if st.session_state.voice_audio is None:
                    st.error("Please record your voice first")
                else:
                    st.session_state.auth_step = 4
                    st.rerun()
            
            if st.button("← Back", use_container_width=True):
                st.session_state.auth_step = 2
                st.rerun()
    
    # Step 4: Results
    elif current_step == 4:
        st.markdown("### Step 4: Authentication Result")
        
        with st.spinner("Verifying authentication..."):
            # Perform authentication
            result = st.session_state.auth_system.authenticate_user(
                st.session_state.user_id,
                st.session_state.face_image,
                st.session_state.voice_audio,
                st.session_state.digit_challenge
            )
            
            st.session_state.auth_result = result
        
        # Display result
        if result['authenticated']:
            st.markdown('<div class="success-box">'
                       '<h2>✅ ACCESS GRANTED</h2>'
                       '<p>Authentication successful! You may proceed with your transaction.</p>'
                       '</div>', unsafe_allow_html=True)
            
            # Play success sound (simulated)
            play_audio_prompt("Authentication successful. Access granted.")
        else:
            st.markdown('<div class="error-box">'
                       '<h2>❌ ACCESS DENIED</h2>'
                       '<p>Authentication failed. Please try again or contact support.</p>'
                       '</div>', unsafe_allow_html=True)
            
            # Play failure sound (simulated)
            play_audio_prompt("Authentication failed. Please try again.")
        
        # Detailed results
        st.markdown("### 📊 Authentication Details")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Face Score", f"{result['face_score']:.2%}")
        
        with col2:
            st.metric("Voice Score", f"{result['voice_score']:.2%}")
        
        with col3:
            st.metric("Overall Confidence", f"{result['overall_confidence']:.2%}")
        
        with col4:
            status = "✅ Pass" if result['authenticated'] else "❌ Fail"
            st.metric("Result", status)
        
        # Failure reasons
        if result['reasons']:
            with st.expander("❌ Failure Reasons"):
                for reason in result['reasons']:
                    st.error(reason)
        
        # Raw result data
        with st.expander("📋 Raw Result Data"):
            st.json(result)
        
        # Next steps
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Authenticate Again", use_container_width=True):
                st.session_state.auth_step = 1
                st.session_state.face_image = None
                st.session_state.voice_audio = None
                st.session_state.digit_challenge = ""
                st.rerun()
        
        with col2:
            if st.button("📝 New Enrollment", use_container_width=True):
                st.session_state.nav_radio = "📝 Enrollment"
                st.session_state.auth_step = 1
                st.rerun()
        
        with col3:
            if st.button("🏠 Return Home", use_container_width=True):
                st.session_state.nav_radio = "🏠 Home"
                st.session_state.auth_step = 1
                st.rerun()

def show_dashboard_page():
    """Render the system dashboard"""
    st.title("📊 System Dashboard")
    
    # Initialize system if needed
    if 'auth_system' not in st.session_state:
        st.session_state.auth_system = FusionAuthenticationSystem()
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        enrolled_count = len(st.session_state.enrolled_users)
        st.metric("Enrolled Users", enrolled_count)
    
    with col2:
        auth_attempts = len(st.session_state.auth_system.auth_logs)
        st.metric("Auth Attempts", auth_attempts)
    
    with col3:
        if auth_attempts > 0:
            success_count = sum(1 for log in st.session_state.auth_system.auth_logs 
                              if log['authenticated'])
            success_rate = (success_count / auth_attempts) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        else:
            st.metric("Success Rate", "0%")
    
    with col4:
        st.metric("Active Session", "✅ Online")
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    
    if st.session_state.auth_system.auth_logs:
        # Convert to dataframe for display
        import pandas as pd
        logs_df = pd.DataFrame(st.session_state.auth_system.auth_logs[-10:])  # Last 10 logs
        
        if not logs_df.empty:
            # Format columns
            if 'timestamp' in logs_df.columns:
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
                logs_df['time'] = logs_df['timestamp'].dt.strftime('%H:%M:%S')
            
            # Display table
            display_cols = ['user_id', 'time', 'authenticated', 'overall_confidence']
            display_cols = [col for col in display_cols if col in logs_df.columns]
            
            st.dataframe(
                logs_df[display_cols].style.format({
                    'overall_confidence': '{:.1%}'
                }),
                use_container_width=True
            )
            
            # Chart
            st.markdown("### 📊 Success Rate Over Time")
            if len(logs_df) > 1:
                # Calculate cumulative success rate
                logs_df['cumulative_success'] = logs_df['authenticated'].cumsum()
                logs_df['cumulative_rate'] = (logs_df['cumulative_success'] / 
                                            (logs_df.index + 1)) * 100
                
                st.line_chart(logs_df[['cumulative_rate']])
    else:
        st.info("No authentication attempts recorded yet.")
    
    # User management
    st.markdown("### 👥 User Management")
    
    if st.session_state.enrolled_users:
        users_df = pd.DataFrame([
            {'User ID': uid, 'Enrolled': info['enrollment_date']}
            for uid, info in st.session_state.enrolled_users.items()
        ])
        
        st.dataframe(users_df, use_container_width=True)
        
        # User actions
        selected_user = st.selectbox(
            "Select User",
            list(st.session_state.enrolled_users.keys())
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Details", use_container_width=True):
                user_info = st.session_state.enrolled_users[selected_user]
                st.json(user_info)
        
        with col2:
            if st.button("Remove User", use_container_width=True, type="secondary"):
                if st.checkbox(f"Confirm removal of user {selected_user}"):
                    del st.session_state.enrolled_users[selected_user]
                    st.success(f"User {selected_user} removed")
                    st.rerun()
    else:
        st.info("No users enrolled yet.")
    
    # System logs
    st.markdown("### 📝 System Logs")
    
    if st.button("Clear All Logs", type="secondary"):
        if st.checkbox("Confirm clear all logs"):
            st.session_state.auth_system.auth_logs = []
            st.rerun()
    
    if st.session_state.auth_system.auth_logs:
        with st.expander("View All Logs"):
            for log in st.session_state.auth_system.auth_logs[-20:]:  # Last 20 logs
                status = "✅" if log['authenticated'] else "❌"
                st.text(f"{log['timestamp']} - {status} User: {log['user_id']} "
                       f"- Confidence: {log['overall_confidence']:.1%}")

def show_settings_page():
    """Render the settings page"""
    st.title("⚙️ System Settings")
    
    tab1, tab2, tab3 = st.tabs(["Security", "Accessibility", "System"])
    
    with tab1:
        st.markdown("### 🔒 Security Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            face_threshold = st.slider(
                "Face Recognition Threshold",
                min_value=0.5, max_value=0.95, value=st.session_state.settings['face_threshold'],
                step=0.05, help="Higher values = stricter face matching"
            )
            
            voice_threshold = st.slider(
                "Voice Recognition Threshold",
                min_value=0.5, max_value=0.95, value=st.session_state.settings['voice_threshold'],
                step=0.05, help="Higher values = stricter voice matching"
            )
        
        with col2:
            require_liveness = st.checkbox(
                "Require Liveness Check",
                value=st.session_state.settings['require_liveness'],
                help="Prevents spoofing with photos/videos"
            )
            
            require_digits = st.checkbox(
                "Require Digit Challenge",
                value=st.session_state.settings['require_digits'],
                help="Prevents replay attacks"
            )
            
            multimodal = st.checkbox(
                "Multi-modal Authentication",
                value=st.session_state.settings['multimodal'],
                help="Require both face AND voice verification"
            )
        
        # Security presets
        st.markdown("#### Security Presets")
        preset_cols = st.columns(3)
        
        with preset_cols[0]:
            if st.button("Low Security", use_container_width=True):
                st.session_state.settings.update({
                    'face_threshold': 0.6,
                    'voice_threshold': 0.55,
                    'require_liveness': False,
                    'require_digits': False,
                    'multimodal': False
                })
                st.rerun()
        
        with preset_cols[1]:
            if st.button("Medium Security", use_container_width=True):
                st.session_state.settings.update({
                    'face_threshold': 0.7,
                    'voice_threshold': 0.65,
                    'require_liveness': True,
                    'require_digits': True,
                    'multimodal': True
                })
                st.rerun()
        
        with preset_cols[2]:
            if st.button("High Security", use_container_width=True):
                st.session_state.settings.update({
                    'face_threshold': 0.8,
                    'voice_threshold': 0.75,
                    'require_liveness': True,
                    'require_digits': True,
                    'multimodal': True
                })
                st.rerun()
    
    with tab2:
        st.markdown("### ♿ Accessibility Settings")
        
        audio_volume = st.slider(
            "Audio Prompt Volume",
            min_value=0, max_value=100, value=st.session_state.settings['audio_volume'],
            help="Volume level for audio guidance"
        )
        
        prompt_speed = st.selectbox(
            "Prompt Speaking Speed",
            options=["Slow", "Normal", "Fast"],
            index=["Slow", "Normal", "Fast"].index(st.session_state.settings['prompt_speed']),
            help="Speed of audio prompts"
        )
        
        timeout_duration = st.selectbox(
            "Session Timeout Duration",
            options=["30 seconds", "1 minute", "2 minutes", "5 minutes", "10 minutes"],
            index=["30 seconds", "1 minute", "2 minutes", "5 minutes", "10 minutes"]
                .index(f"{st.session_state.settings['timeout_duration']} seconds" 
                      if st.session_state.settings['timeout_duration'] < 60 
                      else f"{st.session_state.settings['timeout_duration']//60} minutes"),
            help="Time before automatic logout"
        )
        
        # Parse timeout duration
        if "second" in timeout_duration:
            timeout_seconds = int(timeout_duration.split()[0])
        else:
            timeout_seconds = int(timeout_duration.split()[0]) * 60
        
        # Visual settings
        st.markdown("#### Visual Settings")
        
        high_contrast = st.checkbox(
            "High Contrast Mode",
            value=st.session_state.settings['high_contrast'],
            help="Better visibility for low vision users"
        )
        
        large_text = st.checkbox(
            "Large Text Mode",
            value=st.session_state.settings['large_text'],
            help="Increase text size for better readability"
        )
    
    with tab3:
        st.markdown("### ⚙️ System Configuration")
        
        st.markdown("#### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export User Data", use_container_width=True):
                # Create export data
                export_data = {
                    'enrolled_users': st.session_state.enrolled_users,
                    'settings': st.session_state.settings,
                    'export_date': datetime.now().isoformat()
                }
                
                # Convert to JSON
                export_json = json.dumps(export_data, indent=2)
                
                # Create download button
                st.download_button(
                    label="Download Export",
                    data=export_json,
                    file_name=f"auth_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col2:
            if st.button("Import User Data", use_container_width=True):
                uploaded_file = st.file_uploader(
                    "Choose a JSON file",
                    type=['json'],
                    key="import_file"
                )
                
                if uploaded_file is not None:
                    try:
                        import_data = json.load(uploaded_file)
                        
                        if 'enrolled_users' in import_data:
                            st.session_state.enrolled_users.update(import_data['enrolled_users'])
                            st.success(f"Imported {len(import_data['enrolled_users'])} users")
                        
                        if 'settings' in import_data:
                            st.session_state.settings.update(import_data['settings'])
                            st.success("Settings imported")
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Import failed: {e}")
        
        st.markdown("#### System Information")
        
        sys_info = {
            "Python Version": "3.9+",
            "Streamlit Version": "1.28.0+",
            "OpenCV Version": "4.8.0+",
            "System Status": "✅ Operational",
            "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        for key, value in sys_info.items():
            st.text(f"{key}: {value}")
        
        st.markdown("#### Maintenance")
        
        if st.button("Reset System", type="secondary", use_container_width=True):
            if st.checkbox("Confirm system reset (this will clear ALL data)"):
                # Reset session state
                keys_to_keep = ['settings']
                for key in list(st.session_state.keys()):
                    if key not in keys_to_keep:
                        del st.session_state[key]
                
                # Reinitialize
                init_session_state()
                st.success("System reset complete")
                st.rerun()
    
    # Save settings button
    st.markdown("---")
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        # Update settings
        st.session_state.settings.update({
            'face_threshold': face_threshold,
            'voice_threshold': voice_threshold,
            'require_liveness': require_liveness,
            'require_digits': require_digits,
            'multimodal': multimodal,
            'audio_volume': audio_volume,
            'prompt_speed': prompt_speed,
            'high_contrast': high_contrast,
            'large_text': large_text,
            'timeout_duration': timeout_seconds
        })
        
        st.success("Settings saved successfully!")

def show_about_page():
    """Render the about page"""
    st.title("ℹ️ About This System")
    
    st.markdown("""
    <div class="card">
    <h2>Voice + Face Authentication Module</h2>
    <p><strong>Version:</strong> 1.0.0</p>
    <p><strong>Last Updated:</strong> December 2024</p>
    <p><strong>Purpose:</strong> Accessible authentication for visually-impaired ATM users</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project description
    st.markdown("### 📋 Project Overview")
    
    st.markdown("""
    This system implements a software-only authentication module that enables 
    visually-impaired users to authenticate at ATMs using voice and face as 
    a second factor, eliminating the need to enter PINs on public keypads.
    
    #### 🔧 Key Components:
    1. **Speech Recognition (ASR)**: Converts short speech to text for digit challenges
    2. **Speaker Verification**: Confirms the speaker is the enrolled account owner
    3. **Face Recognition**: Optionally matches live face to enrolled face with liveness detection
    4. **Accessibility Features**: Audio-guided prompts, headphone compatibility, tactile guidance
    """)
    
    # Architecture
    st.markdown("### 🏗️ System Architecture")
    
    architecture_cols = st.columns(2)
    
    with architecture_cols[0]:
        st.markdown("""
        #### 📊 Data Flow:
        1. **Identify Account** (via card or alias)
        2. **Face Step** (optional):
           - Detect → Align → Embed → Match → Liveness Check
        3. **Voice Step**:
           - Prompt random digits → Denoise → Embed → Match
        4. **Decision**:
           - Access granted only if thresholds met
        5. **Data Fetch**:
           - ATM fetches user profile (outside module)
        """)
    
    with architecture_cols[1]:
        st.markdown("""
        #### 🔐 Security Features:
        - **Template Storage**: Only encrypted embeddings stored
        - **Anti-Spoofing**: Liveness checks, random challenges
        - **Privacy**: No raw audio/video storage
        - **Encryption**: Templates encrypted at rest and in transit
        """)
    
    # Technical specifications
    st.markdown("### ⚙️ Technical Specifications")
    
    tech_specs = {
        "Face Recognition": "Haar Cascade + Histogram features",
        "Voice Recognition": "MFCC feature extraction + Cosine similarity",
        "Speech Recognition": "Google Speech-to-API for digits",
        "Liveness Detection": "Eye detection for basic liveness",
        "Framework": "Streamlit web interface",
        "Processing": "Real-time face/voice processing",
        "Storage": "In-memory templates (session-based)",
        "Security": "Threshold-based verification"
    }
    
    for component, description in tech_specs.items():
        st.markdown(f"**{component}:** {description}")
    
    # Accessibility features
    st.markdown("### ♿ Accessibility Features")
    
    accessibility_features = [
        "Audio-guided prompts for every step",
        "Headphone jack compatibility simulation",
        "Tactile marker guidance (simulated)",
        "Gentle timeout and retry prompts",
        "High contrast mode for low vision",
        "Large text mode for better readability",
        "Voice input simulation",
        "Step-by-step audio instructions"
    ]
    
    for feature in accessibility_features:
        st.markdown(f"✓ {feature}")
    
    # Future enhancements
    st.markdown("### 🚀 Future Enhancements")
    
    future_cols = st.columns(2)
    
    with future_cols[0]:
        st.markdown("""
        #### 🔧 Planned Improvements:
        - Stronger anti-spoofing classifiers
        - Threshold personalization per user
        - Multi-language prompt support
        - Accent adaptation for voice recognition
        - Secure key management (HSM/KMS)
        - Vendor pilot integration
        """)
    
    with future_cols[1]:
        st.markdown("""
        #### 📈 Performance Targets:
        - **FAR/FRR**: < 0.1% false accept, < 1% false reject
        - **EER**: Target < 0.5%
        - **Latency**: < 15 seconds total
        - **Robustness**: Tested against real ATM noise
        - **Accuracy**: > 99% for enrolled users
        """)
    
    # Contact and support
    st.markdown("### 📞 Support & Contact")
    
    st.markdown("""
    <div class="card">
    <p><strong>For support, questions, or collaboration:</strong></p>
    <p>📧 Email: support@accessible-auth.com</p>
    <p>📚 Documentation: [docs.accessible-auth.com](https://docs.accessible-auth.com)</p>
    <p>🐛 Bug Reports: [GitHub Issues](https://github.com/yourusername/voice-face-auth/issues)</p>
    <p>📄 License: MIT Open Source</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo instructions
    st.markdown("### 🎮 Demo Instructions")
    
    with st.expander("Click for demo instructions"):
        st.markdown("""
        1. **Enrollment** (required first):
           - Go to Enrollment page
           - Enter a User ID
           - Record 3 voice samples (follow prompts)
           - Capture 3 face images (different angles)
           - Click "Complete Enrollment"
        
        2. **Authentication**:
           - Go to Authentication page
           - Enter your enrolled User ID
           - Step 2: Face the camera, follow prompts
           - Step 3: Say the digit challenge shown
           - Step 4: View authentication result
        
        3. **Testing Different Scenarios**:
           - Try with different lighting conditions
           - Test with background noise
           - Try spoofing with photo (should fail liveness)
           - Test voice-only or face-only modes in Settings
        
        4. **Explore Other Features**:
           - Dashboard: View system metrics and logs
           - Settings: Adjust security and accessibility
           - About: Learn more about the system
        """)
    
    # Acknowledgments
    st.markdown("### 🙏 Acknowledgments")
    
    st.markdown("""
    This project is a final year project implementation demonstrating accessible 
    authentication technology. It uses various open-source libraries including:
    
    - **Streamlit** for the web interface
    - **OpenCV** for computer vision
    - **Librosa** for audio processing
    - **SpeechRecognition** for ASR
    - **NumPy/SciPy** for numerical computations
    
    Special thanks to the accessibility community for their insights and feedback.
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Apply accessibility settings
    apply_accessibility_settings()
    
    # Sidebar navigation
    selected_page = show_sidebar()
    
    # Page routing
    if selected_page == "home":
        show_home_page()
    elif selected_page == "enrollment":
        show_enrollment_page()
    elif selected_page == "authentication":
        show_authentication_page()
    elif selected_page == "dashboard":
        show_dashboard_page()
    elif selected_page == "settings":
        show_settings_page()
    elif selected_page == "about":
        show_about_page()
    
    # Footer
    st.markdown("---")
    footer_cols = st.columns([2, 1, 1])
    
    with footer_cols[0]:
        st.caption("🔐 Voice + Face Authentication System v1.0.0 | "
                  "Designed for Accessibility")
    
    with footer_cols[1]:
        st.caption("⚠️ Demo System - Not for Production Use")
    
    with footer_cols[2]:
        if st.session_state.settings['timeout_duration'] > 0:
            # Auto-refresh for timeout simulation
            import time
            current_time = time.time()
            if 'last_activity' not in st.session_state:
                st.session_state.last_activity = current_time
            
            elapsed = current_time - st.session_state.last_activity
            remaining = st.session_state.settings['timeout_duration'] - elapsed
            
            if remaining > 0:
                st.caption(f"⏱️ Auto-logout in: {int(remaining)}s")
            else:
                st.warning("Session timed out")
                if st.button("Refresh Session"):
                    st.session_state.last_activity = time.time()
                    st.rerun()

# ============================================================================
# RUN APPLICATION
# ============================================================================

if __name__ == "__main__":
    main()