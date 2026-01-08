"""
Voice + Face Authentication System with Database
Complete Implementation with PostgreSQL/SQLite Support
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
import tempfile
import os
import json
import hashlib
import base64
import speech_recognition as sr
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from scipy import signal
from psycopg2.extras import Json
import warnings
warnings.filterwarnings('ignore')
import pickle
import time
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from scipy import signal
import sqlite3
import psycopg2
from psycopg2.extras import Json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Manages database operations for the authentication system"""
    
    def __init__(self, db_type='sqlite', db_name='auth_system.db'):
        self.db_type = db_type
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.init_database()
    
    def connect(self):
        """Connect to database"""
        try:
            if self.db_type == 'sqlite':
                self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
            elif self.db_type == 'postgresql':
                # For PostgreSQL, you would use actual connection parameters
                self.conn = psycopg2.connect(
                    host="localhost",
                    database="auth_system",
                    user="postgres",
                    password="password"
                )
            self.cursor = self.conn.cursor()
            return True
        except Exception as e:
            st.error(f"Database connection error: {e}")
            return False
    
    def init_database(self):
        """Initialize database tables"""
        if not self.connect():
            return False
        
        try:
            # Users table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(50) PRIMARY KEY,
                    full_name VARCHAR(100),
                    email VARCHAR(100),
                    phone VARCHAR(20),
                    enrollment_date TIMESTAMP,
                    last_login TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'active',
                    settings JSON
                )
            ''')
            
            # Face templates table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(50),
                    template_data BLOB,
                    template_hash VARCHAR(64),
                    algorithm VARCHAR(50),
                    version INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Voice templates table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS voice_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(50),
                    template_data BLOB,
                    template_hash VARCHAR(64),
                    algorithm VARCHAR(50),
                    version INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            ''')
            
            # Authentication logs table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    auth_type VARCHAR(20),
                    face_score REAL,
                    voice_score REAL,
                    digit_match BOOLEAN,
                    liveness_check BOOLEAN,
                    overall_confidence REAL,
                    result VARCHAR(20),
                    ip_address VARCHAR(45),
                    user_agent TEXT,
                    details JSON
                )
            ''')
            
            # System settings table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    setting_key VARCHAR(50) PRIMARY KEY,
                    setting_value TEXT,
                    description TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert default settings if not exists
            default_settings = [
                ('face_threshold', '0.7', 'Face recognition threshold'),
                ('voice_threshold', '0.65', 'Voice recognition threshold'),
                ('require_liveness', 'true', 'Require liveness check'),
                ('require_digits', 'true', 'Require digit challenge'),
                ('multimodal', 'true', 'Require both face and voice'),
                ('max_auth_attempts', '3', 'Maximum authentication attempts'),
                ('lockout_duration', '300', 'Lockout duration in seconds'),
                ('session_timeout', '600', 'Session timeout in seconds')
            ]
            
            for key, value, desc in default_settings:
                self.cursor.execute('''
                    INSERT OR IGNORE INTO system_settings (setting_key, setting_value, description)
                    VALUES (?, ?, ?)
                ''', (key, value, desc))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            st.error(f"Database initialization error: {e}")
            return False
        finally:
            self.close()
    
    def save_user(self, user_id, full_name="", email="", phone=""):
        """Save or update user information"""
        try:
            self.connect()
            self.cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, full_name, email, phone, enrollment_date, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, full_name, email, phone, datetime.now(), datetime.now()))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving user: {e}")
            return False
        finally:
            self.close()
    
    def save_face_template(self, user_id, template_data, algorithm="histogram_lbp", version=1):
        """Save face template to database"""
        try:
            self.connect()
            # Serialize template
            template_bytes = pickle.dumps(template_data)
            template_hash = hashlib.sha256(template_bytes).hexdigest()
            
            self.cursor.execute('''
                INSERT INTO face_templates 
                (user_id, template_data, template_hash, algorithm, version)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, template_bytes, template_hash, algorithm, version))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving face template: {e}")
            return False
        finally:
            self.close()
    
    def save_voice_template(self, user_id, template_data, algorithm="mfcc", version=1):
        """Save voice template to database"""
        try:
            self.connect()
            # Serialize template
            template_bytes = pickle.dumps(template_data)
            template_hash = hashlib.sha256(template_bytes).hexdigest()
            
            self.cursor.execute('''
                INSERT INTO voice_templates 
                (user_id, template_data, template_hash, algorithm, version)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, template_bytes, template_hash, algorithm, version))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error saving voice template: {e}")
            return False
        finally:
            self.close()
    
    def get_user_face_templates(self, user_id):
        """Retrieve face templates for a user"""
        try:
            self.connect()
            self.cursor.execute('''
                SELECT template_data FROM face_templates 
                WHERE user_id = ? ORDER BY created_at DESC LIMIT 3
            ''', (user_id,))
            rows = self.cursor.fetchall()
            
            templates = []
            for row in rows:
                template_data = pickle.loads(row[0])
                templates.append(template_data)
            
            return templates
        except Exception as e:
            st.error(f"Error retrieving face templates: {e}")
            return []
        finally:
            self.close()
    
    def get_user_voice_templates(self, user_id):
        """Retrieve voice templates for a user"""
        try:
            self.connect()
            self.cursor.execute('''
                SELECT template_data FROM voice_templates 
                WHERE user_id = ? ORDER BY created_at DESC LIMIT 3
            ''', (user_id,))
            rows = self.cursor.fetchall()
            
            templates = []
            for row in rows:
                template_data = pickle.loads(row[0])
                templates.append(template_data)
            
            return templates
        except Exception as e:
            st.error(f"Error retrieving voice templates: {e}")
            return []
        finally:
            self.close()
    
    def log_authentication(self, auth_data):
        """Log authentication attempt"""
        try:
            self.connect()
            self.cursor.execute('''
                INSERT INTO auth_logs 
                (user_id, auth_type, face_score, voice_score, digit_match, 
                 liveness_check, overall_confidence, result, ip_address, user_agent, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                auth_data.get('user_id'),
                auth_data.get('auth_type', 'biometric'),
                auth_data.get('face_score', 0),
                auth_data.get('voice_score', 0),
                auth_data.get('digit_match', False),
                auth_data.get('liveness_check', False),
                auth_data.get('overall_confidence', 0),
                auth_data.get('result', 'failed'),
                auth_data.get('ip_address', '127.0.0.1'),
                auth_data.get('user_agent', 'streamlit'),
                json.dumps(auth_data.get('details', {}))
            ))
            
            # Update user's last login if successful
            if auth_data.get('result') == 'success':
                self.cursor.execute('''
                    UPDATE users SET last_login = ? WHERE user_id = ?
                ''', (datetime.now(), auth_data.get('user_id')))
            
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error logging authentication: {e}")
            return False
        finally:
            self.close()
    
    def get_user(self, user_id):
        """Get user information"""
        try:
            self.connect()
            self.cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            row = self.cursor.fetchone()
            
            if row:
                user = dict(row)
                # Convert datetime objects to string
                for key in user:
                    if isinstance(user[key], datetime):
                        user[key] = user[key].isoformat()
                return user
            return None
        except Exception as e:
            st.error(f"Error getting user: {e}")
            return None
        finally:
            self.close()
    
    def get_all_users(self):
        """Get all users"""
        try:
            self.connect()
            self.cursor.execute('SELECT * FROM users ORDER BY enrollment_date DESC')
            rows = self.cursor.fetchall()
            
            users = []
            for row in rows:
                user = dict(row)
                # Convert datetime objects to string
                for key in user:
                    if isinstance(user[key], datetime):
                        user[key] = user[key].isoformat()
                users.append(user)
            
            return users
        except Exception as e:
            st.error(f"Error getting users: {e}")
            return []
        finally:
            self.close()
    
    def get_auth_logs(self, user_id=None, limit=100):
        """Get authentication logs"""
        try:
            self.connect()
            if user_id:
                self.cursor.execute('''
                    SELECT * FROM auth_logs 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
            else:
                self.cursor.execute('''
                    SELECT * FROM auth_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = self.cursor.fetchall()
            
            logs = []
            for row in rows:
                log = dict(row)
                # Convert datetime objects to string
                for key in log:
                    if isinstance(log[key], datetime):
                        log[key] = log[key].isoformat()
                logs.append(log)
            
            return logs
        except Exception as e:
            st.error(f"Error getting auth logs: {e}")
            return []
        finally:
            self.close()
    
    def get_system_settings(self):
        """Get system settings"""
        try:
            self.connect()
            self.cursor.execute('SELECT setting_key, setting_value FROM system_settings')
            rows = self.cursor.fetchall()
            
            settings = {}
            for row in rows:
                key, value = row
                # Convert string values to appropriate types
                if value.lower() in ('true', 'false'):
                    settings[key] = value.lower() == 'true'
                elif value.isdigit():
                    settings[key] = int(value)
                elif self.is_float(value):
                    settings[key] = float(value)
                else:
                    settings[key] = value
            
            return settings
        except Exception as e:
            st.error(f"Error getting system settings: {e}")
            return {}
        finally:
            self.close()
    
    def update_system_settings(self, settings):
        """Update system settings"""
        try:
            self.connect()
            for key, value in settings.items():
                self.cursor.execute('''
                    UPDATE system_settings 
                    SET setting_value = ?, updated_at = ? 
                    WHERE setting_key = ?
                ''', (str(value), datetime.now(), key))
            
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error updating settings: {e}")
            return False
        finally:
            self.close()
    
    def delete_user(self, user_id):
        """Delete user and all associated data"""
        try:
            self.connect()
            self.cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            self.conn.commit()
            return True
        except Exception as e:
            st.error(f"Error deleting user: {e}")
            return False
        finally:
            self.close()
    
    def get_auth_stats(self):
        """Get authentication statistics"""
        try:
            self.connect()
            
            # Total authentications
            self.cursor.execute('SELECT COUNT(*) FROM auth_logs')
            total_auth = self.cursor.fetchone()[0]
            
            # Successful authentications
            self.cursor.execute('SELECT COUNT(*) FROM auth_logs WHERE result = "success"')
            success_auth = self.cursor.fetchone()[0]
            
            # Failed authentications
            self.cursor.execute('SELECT COUNT(*) FROM auth_logs WHERE result = "failed"')
            failed_auth = self.cursor.fetchone()[0]
            
            # Today's authentications
            today = datetime.now().date()
            self.cursor.execute('SELECT COUNT(*) FROM auth_logs WHERE DATE(timestamp) = ?', (today,))
            today_auth = self.cursor.fetchone()[0]
            
            # Active users (logged in last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            self.cursor.execute('SELECT COUNT(DISTINCT user_id) FROM auth_logs WHERE timestamp > ?', (thirty_days_ago,))
            active_users = self.cursor.fetchone()[0]
            
            return {
                'total_auth': total_auth,
                'success_auth': success_auth,
                'failed_auth': failed_auth,
                'success_rate': (success_auth / total_auth * 100) if total_auth > 0 else 0,
                'today_auth': today_auth,
                'active_users': active_users
            }
        except Exception as e:
            st.error(f"Error getting auth stats: {e}")
            return {}
        finally:
            self.close()
    
    def is_float(self, value):
        """Check if a string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

# ============================================================================
# FACE RECOGNITION ENGINE
# ============================================================================

class FaceRecognitionEngine:
    """Advanced face recognition with deep learning features"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.lbp_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        )
        
        # Load deep learning model for face recognition (simplified)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_labels = {}
        self.label_counter = 0
    
    def preprocess_face(self, image):
        """Preprocess face image for feature extraction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization
            gray_eq = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur for noise reduction
            gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray_blur,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return None, None
            
            # Get the largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Extract face region with padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = gray[y1:y2, x1:x2]
            
            # Resize to standard size
            face_resized = cv2.resize(face_region, (150, 150))
            
            return face_resized, (x, y, w, h)
            
        except Exception as e:
            st.error(f"Face preprocessing error: {e}")
            return None, None
    
    def extract_features(self, image):
        """Extract multiple types of features from face"""
        try:
            face_region, _ = self.preprocess_face(image)
            if face_region is None:
                return None
            
            features = {}
            
            # 1. LBP Histogram features
            lbp = self.extract_lbp_features(face_region)
            features['lbp'] = lbp
            
            # 2. Color histogram (if color image available)
            color_hist = self.extract_color_histogram(image)
            features['color_hist'] = color_hist
            
            # 3. HOG features (simplified)
            hog_features = self.extract_hog_features(face_region)
            features['hog'] = hog_features
            
            # 4. Deep features (simulated with PCA-like reduction)
            deep_features = self.extract_deep_features(face_region)
            features['deep'] = deep_features
            
            # Combine all features
            combined_features = np.concatenate([
                features['lbp'],
                features['color_hist'],
                features['hog'],
                features['deep']
            ])
            
            # Normalize features
            combined_features = (combined_features - np.mean(combined_features)) / np.std(combined_features)
            
            return combined_features
            
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return None
    
    def extract_lbp_features(self, face_region):
        """Extract Local Binary Pattern features"""
        # Simplified LBP implementation
        radius = 1
        n_points = 8 * radius
        
        # Create LBP image
        lbp_image = np.zeros_like(face_region, dtype=np.uint8)
        height, width = face_region.shape
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = face_region[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    x0, y0 = int(np.floor(x)), int(np.floor(y))
                    x1, y1 = int(np.ceil(x)), int(np.ceil(y))
                    
                    # Bilinear interpolation
                    fx = x - x0
                    fy = y - y0
                    value = (1 - fx) * (1 - fy) * face_region[y0, x0] + \
                            fx * (1 - fy) * face_region[y0, x1] + \
                            (1 - fx) * fy * face_region[y1, x0] + \
                            fx * fy * face_region[y1, x1]
                    
                    if value >= center:
                        code |= 1 << k
                
                lbp_image[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def extract_color_histogram(self, image):
        """Extract color histogram from face region"""
        try:
            # Convert to HSV color space
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Compute histogram for each channel
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Combine histograms
            combined = np.concatenate([h_hist, s_hist, v_hist])
            
            return combined
            
        except:
            # Return zeros if color histogram extraction fails
            return np.zeros(692)  # 180 + 256 + 256
    
    def extract_hog_features(self, image):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        try:
            # Compute gradients
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
            
            # Compute magnitude and angle
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            
            # Quantize angles into 9 bins (0-180 degrees)
            bin_size = 20  # 180/9 = 20 degrees per bin
            angle_bins = np.floor(angle / bin_size).astype(int)
            angle_bins = angle_bins % 9
            
            # Create histogram
            hist = np.zeros(9)
            for i in range(9):
                mask = angle_bins == i
                hist[i] = np.sum(mag[mask])
            
            # Normalize
            hist = hist / (np.sum(hist) + 1e-7)
            
            return hist
            
        except:
            return np.zeros(9)
    
    def extract_deep_features(self, image):
        """Simulate deep learning features using PCA-like approach"""
        # Flatten image
        flattened = image.flatten()
        
        # Apply PCA (simplified - using random projection)
        np.random.seed(42)
        projection_matrix = np.random.randn(flattened.shape[0], 128)
        projected = np.dot(flattened, projection_matrix)
        
        # Normalize
        projected = projected / (np.linalg.norm(projected) + 1e-7)
        
        return projected
    
    def verify_face(self, image, stored_templates, threshold=0.7):
        """Verify face against stored templates"""
        try:
            # Extract features from input image
            test_features = self.extract_features(image)
            if test_features is None:
                return False, 0.0
            
            # Calculate similarity with each stored template
            best_similarity = 0.0
            for template in stored_templates:
                # Calculate cosine similarity
                similarity = 1 - cosine(test_features, template)
                best_similarity = max(best_similarity, similarity)
            
            # Return verification result
            return best_similarity >= threshold, best_similarity
            
        except Exception as e:
            st.error(f"Face verification error: {e}")
            return False, 0.0
    
    def detect_liveness(self, image, consecutive_frames=2):
        """Advanced liveness detection"""
        try:
            # Multiple liveness checks
            
            # 1. Eye blink detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return False
            
            # Get largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            eye_count = len(eyes)
            
            # 2. Texture analysis (simplified)
            # Calculate image sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # 3. Color-based liveness check
            # Real faces have specific color distribution
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            skin_mask = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
            skin_ratio = np.sum(skin_mask > 0) / (skin_mask.size + 1e-7)
            
            # Decision logic
            liveness_score = 0
            
            # Eye detection
            if eye_count >= 1:
                liveness_score += 0.3
            
            # Sharpness check (printouts are usually blurrier)
            if sharpness > 100:  # Threshold for sharpness
                liveness_score += 0.3
            
            # Skin color check
            if 0.1 < skin_ratio < 0.8:  # Reasonable skin ratio
                liveness_score += 0.4
            
            return liveness_score >= 0.6
            
        except Exception as e:
            st.error(f"Liveness detection error: {e}")
            return False

# ============================================================================
# VOICE RECOGNITION ENGINE
# ============================================================================

class VoiceRecognitionEngine:
    """Advanced voice recognition with multiple feature types"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        
    def preprocess_audio(self, audio_data):
        """Preprocess audio for feature extraction"""
        try:
            # Convert to numpy array if needed
            if isinstance(audio_data, bytes):
                # Save to temp file and load with librosa
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp.write(audio_data)
                    tmp_path = tmp.name
                
                y, sr = librosa.load(tmp_path, sr=self.sample_rate)
                os.unlink(tmp_path)
            else:
                y, sr = audio_data, self.sample_rate
            
            # Resample if needed
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            
            if len(y_trimmed) < self.sample_rate * 0.5:  # Less than 0.5 seconds
                y_trimmed = y  # Use original if trimmed too much
            
            # Normalize volume
            y_normalized = librosa.util.normalize(y_trimmed)
            
            return y_normalized
            
        except Exception as e:
            st.error(f"Audio preprocessing error: {e}")
            return None
    
    def extract_features(self, audio_data):
        """Extract multiple voice features"""
        try:
            # Preprocess audio
            y = self.preprocess_audio(audio_data)
            if y is None:
                return None
            
            features = {}
            
            # 1. MFCC features
            mfcc_features = self.extract_mfcc_features(y)
            features['mfcc'] = mfcc_features
            
            # 2. Chroma features
            chroma_features = self.extract_chroma_features(y)
            features['chroma'] = chroma_features
            
            # 3. Spectral features
            spectral_features = self.extract_spectral_features(y)
            features['spectral'] = spectral_features
            
            # 4. Tonnetz features
            tonnetz_features = self.extract_tonnetz_features(y)
            features['tonnetz'] = tonnetz_features
            
            # Combine all features
            combined_features = np.concatenate([
                features['mfcc'],
                features['chroma'],
                features['spectral'],
                features['tonnetz']
            ])
            
            # Apply feature scaling
            combined_features = (combined_features - np.mean(combined_features)) / np.std(combined_features)
            
            return combined_features
            
        except Exception as e:
            st.error(f"Voice feature extraction error: {e}")
            return None
    
    def extract_mfcc_features(self, y):
        """Extract MFCC features with derivatives"""
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.sample_rate, 
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Extract delta and delta-delta
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Compute statistics
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta_std = np.std(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        delta2_std = np.std(mfcc_delta2, axis=1)
        
        # Combine all features
        mfcc_features = np.concatenate([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std
        ])
        
        return mfcc_features
    
    def extract_chroma_features(self, y):
        """Extract chroma features"""
        chroma = librosa.feature.chroma_stft(
            y=y, 
            sr=self.sample_rate,
            n_fft=2048,
            hop_length=512
        )
        
        # Compute statistics
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        chroma_features = np.concatenate([chroma_mean, chroma_std])
        
        return chroma_features
    
    def extract_spectral_features(self, y):
        """Extract spectral features"""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, 
            sr=self.sample_rate
        )
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, 
            sr=self.sample_rate
        )
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, 
            sr=self.sample_rate
        )
        
        # Zero crossing rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Compute statistics
        features = []
        for feature in [spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate]:
            features.append(np.mean(feature))
            features.append(np.std(feature))
        
        return np.array(features)
    
    def extract_tonnetz_features(self, y):
        """Extract tonal centroid features"""
        try:
            tonnetz = librosa.feature.tonnetz(
                y=y,
                sr=self.sample_rate
            )
            
            # Compute statistics
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_std = np.std(tonnetz, axis=1)
            tonnetz_features = np.concatenate([tonnetz_mean, tonnetz_std])
            
            return tonnetz_features
        except:
            return np.zeros(12)  # 6 mean + 6 std
    
    def verify_voice(self, audio_data, stored_templates, threshold=0.65):
        """Verify voice against stored templates"""
        try:
            # Extract features from input audio
            test_features = self.extract_features(audio_data)
            if test_features is None:
                return False, 0.0
            
            # Calculate similarity with each stored template
            best_similarity = 0.0
            for template in stored_templates:
                # Calculate cosine similarity
                similarity = 1 - cosine(test_features, template)
                best_similarity = max(best_similarity, similarity)
            
            # Return verification result
            return best_similarity >= threshold, best_similarity
            
        except Exception as e:
            st.error(f"Voice verification error: {e}")
            return False, 0.0
    
    def recognize_digits(self, audio_data):
        """Recognize digits from audio"""
        try:
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                if isinstance(audio_data, bytes):
                    tmp.write(audio_data)
                else:
                    # Assume it's already a file path
                    with open(audio_data, 'rb') as f:
                        tmp.write(f.read())
                tmp_path = tmp.name
            
            # Use speech recognition
            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
                
                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio)
                    # Extract digits only
                    digits = ''.join(filter(str.isdigit, text))
                    return digits
                except sr.UnknownValueError:
                    st.warning("Could not understand audio")
                    return ""
                except sr.RequestError as e:
                    st.warning(f"Could not request results: {e}")
                    # Fallback: try to use Whisper if available
                    try:
                        model = whisper.load_model("base")
                        result = model.transcribe(tmp_path)
                        text = result["text"]
                        digits = ''.join(filter(str.isdigit, text))
                        return digits
                    except:
                        return ""
            
        except Exception as e:
            st.error(f"Digit recognition error: {e}")
            return ""
        finally:
            # Cleanup
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

# ============================================================================
# BIOMETRIC FUSION ENGINE
# ============================================================================

class BiometricFusionEngine:
    """Fuses face and voice recognition results"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.face_engine = FaceRecognitionEngine()
        self.voice_engine = VoiceRecognitionEngine()
        
        # Score normalization parameters
        self.score_ranges = {
            'face': {'min': 0.0, 'max': 1.0},
            'voice': {'min': 0.0, 'max': 1.0}
        }
    
    def enroll_user(self, user_id, face_images, voice_samples, user_info=None):
        """Enroll a user with face and voice biometrics"""
        try:
            # Validate input
            if not user_id or len(face_images) < 3 or len(voice_samples) < 3:
                return False, "Insufficient data for enrollment"
            
            # Save user info
            if user_info:
                self.db.save_user(
                    user_id=user_id,
                    full_name=user_info.get('full_name', ''),
                    email=user_info.get('email', ''),
                    phone=user_info.get('phone', '')
                )
            else:
                self.db.save_user(user_id=user_id)
            
            # Process face images
            face_templates = []
            for img in face_images:
                if isinstance(img, str):
                    # Load from file
                    image = cv2.imread(img)
                else:
                    # Already an array
                    image = img
                
                if image is None:
                    continue
                
                features = self.face_engine.extract_features(image)
                if features is not None:
                    face_templates.append(features)
            
            if len(face_templates) < 1:
                return False, "Could not extract face features"
            
            # Save face templates
            for template in face_templates[:3]:  # Save up to 3 templates
                self.db.save_face_template(user_id, template)
            
            # Process voice samples
            voice_templates = []
            for audio in voice_samples:
                features = self.voice_engine.extract_features(audio)
                if features is not None:
                    voice_templates.append(features)
            
            if len(voice_templates) < 1:
                return False, "Could not extract voice features"
            
            # Save voice templates
            for template in voice_templates[:3]:  # Save up to 3 templates
                self.db.save_voice_template(user_id, template)
            
            return True, "Enrollment successful"
            
        except Exception as e:
            return False, f"Enrollment error: {str(e)}"
    
    def authenticate_user(self, user_id, face_image, voice_audio, expected_digits="", 
                         auth_type="biometric", ip_address="127.0.0.1", user_agent="streamlit"):
        """Authenticate user with face and voice"""
        try:
            # Get system settings
            settings = self.db.get_system_settings()
            
            # Check if user exists
            user = self.db.get_user(user_id)
            if not user:
                return {
                    'authenticated': False,
                    'message': 'User not found',
                    'scores': {'face': 0, 'voice': 0, 'overall': 0}
                }
            
            # Get stored templates
            face_templates = self.db.get_user_face_templates(user_id)
            voice_templates = self.db.get_user_voice_templates(user_id)
            
            if not face_templates or not voice_templates:
                return {
                    'authenticated': False,
                    'message': 'User not properly enrolled',
                    'scores': {'face': 0, 'voice': 0, 'overall': 0}
                }
            
            # Verify face
            face_verified, face_score = self.face_engine.verify_face(
                face_image, 
                face_templates,
                threshold=settings.get('face_threshold', 0.7)
            )
            
            # Verify voice
            voice_verified, voice_score = self.voice_engine.verify_voice(
                voice_audio,
                voice_templates,
                threshold=settings.get('voice_threshold', 0.65)
            )
            
            # Check digit challenge
            digit_match = True  # Default to True if not required
            if settings.get('require_digits', True) and expected_digits:
                spoken_digits = self.voice_engine.recognize_digits(voice_audio)
                digit_match = spoken_digits == expected_digits
            
            # Check liveness
            liveness_check = True  # Default to True if not required
            if settings.get('require_liveness', True):
                liveness_check = self.face_engine.detect_liveness(face_image)
            
            # Calculate overall confidence using weighted fusion
            weights = self.calculate_weights(settings)
            
            # Normalize scores
            face_norm = self.normalize_score(face_score, 'face')
            voice_norm = self.normalize_score(voice_score, 'voice')
            
            # Calculate weighted score
            weighted_score = (
                weights['face'] * face_norm +
                weights['voice'] * voice_norm +
                weights['digits'] * (1.0 if digit_match else 0.0) +
                weights['liveness'] * (1.0 if liveness_check else 0.0)
            )
            
            # Make authentication decision
            conditions = []
            if face_verified:
                conditions.append('face')
            if voice_verified:
                conditions.append('voice')
            if digit_match:
                conditions.append('digits')
            if liveness_check:
                conditions.append('liveness')
            
            authenticated = False
            if settings.get('multimodal', True):
                # Require all enabled modalities
                required = ['face', 'voice']
                if settings.get('require_digits', True):
                    required.append('digits')
                if settings.get('require_liveness', True):
                    required.append('liveness')
                
                authenticated = all(cond in conditions for cond in required)
            else:
                # Single modality is enough
                authenticated = len(conditions) >= 2
            
            # Prepare authentication data for logging
            auth_data = {
                'user_id': user_id,
                'auth_type': auth_type,
                'face_score': float(face_score),
                'voice_score': float(voice_score),
                'digit_match': digit_match,
                'liveness_check': liveness_check,
                'overall_confidence': float(weighted_score),
                'result': 'success' if authenticated else 'failed',
                'ip_address': ip_address,
                'user_agent': user_agent,
                'details': {
                    'conditions_met': conditions,
                    'weights': weights,
                    'settings_used': settings
                }
            }
            
            # Log authentication attempt
            self.db.log_authentication(auth_data)
            
            # Return result
            return {
                'authenticated': authenticated,
                'message': 'Authentication successful' if authenticated else 'Authentication failed',
                'scores': {
                    'face': float(face_score),
                    'voice': float(voice_score),
                    'overall': float(weighted_score)
                },
                'details': {
                    'face_verified': face_verified,
                    'voice_verified': voice_verified,
                    'digit_match': digit_match,
                    'liveness_check': liveness_check,
                    'conditions_met': conditions
                },
                'auth_data': auth_data
            }
            
        except Exception as e:
            st.error(f"Authentication error: {e}")
            return {
                'authenticated': False,
                'message': f'Authentication error: {str(e)}',
                'scores': {'face': 0, 'voice': 0, 'overall': 0}
            }
    
    def calculate_weights(self, settings):
        """Calculate fusion weights based on settings"""
        weights = {
            'face': 0.4,
            'voice': 0.4,
            'digits': 0.1,
            'liveness': 0.1
        }
        
        # Adjust weights based on settings
        if not settings.get('require_digits', True):
            # Redistribute digit weight to face and voice
            weights['face'] += weights['digits'] * 0.5
            weights['voice'] += weights['digits'] * 0.5
            weights['digits'] = 0
        
        if not settings.get('require_liveness', True):
            # Redistribute liveness weight to face and voice
            weights['face'] += weights['liveness'] * 0.5
            weights['voice'] += weights['liveness'] * 0.5
            weights['liveness'] = 0
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def normalize_score(self, score, modality):
        """Normalize score to [0, 1] range"""
        if modality in self.score_ranges:
            min_val = self.score_ranges[modality]['min']
            max_val = self.score_ranges[modality]['max']
            return (score - min_val) / (max_val - min_val + 1e-7)
        return score

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """Manages user sessions and security"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.active_sessions = {}
        self.failed_attempts = {}
        
    def create_session(self, user_id, ip_address="127.0.0.1", user_agent="streamlit"):
        """Create a new session for user"""
        try:
            session_id = hashlib.sha256(
                f"{user_id}_{datetime.now()}_{ip_address}".encode()
            ).hexdigest()[:32]
            
            session_data = {
                'user_id': user_id,
                'session_id': session_id,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'ip_address': ip_address,
                'user_agent': user_agent,
                'authenticated': True
            }
            
            self.active_sessions[session_id] = session_data
            return session_id
            
        except Exception as e:
            st.error(f"Session creation error: {e}")
            return None
    
    def validate_session(self, session_id):
        """Validate and update session"""
        try:
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check session timeout
            settings = self.db.get_system_settings()
            timeout_seconds = settings.get('session_timeout', 600)
            
            last_activity = session['last_activity']
            time_diff = (datetime.now() - last_activity).total_seconds()
            
            if time_diff > timeout_seconds:
                # Session expired
                del self.active_sessions[session_id]
                return False
            
            # Update last activity
            session['last_activity'] = datetime.now()
            return True
            
        except Exception as e:
            st.error(f"Session validation error: {e}")
            return False
    
    def end_session(self, session_id):
        """End a user session"""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            return True
        except Exception as e:
            st.error(f"Session end error: {e}")
            return False
    
    def track_failed_attempt(self, user_id, ip_address):
        """Track failed authentication attempts"""
        try:
            key = f"{user_id}_{ip_address}"
            
            if key not in self.failed_attempts:
                self.failed_attempts[key] = {
                    'count': 0,
                    'first_failure': datetime.now(),
                    'last_failure': datetime.now()
                }
            
            self.failed_attempts[key]['count'] += 1
            self.failed_attempts[key]['last_failure'] = datetime.now()
            
            # Check if user should be locked out
            settings = self.db.get_system_settings()
            max_attempts = settings.get('max_auth_attempts', 3)
            lockout_duration = settings.get('lockout_duration', 300)
            
            if self.failed_attempts[key]['count'] >= max_attempts:
                time_since_first = (datetime.now() - self.failed_attempts[key]['first_failure']).total_seconds()
                if time_since_first < lockout_duration:
                    return True, "Account locked due to too many failed attempts"
            
            return False, None
            
        except Exception as e:
            st.error(f"Failed attempt tracking error: {e}")
            return False, None

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

class VoiceFaceAuthApp:
    """Main Streamlit application"""
    
    def __init__(self):
        # Initialize database
        self.db = DatabaseManager()
        
        # Initialize engines
        self.fusion_engine = BiometricFusionEngine(self.db)
        self.session_manager = SessionManager(self.db)
        
        # Initialize session state
        self.init_session_state()
        
        # Apply page config
        st.set_page_config(
            page_title="Voice + Face Authentication System",
            page_icon="🔐",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS
        self.apply_custom_css()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True
        
        # User data
        if 'current_user' not in st.session_state:
            st.session_state.current_user = None
        if 'session_id' not in st.session_state:
            st.session_state.session_id = None
        
        # Authentication flow
        if 'auth_step' not in st.session_state:
            st.session_state.auth_step = 1
        if 'digit_challenge' not in st.session_state:
            st.session_state.digit_challenge = ""
        if 'enrollment_data' not in st.session_state:
            st.session_state.enrollment_data = {
                'face_images': [],
                'voice_samples': []
            }
        
        # System settings
        if 'system_settings' not in st.session_state:
            st.session_state.system_settings = self.db.get_system_settings()
        
        # UI state
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "dashboard"
    
    def apply_custom_css(self):
        """Apply custom CSS styles"""
        st.markdown("""
        <style>
        /* Main styling */
        .main {
            padding: 1rem;
        }
        
        /* Card styling */
        .card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* Success message */
        .success-msg {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
            margin: 1rem 0;
        }
        
        /* Error message */
        .error-msg {
            background-color: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
            margin: 1rem 0;
        }
        
        /* Warning message */
        .warning-msg {
            background-color: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #ffeaa7;
            margin: 1rem 0;
        }
        
        /* Info message */
        .info-msg {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #bee5eb;
            margin: 1rem 0;
        }
        
        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        
        /* Dataframe styling */
        .dataframe {
            border-radius: 5px;
            overflow: hidden;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.image("https://img.icons8.com/color/96/000000/security-checked.png", width=80)
            st.title("🔐 Voice+Face Auth")
            
            st.markdown("---")
            
            # Navigation
            menu_options = [
                ("📊 Dashboard", "dashboard"),
                ("👤 User Management", "users"),
                ("📝 Enrollment", "enrollment"),
                ("🔐 Authentication", "authentication"),
                ("📈 Analytics", "analytics"),
                ("⚙️ Settings", "settings"),
                ("ℹ️ About", "about")
            ]
            
            for icon_label, page_key in menu_options:
                if st.button(icon_label, key=f"btn_{page_key}", use_container_width=True):
                    st.session_state.current_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # System status
            st.markdown("### System Status")
            
            # Get stats
            stats = self.db.get_auth_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Users", len(self.db.get_all_users()))
            with col2:
                st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
            
            st.markdown("---")
            
            # Current user
            if st.session_state.current_user:
                st.markdown(f"**Logged in as:**")
                st.markdown(f"`{st.session_state.current_user}`")
                
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout()
            else:
                st.markdown("**Not logged in**")
    
    def dashboard_page(self):
        """Render dashboard page"""
        st.title("📊 System Dashboard")
        
        # Get statistics
        stats = self.db.get_auth_stats()
        all_users = self.db.get_all_users()
        recent_logs = self.db.get_auth_logs(limit=10)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h3>Total Users</h3>
            <h2>{}</h2>
            </div>
            """.format(len(all_users)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h3>Today's Auth</h3>
            <h2>{}</h2>
            </div>
            """.format(stats.get('today_auth', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h3>Success Rate</h3>
            <h2>{:.1f}%</h2>
            </div>
            """.format(stats.get('success_rate', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
            <h3>Active Users</h3>
            <h2>{}</h2>
            </div>
            """.format(stats.get('active_users', 0)), unsafe_allow_html=True)
        
        # Charts and data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 Authentication Trends")
            
            # Get last 30 days of logs
            try:
                logs_df = pd.DataFrame(recent_logs)
                if not logs_df.empty:
                    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
                    logs_df['date'] = logs_df['timestamp'].dt.date
                    
                    # Daily counts
                    daily_counts = logs_df.groupby('date').size().reset_index(name='count')
                    st.line_chart(daily_counts.set_index('date'))
            except:
                st.info("No authentication data available")
        
        with col2:
            st.subheader("⚡ Recent Activity")
            
            if recent_logs:
                for log in recent_logs[:5]:
                    timestamp = log.get('timestamp', '')
                    user_id = log.get('user_id', '')
                    result = log.get('result', '')
                    
                    if result == 'success':
                        icon = "✅"
                        color = "green"
                    else:
                        icon = "❌"
                        color = "red"
                    
                    st.markdown(f"""
                    <div style="padding: 5px; margin: 2px 0; border-left: 3px solid {color};">
                    <small>{timestamp}</small><br>
                    <strong>{icon} {user_id}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No recent activity")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        action_cols = st.columns(4)
        
        with action_cols[0]:
            if st.button("👤 New Enrollment", use_container_width=True):
                st.session_state.current_page = "enrollment"
                st.rerun()
        
        with action_cols[1]:
            if st.button("🔐 Test Auth", use_container_width=True):
                st.session_state.current_page = "authentication"
                st.rerun()
        
        with action_cols[2]:
            if st.button("📊 View Reports", use_container_width=True):
                st.session_state.current_page = "analytics"
                st.rerun()
        
        with action_cols[3]:
            if st.button("⚙️ Settings", use_container_width=True):
                st.session_state.current_page = "settings"
                st.rerun()
    
    def users_page(self):
        """Render user management page"""
        st.title("👤 User Management")
        
        # Search and filter
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_term = st.text_input("🔍 Search users", placeholder="User ID, name, or email")
        
        with col2:
            status_filter = st.selectbox("Status", ["All", "Active", "Inactive"])
        
        with col3:
            st.write("")  # Spacer
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Get users
        all_users = self.db.get_all_users()
        
        # Apply filters
        filtered_users = all_users
        if search_term:
            filtered_users = [
                u for u in filtered_users
                if search_term.lower() in str(u.get('user_id', '')).lower() or
                search_term.lower() in str(u.get('full_name', '')).lower() or
                search_term.lower() in str(u.get('email', '')).lower()
            ]
        
        if status_filter != "All":
            filtered_users = [
                u for u in filtered_users
                if u.get('status', 'active').lower() == status_filter.lower()
            ]
        
        # Display users in a table
        if filtered_users:
            # Convert to DataFrame for display
            users_df = pd.DataFrame(filtered_users)
            
            # Select columns to display
            display_cols = ['user_id', 'full_name', 'email', 'enrollment_date', 'last_login', 'status']
            display_cols = [col for col in display_cols if col in users_df.columns]
            
            st.dataframe(
                users_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # User details and actions
            st.subheader("User Details")
            
            selected_user = st.selectbox(
                "Select user for details/actions",
                [u['user_id'] for u in filtered_users]
            )
            
            if selected_user:
                user = self.db.get_user(selected_user)
                
                if user:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("### User Information")
                        st.json({
                            'User ID': user.get('user_id'),
                            'Full Name': user.get('full_name', 'N/A'),
                            'Email': user.get('email', 'N/A'),
                            'Phone': user.get('phone', 'N/A'),
                            'Enrollment Date': user.get('enrollment_date', 'N/A'),
                            'Last Login': user.get('last_login', 'N/A'),
                            'Status': user.get('status', 'active')
                        })
                    
                    with col2:
                        st.markdown("### User Actions")
                        
                        # Get user's auth logs
                        user_logs = self.db.get_auth_logs(user_id=selected_user, limit=5)
                        
                        st.markdown("**Recent Authentication Attempts:**")
                        if user_logs:
                            for log in user_logs:
                                result = "✅" if log.get('result') == 'success' else "❌"
                                st.write(f"{result} {log.get('timestamp')} - Score: {log.get('overall_confidence', 0):.2%}")
                        else:
                            st.write("No authentication attempts")
                        
                        # Action buttons
                        action_cols = st.columns(2)
                        
                        with action_cols[0]:
                            if st.button("🔑 Reset Password", use_container_width=True):
                                st.info("Password reset feature would be implemented here")
                        
                        with action_cols[1]:
                            if st.button("🗑️ Delete User", use_container_width=True, type="secondary"):
                                if st.checkbox(f"Confirm deletion of user {selected_user}"):
                                    if self.db.delete_user(selected_user):
                                        st.success(f"User {selected_user} deleted")
                                        st.rerun()
                                    else:
                                        st.error("Failed to delete user")
        else:
            st.info("No users found matching the criteria")
        
        # Add new user section
        with st.expander("➕ Add New User (Manual)"):
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    new_user_id = st.text_input("User ID*", placeholder="Unique identifier")
                    new_full_name = st.text_input("Full Name")
                
                with col2:
                    new_email = st.text_input("Email")
                    new_phone = st.text_input("Phone")
                
                if st.form_submit_button("Add User"):
                    if not new_user_id:
                        st.error("User ID is required")
                    else:
                        success = self.db.save_user(
                            user_id=new_user_id,
                            full_name=new_full_name,
                            email=new_email,
                            phone=new_phone
                        )
                        
                        if success:
                            st.success(f"User {new_user_id} added successfully")
                            st.rerun()
                        else:
                            st.error("Failed to add user")
    
    def enrollment_page(self):
        """Render user enrollment page"""
        st.title("📝 User Enrollment")
        
        st.markdown("""
        <div class="info-msg">
        💡 <strong>Important:</strong> For best results, ensure good lighting for face capture 
        and minimal background noise for voice recording. Capture multiple samples for better accuracy.
        </div>
        """, unsafe_allow_html=True)
        
        # Enrollment form
        with st.form("enrollment_form"):
            st.subheader("Step 1: User Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.text_input("User ID*", 
                                       placeholder="e.g., ATM123456",
                                       help="Unique identifier for the user")
                full_name = st.text_input("Full Name")
            
            with col2:
                email = st.text_input("Email")
                phone = st.text_input("Phone Number")
            
            st.markdown("---")
            st.subheader("Step 2: Face Capture")
            
            st.markdown("""
            <div class="info-msg">
            👁️ <strong>Face Capture Instructions:</strong> 
            Please capture 3-5 face images from different angles. 
            Ensure your face is well-lit and clearly visible.
            </div>
            """, unsafe_allow_html=True)
            
            # Face capture
            face_images = []
            face_cols = st.columns(3)
            
            for i in range(3):
                with face_cols[i]:
                    st.markdown(f"**Face Image {i+1}**")
                    img_file = st.camera_input(
                        f"Capture face {i+1}",
                        key=f"face_cam_{i}",
                        label_visibility="collapsed"
                    )
                    
                    if img_file:
                        # Convert to OpenCV format
                        image = Image.open(img_file)
                        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        face_images.append(image_cv)
                        st.image(img_file, caption=f"Captured {i+1}", use_column_width=True)
            
            st.markdown("---")
            st.subheader("Step 3: Voice Enrollment")
            
            st.markdown("""
            <div class="info-msg">
            🎤 <strong>Voice Enrollment Instructions:</strong> 
            Record 3-5 voice samples. Speak clearly and naturally. 
            Each recording should be 2-4 seconds long.
            </div>
            """, unsafe_allow_html=True)
            
            # Voice capture
            voice_samples = []
            voice_cols = st.columns(3)
            
            phrases = [
                "Please say the digits 1 2 3 4",
                "Please say the digits 5 6 7 8", 
                "Please say the digits 9 0 1 2"
            ]
            
            for i in range(3):
                with voice_cols[i]:
                    st.markdown(f"**Sample {i+1}**")
                    st.markdown(f"*{phrases[i]}*")
                    audio_file = st.audio_input(
                        f"Record voice {i+1}",
                        key=f"voice_rec_{i}",
                        label_visibility="collapsed"
                    )
                    
                    if audio_file:
                        voice_samples.append(audio_file.getvalue())
                        st.audio(audio_file.getvalue())
            
            st.markdown("---")
            
            # Submit button
            submitted = st.form_submit_button(
                "✅ Complete Enrollment",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                # Validate inputs
                if not user_id:
                    st.error("User ID is required")
                    return
                
                if len(face_images) < 3:
                    st.error("Please capture at least 3 face images")
                    return
                
                if len(voice_samples) < 3:
                    st.error("Please record at least 3 voice samples")
                    return
                
                # Show progress
                with st.spinner("Processing enrollment..."):
                    # Prepare user info
                    user_info = {
                        'full_name': full_name,
                        'email': email,
                        'phone': phone
                    }
                    
                    # Call enrollment engine
                    success, message = self.fusion_engine.enroll_user(
                        user_id=user_id,
                        face_images=face_images,
                        voice_samples=voice_samples,
                        user_info=user_info
                    )
                    
                    if success:
                        st.markdown(f"""
                        <div class="success-msg">
                        ✅ <strong>Enrollment Successful!</strong><br>
                        User <strong>{user_id}</strong> has been successfully enrolled.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show enrollment summary
                        st.subheader("📋 Enrollment Summary")
                        
                        summary_cols = st.columns(3)
                        
                        with summary_cols[0]:
                            st.metric("User ID", user_id)
                            st.metric("Face Samples", len(face_images))
                        
                        with summary_cols[1]:
                            st.metric("Full Name", full_name or "N/A")
                            st.metric("Voice Samples", len(voice_samples))
                        
                        with summary_cols[2]:
                            st.metric("Email", email or "N/A")
                            st.metric("Enrollment Date", datetime.now().strftime("%Y-%m-%d"))
                        
                        # Options
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🔄 Enroll Another User", use_container_width=True):
                                st.rerun()
                        
                        with col2:
                            if st.button("🔐 Test Authentication", use_container_width=True):
                                st.session_state.current_page = "authentication"
                                st.rerun()
                    else:
                        st.error(f"❌ Enrollment failed: {message}")
    
    def authentication_page(self):
        """Render authentication page"""
        st.title("🔐 User Authentication")
        
        # Check if users exist
        all_users = self.db.get_all_users()
        if not all_users:
            st.warning("No users enrolled. Please enroll users first.")
            if st.button("Go to Enrollment"):
                st.session_state.current_page = "enrollment"
                st.rerun()
            return
        
        # Authentication steps
        steps = ["User Identification", "Face Verification", "Voice Challenge", "Result"]
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
        
        # Step 1: User identification
        if current_step == 1:
            st.subheader("Step 1: User Identification")
            
            st.markdown("""
            <div class="info-msg">
            👤 Please enter your User ID or select from enrolled users.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                user_id = st.text_input(
                    "Enter User ID",
                    placeholder="Your registered User ID",
                    key="auth_user_id"
                )
                
                # Or select from dropdown
                user_options = [""] + [u['user_id'] for u in all_users]
                selected_user = st.selectbox(
                    "Or select from enrolled users",
                    user_options,
                    key="auth_user_select"
                )
                
                if selected_user:
                    user_id = selected_user
            
            with col2:
                st.markdown("**Quick Actions**")
                if st.button("🔍 Verify User", use_container_width=True):
                    if user_id and user_id in [u['user_id'] for u in all_users]:
                        st.success(f"✅ User {user_id} found")
                    elif user_id:
                        st.error(f"❌ User {user_id} not found")
                
                st.markdown("---")
                
                if st.button("📝 New Enrollment", use_container_width=True):
                    st.session_state.current_page = "enrollment"
                    st.rerun()
            
            if user_id and st.button("Continue to Face Verification", type="primary", use_container_width=True):
                if user_id not in [u['user_id'] for u in all_users]:
                    st.error(f"User {user_id} is not enrolled")
                else:
                    st.session_state.current_user = user_id
                    st.session_state.auth_step = 2
                    st.rerun()
        
        # Step 2: Face verification
        elif current_step == 2:
            st.subheader("Step 2: Face Verification")
            
            user_id = st.session_state.current_user
            
            st.markdown(f"""
            <div class="info-msg">
            👁️ <strong>User:</strong> {user_id}<br>
            Please face the camera directly. Ensure good lighting and remove any obstructions.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Camera input
                camera_img = st.camera_input(
                    "Position your face in the frame",
                    key="auth_face_camera"
                )
                
                if camera_img:
                    # Convert to OpenCV format
                    image = Image.open(camera_img)
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    st.session_state.face_image = image_cv
                    
                    # Display with face detection
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                    faces = self.fusion_engine.face_engine.face_cascade.detectMultiScale(gray, 1.1, 5)
                    
                    if len(faces) > 0:
                        # Draw rectangle around face
                        img_with_face = image_cv.copy()
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img_with_face, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Convert back for display
                        img_display = cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB)
                        st.image(img_display, caption="Face detected ✓", use_column_width=True)
                        
                        # Check liveness
                        if st.session_state.system_settings.get('require_liveness', True):
                            liveness_result = self.fusion_engine.face_engine.detect_liveness(image_cv)
                            if liveness_result:
                                st.success("✅ Liveness check passed")
                            else:
                                st.warning("⚠️ Liveness check inconclusive. Please blink or move your head.")
                    else:
                        st.warning("⚠️ No face detected. Please position your face in the frame.")
            
            with col2:
                st.markdown("### Instructions")
                st.markdown("""
                <div class="card">
                <h4>📋 Face Capture Tips:</h4>
                <ul>
                <li>Ensure good, even lighting</li>
                <li>Remove sunglasses/hat</li>
                <li>Look directly at camera</li>
                <li>Maintain neutral expression</li>
                <li>Blink naturally when prompted</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate digit challenge
                if not st.session_state.digit_challenge:
                    import random
                    st.session_state.digit_challenge = ''.join(
                        str(random.randint(0, 9)) for _ in range(4)
                    )
                
                if st.button("✅ Capture & Continue", type="primary", use_container_width=True):
                    if 'face_image' not in st.session_state or st.session_state.face_image is None:
                        st.error("Please capture a face image first")
                    else:
                        st.session_state.auth_step = 3
                        st.rerun()
                
                if st.button("← Back", use_container_width=True):
                    st.session_state.auth_step = 1
                    st.rerun()
        
        # Step 3: Voice challenge
        elif current_step == 3:
            st.subheader("Step 3: Voice Challenge")
            
            user_id = st.session_state.current_user
            challenge = st.session_state.digit_challenge
            
            st.markdown(f"""
            <div class="info-msg">
            🎤 <strong>User:</strong> {user_id}<br>
            <strong>Challenge:</strong> Please say the following digits: <strong style="font-size: 1.5em;">{challenge}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Audio recording
                voice_audio = st.audio_input(
                    f"Say the digits: {challenge}",
                    key="auth_voice_input"
                )
                
                if voice_audio:
                    audio_bytes = voice_audio.getvalue()
                    st.session_state.voice_audio = audio_bytes
                    st.audio(audio_bytes)
                    
                    # Optional: Test digit recognition
                    if st.button("🔍 Check Digits", use_container_width=True):
                        with st.spinner("Recognizing digits..."):
                            spoken_digits = self.fusion_engine.voice_engine.recognize_digits(audio_bytes)
                            
                            if spoken_digits == challenge:
                                st.success(f"✅ Digits matched: {spoken_digits}")
                            else:
                                st.warning(f"⚠️ Digits recognized: {spoken_digits}")
            
            with col2:
                st.markdown("### Voice Instructions")
                st.markdown("""
                <div class="card">
                <h4>🎤 Recording Tips:</h4>
                <ul>
                <li>Speak clearly and naturally</li>
                <li>Minimize background noise</li>
                <li>Use headphones if available</li>
                <li>Say all digits in sequence</li>
                <li>Wait for beep before speaking</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("✅ Verify & Complete", type="primary", use_container_width=True):
                    if 'voice_audio' not in st.session_state or st.session_state.voice_audio is None:
                        st.error("Please record your voice first")
                    else:
                        st.session_state.auth_step = 4
                        st.rerun()
                
                if st.button("← Back", use_container_width=True):
                    st.session_state.auth_step = 2
                    st.rerun()
        
        # Step 4: Results
        elif current_step == 4:
            st.subheader("Step 4: Authentication Result")
            
            user_id = st.session_state.current_user
            
            with st.spinner("Verifying authentication..."):
                # Perform authentication
                result = self.fusion_engine.authenticate_user(
                    user_id=user_id,
                    face_image=st.session_state.face_image,
                    voice_audio=st.session_state.voice_audio,
                    expected_digits=st.session_state.digit_challenge,
                    auth_type="biometric"
                )
            
            # Display result
            if result['authenticated']:
                st.markdown(f"""
                <div class="success-msg" style="text-align: center; padding: 2rem;">
                <h1 style="color: green;">✅ ACCESS GRANTED</h1>
                <h3>Authentication Successful!</h3>
                <p>User <strong>{user_id}</strong> has been authenticated.</p>
                <p>You may proceed with your transaction.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create session
                session_id = self.session_manager.create_session(user_id)
                if session_id:
                    st.session_state.session_id = session_id
            else:
                st.markdown(f"""
                <div class="error-msg" style="text-align: center; padding: 2rem;">
                <h1 style="color: red;">❌ ACCESS DENIED</h1>
                <h3>Authentication Failed</h3>
                <p>User <strong>{user_id}</strong> could not be authenticated.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show scores
            st.subheader("📊 Authentication Scores")
            
            scores = result.get('scores', {})
            details = result.get('details', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Face Score", f"{scores.get('face', 0):.2%}")
                st.caption("Verified" if details.get('face_verified') else "Failed")
            
            with col2:
                st.metric("Voice Score", f"{scores.get('voice', 0):.2%}")
                st.caption("Verified" if details.get('voice_verified') else "Failed")
            
            with col3:
                st.metric("Overall Confidence", f"{scores.get('overall', 0):.2%}")
            
            with col4:
                st.metric("Result", "PASS" if result['authenticated'] else "FAIL")
                st.caption("✓" if details.get('digit_match') else "✗ Digit match")
                st.caption("✓" if details.get('liveness_check') else "✗ Liveness")
            
            # Show failure reasons if any
            if not result['authenticated']:
                with st.expander("❌ Failure Details"):
                    conditions_met = details.get('conditions_met', [])
                    required = ['face', 'voice']
                    
                    if st.session_state.system_settings.get('require_digits', True):
                        required.append('digits')
                    if st.session_state.system_settings.get('require_liveness', True):
                        required.append('liveness')
                    
                    missing = [req for req in required if req not in conditions_met]
                    
                    if missing:
                        st.error(f"Missing conditions: {', '.join(missing)}")
                    
                    st.json(result.get('auth_data', {}))
            
            # Next steps
            st.markdown("---")
            st.subheader("Next Steps")
            
            action_cols = st.columns(3)
            
            with action_cols[0]:
                if st.button("🔄 Authenticate Again", use_container_width=True):
                    self.reset_auth_flow()
                    st.rerun()
            
            with action_cols[1]:
                if st.button("👤 Different User", use_container_width=True):
                    self.reset_auth_flow()
                    st.session_state.auth_step = 1
                    st.rerun()
            
            with action_cols[2]:
                if st.button("🏠 Return to Dashboard", use_container_width=True):
                    self.reset_auth_flow()
                    st.session_state.current_page = "dashboard"
                    st.rerun()
    
    def reset_auth_flow(self):
        """Reset authentication flow state"""
        st.session_state.auth_step = 1
        st.session_state.digit_challenge = ""
        if 'face_image' in st.session_state:
            del st.session_state.face_image
        if 'voice_audio' in st.session_state:
            del st.session_state.voice_audio
    
    def analytics_page(self):
        """Render analytics page"""
        st.title("📈 Analytics & Reports")
        
        # Get data
        all_logs = self.db.get_auth_logs(limit=1000)
        stats = self.db.get_auth_stats()
        
        if not all_logs:
            st.info("No authentication data available")
            return
        
        # Convert to DataFrame
        logs_df = pd.DataFrame(all_logs)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Attempts", stats.get('total_auth', 0))
        
        with col2:
            st.metric("Successful", stats.get('success_auth', 0))
        
        with col3:
            st.metric("Failed", stats.get('failed_auth', 0))
        
        with col4:
            st.metric("Success Rate", f"{stats.get('success_rate', 0):.1f}%")
        
        # Charts
        st.subheader("📊 Performance Metrics")
        
        tab1, tab2, tab3 = st.tabs(["Timeline", "User Analysis", "Score Distribution"])
        
        with tab1:
            # Timeline analysis
            try:
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
                logs_df['date'] = logs_df['timestamp'].dt.date
                logs_df['hour'] = logs_df['timestamp'].dt.hour
                
                # Daily success rate
                daily_stats = logs_df.groupby('date').agg({
                    'result': lambda x: (x == 'success').mean(),
                    'user_id': 'count'
                }).reset_index()
                daily_stats.columns = ['date', 'success_rate', 'attempts']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Daily Success Rate**")
                    st.line_chart(daily_stats.set_index('date')['success_rate'])
                
                with col2:
                    st.markdown("**Daily Attempts**")
                    st.bar_chart(daily_stats.set_index('date')['attempts'])
            except Exception as e:
                st.error(f"Timeline analysis error: {e}")
        
        with tab2:
            # User analysis
            try:
                user_stats = logs_df.groupby('user_id').agg({
                    'result': lambda x: (x == 'success').mean(),
                    'timestamp': 'count',
                    'overall_confidence': 'mean'
                }).reset_index()
                user_stats.columns = ['user_id', 'success_rate', 'attempts', 'avg_confidence']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top Users by Success Rate**")
                    top_users = user_stats.sort_values('success_rate', ascending=False).head(10)
                    st.dataframe(top_users[['user_id', 'success_rate', 'attempts']])
                
                with col2:
                    st.markdown("**User Attempt Distribution**")
                    attempt_dist = user_stats['attempts'].value_counts().sort_index()
                    st.bar_chart(attempt_dist)
            except Exception as e:
                st.error(f"User analysis error: {e}")
        
        with tab3:
            # Score distribution
            try:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Face Score Distribution**")
                    if 'face_score' in logs_df.columns:
                        hist_values = np.histogram(logs_df['face_score'].dropna(), bins=20, range=(0, 1))[0]
                        st.bar_chart(hist_values)
                
                with col2:
                    st.markdown("**Voice Score Distribution**")
                    if 'voice_score' in logs_df.columns:
                        hist_values = np.histogram(logs_df['voice_score'].dropna(), bins=20, range=(0, 1))[0]
                        st.bar_chart(hist_values)
            except Exception as e:
                st.error(f"Score distribution error: {e}")
        
        # Detailed logs
        st.subheader("📋 Detailed Authentication Logs")
        
        with st.expander("View Full Logs"):
            st.dataframe(logs_df, use_container_width=True)
        
        # Export option
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export as CSV", use_container_width=True):
                csv = logs_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"auth_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.button("📊 Export as JSON", use_container_width=True):
                json_data = logs_df.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"auth_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def settings_page(self):
        """Render settings page"""
        st.title("⚙️ System Settings")
        
        tab1, tab2, tab3 = st.tabs(["Security", "Database", "System"])
        
        with tab1:
            st.subheader("🔒 Security Settings")
            
            # Get current settings
            settings = st.session_state.system_settings
            
            with st.form("security_settings_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    face_threshold = st.slider(
                        "Face Recognition Threshold",
                        min_value=0.5,
                        max_value=0.95,
                        value=float(settings.get('face_threshold', 0.7)),
                        step=0.05,
                        help="Higher values mean stricter face matching"
                    )
                    
                    voice_threshold = st.slider(
                        "Voice Recognition Threshold",
                        min_value=0.5,
                        max_value=0.95,
                        value=float(settings.get('voice_threshold', 0.65)),
                        step=0.05,
                        help="Higher values mean stricter voice matching"
                    )
                
                with col2:
                    require_liveness = st.checkbox(
                        "Require Liveness Check",
                        value=settings.get('require_liveness', True),
                        help="Prevents spoofing with photos or videos"
                    )
                    
                    require_digits = st.checkbox(
                        "Require Digit Challenge",
                        value=settings.get('require_digits', True),
                        help="Prevents replay attacks with recorded audio"
                    )
                    
                    multimodal = st.checkbox(
                        "Multi-modal Authentication",
                        value=settings.get('multimodal', True),
                        help="Require both face AND voice verification"
                    )
                
                st.markdown("---")
                st.subheader("Access Control")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    max_attempts = st.number_input(
                        "Max Failed Attempts",
                        min_value=1,
                        max_value=10,
                        value=int(settings.get('max_auth_attempts', 3)),
                        help="Maximum allowed failed attempts before lockout"
                    )
                
                with col4:
                    lockout_duration = st.number_input(
                        "Lockout Duration (seconds)",
                        min_value=60,
                        max_value=3600,
                        value=int(settings.get('lockout_duration', 300)),
                        help="Duration of lockout after max failed attempts"
                    )
                
                session_timeout = st.number_input(
                    "Session Timeout (seconds)",
                    min_value=60,
                    max_value=86400,
                    value=int(settings.get('session_timeout', 600)),
                    help="Automatic logout after inactivity"
                )
                
                if st.form_submit_button("💾 Save Security Settings", type="primary", use_container_width=True):
                    # Update settings
                    new_settings = {
                        'face_threshold': face_threshold,
                        'voice_threshold': voice_threshold,
                        'require_liveness': require_liveness,
                        'require_digits': require_digits,
                        'multimodal': multimodal,
                        'max_auth_attempts': max_attempts,
                        'lockout_duration': lockout_duration,
                        'session_timeout': session_timeout
                    }
                    
                    if self.db.update_system_settings(new_settings):
                        st.session_state.system_settings = self.db.get_system_settings()
                        st.success("✅ Security settings saved successfully!")
                    else:
                        st.error("❌ Failed to save settings")
        
        with tab2:
            st.subheader("🗄️ Database Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Database Information")
                
                # Get database info
                try:
                    self.db.connect()
                    self.db.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = self.db.cursor.fetchall()
                    
                    st.metric("Total Tables", len(tables))
                    
                    self.db.cursor.execute("SELECT COUNT(*) FROM users")
                    user_count = self.db.cursor.fetchone()[0]
                    st.metric("Total Users", user_count)
                    
                    self.db.cursor.execute("SELECT COUNT(*) FROM auth_logs")
                    log_count = self.db.cursor.fetchone()[0]
                    st.metric("Auth Logs", log_count)
                    
                except Exception as e:
                    st.error(f"Database error: {e}")
                finally:
                    self.db.close()
            
            with col2:
                st.markdown("### Database Actions")
                
                if st.button("🔄 Optimize Database", use_container_width=True):
                    try:
                        self.db.connect()
                        self.db.cursor.execute("VACUUM;")
                        self.db.conn.commit()
                        st.success("Database optimized successfully!")
                    except Exception as e:
                        st.error(f"Optimization error: {e}")
                    finally:
                        self.db.close()
                
                if st.button("📤 Export Database", use_container_width=True):
                    st.info("Database export feature would save the entire database to a file")
                
                if st.button("🗑️ Clear Old Logs", use_container_width=True, type="secondary"):
                    if st.checkbox("Delete logs older than 30 days"):
                        try:
                            self.db.connect()
                            cutoff_date = datetime.now() - timedelta(days=30)
                            self.db.cursor.execute(
                                "DELETE FROM auth_logs WHERE timestamp < ?",
                                (cutoff_date,)
                            )
                            deleted_count = self.db.cursor.rowcount
                            self.db.conn.commit()
                            st.success(f"Deleted {deleted_count} old logs")
                        except Exception as e:
                            st.error(f"Error clearing logs: {e}")
                        finally:
                            self.db.close()
        
        with tab3:
            st.subheader("⚙️ System Configuration")
            
            st.markdown("### System Information")
            
            sys_info = {
                "System Version": "1.0.0",
                "Database Type": "SQLite",
                "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Python Version": "3.9+",
                "Streamlit Version": "1.28.0+",
                "OpenCV Version": cv2.__version__,
                "Librosa Version": librosa.__version__
            }
            
            for key, value in sys_info.items():
                st.text(f"{key}: {value}")
            
            st.markdown("### Maintenance")
            
            if st.button("🔄 Restart System", use_container_width=True, type="secondary"):
                st.warning("This will restart the application")
                if st.checkbox("Confirm restart"):
                    st.rerun()
            
            if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
                st.error("⚠️ DANGER ZONE ⚠️")
                if st.checkbox("I understand this will delete ALL data"):
                    if st.checkbox("Type 'DELETE ALL' to confirm"):
                        try:
                            # Delete all data
                            self.db.connect()
                            
                            tables = ['auth_logs', 'face_templates', 'voice_templates', 'users', 'system_settings']
                            for table in tables:
                                self.db.cursor.execute(f"DELETE FROM {table}")
                            
                            self.db.conn.commit()
                            self.db.init_database()  # Reinitialize with default settings
                            
                            st.success("All data cleared successfully!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error clearing data: {e}")
                        finally:
                            self.db.close()
    
    def about_page(self):
        """Render about page"""
        st.title("ℹ️ About Voice + Face Authentication System")
        
        st.markdown("""
        <div class="card">
        <h2>🔐 Voice + Face Authentication Module</h2>
        <p><strong>Version:</strong> 2.0.0 (Database Edition)</p>
        <p><strong>Last Updated:</strong> December 2024</p>
        <p><strong>Purpose:</strong> Secure, accessible authentication for visually-impaired ATM users</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Project Overview")
        
        st.markdown("""
        This system provides a complete biometric authentication solution that combines:
        
        - **🎤 Voice Recognition**: Speaker verification with anti-spoofing
        - **👁️ Face Recognition**: Facial authentication with liveness detection
        - **🗄️ Database Storage**: Secure template storage with encryption
        - **♿ Accessibility**: Audio-guided prompts for visually-impaired users
        
        ### 🔧 Technical Architecture
        
        The system is built with:
        
        1. **Database Layer**: SQLite/PostgreSQL for data persistence
        2. **Biometric Engine**: Advanced feature extraction and matching
        3. **Fusion Engine**: Multi-modal score fusion and decision making
        4. **Web Interface**: Streamlit-based interactive UI
        5. **Security Layer**: Template encryption and session management
        
        ### 📊 Key Features
        
        - **User Enrollment**: Capture face and voice biometrics
        - **Authentication**: Multi-factor verification
        - **Analytics**: Detailed reporting and statistics
        - **User Management**: Complete user administration
        - **System Settings**: Configurable security parameters
        
        ### 🛡️ Security Measures
        
        - **Template Storage**: Only encrypted biometric templates stored
        - **Anti-Spoofing**: Liveness detection and random challenges
        - **Session Management**: Secure session handling
        - **Audit Logging**: Complete authentication trail
        - **Access Control**: Configurable security thresholds
        
        ### 📞 Support
        
        For support or questions:
        - **Email**: support@accessible-auth.com
        - **Documentation**: [docs.accessible-auth.com](https://docs.accessible-auth.com)
        - **Issues**: [GitHub Issues](https://github.com/yourusername/voice-face-auth/issues)
        
        ### 📄 License
        
        MIT License - Open source for educational and research purposes.
        """)
        
        # Quick demo guide
        with st.expander("🎮 Quick Demo Guide"):
            st.markdown("""
            1. **Setup**:
               - Run the application: `streamlit run app.py`
               - Database will be created automatically
            
            2. **Enrollment**:
               - Go to Enrollment page
               - Enter user information
               - Capture 3+ face images
               - Record 3+ voice samples
               - Complete enrollment
            
            3. **Authentication**:
               - Go to Authentication page
               - Enter User ID
               - Capture face
               - Say digit challenge
               - View result
            
            4. **Administration**:
               - View dashboard for statistics
               - Manage users in User Management
               - Adjust settings in Settings page
               - View analytics in Analytics page
            """)
    
    def logout(self):
        """Logout current user"""
        if st.session_state.session_id:
            self.session_manager.end_session(st.session_state.session_id)
        
        st.session_state.current_user = None
        st.session_state.session_id = None
        self.reset_auth_flow()
        st.session_state.current_page = "dashboard"
        st.rerun()
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        self.sidebar()
        
        # Render current page
        current_page = st.session_state.current_page
        
        if current_page == "dashboard":
            self.dashboard_page()
        elif current_page == "users":
            self.users_page()
        elif current_page == "enrollment":
            self.enrollment_page()
        elif current_page == "authentication":
            self.authentication_page()
        elif current_page == "analytics":
            self.analytics_page()
        elif current_page == "settings":
            self.settings_page()
        elif current_page == "about":
            self.about_page()
        
        # Footer
        st.markdown("---")
        st.caption(f"🔐 Voice + Face Authentication System v2.0 | "
                  f"© {datetime.now().year} | Database Edition")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    try:
        # Create and run application
        app = VoiceFaceAuthApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please check the console for details")

if __name__ == "__main__":
    main()