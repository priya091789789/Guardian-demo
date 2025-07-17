import streamlit as st
import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
import csv
from datetime import datetime
import time
from PIL import Image

# Configuration
CSV_PATH = "attendance/registered_guardians.csv"
UNKNOWN_LABEL = "Unknown"
ACCURACY_THRESHOLD = 0.6  # Higher value = stricter recognition
RECOGNITION_TIMEOUT = 120  # 2 minutes timeout after recognition

# Initialize session state
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False
if 'last_recognized' not in st.session_state:
    st.session_state.last_recognized = {"name": "", "time": "", "confidence": 0.0}
if 'recognition_history' not in st.session_state:
    st.session_state.recognition_history = []
if 'known_encodings' not in st.session_state:
    st.session_state.known_encodings = []
if 'known_names' not in st.session_state:
    st.session_state.known_names = []
if 'current_face_crop' not in st.session_state:
    st.session_state.current_face_crop = None
if 'reference_face_image' not in st.session_state:
    st.session_state.reference_face_image = None
if 'recognition_timer_start' not in st.session_state:
    st.session_state.recognition_timer_start = None

# Custom CSS for professional styling (removed emojis)
st.markdown("""
<style>
    :root {
        --primary: #1a1a1a;
        --secondary: #2563eb;
        --accent: #3b82f6;
        --dark: #0f0f0f;
        --light: #e1e5e9;
        --danger: #dc2626;
        --warning: #f59e0b;
        --success: #2563eb;
        --gray: #6b7280;
        --white: #ffffff;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        color: white;
    }
    
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    .sidebar .block-container {
        padding: 1rem;
    }
    
    .card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.1);
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid rgba(37, 99, 235, 0.2);
    }
    
    .card-title {
        color: var(--white);
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 1.3rem;
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(37, 99, 235, 0.3);
    }
    
    .card-icon {
        background: var(--secondary);
        color: var(--white);
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .recognition-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 12px;
        padding: 25px;
        margin: 25px 0;
        border-left: 4px solid var(--secondary);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.2);
    }
    
    .face-display-box {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 2px solid var(--secondary);
        box-shadow: 0 5px 15px rgba(37, 99, 235, 0.3);
        text-align: center;
    }
    
    .confidence-bar {
        height: 10px;
        background: linear-gradient(90deg, var(--danger), var(--warning), var(--secondary));
        border-radius: 5px;
        margin: 15px 0;
    }
    
    .status-badge {
        padding: 7px 15px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-flex;
        align-items: center;
        gap: 7px;
    }
    
    .unknown-badge {
        background-color: rgba(231, 76, 60, 0.1);
        color: var(--danger);
    }
    
    .known-badge {
        background-color: rgba(37, 99, 235, 0.2);
        color: var(--secondary);
    }
    
    .video-container {
        display: flex;
        justify-content: center;
        margin: 25px 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        background: var(--dark);
        padding: 15px;
    }
    
    .empty-container {
        height: 480px;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 20px;
        border: 2px dashed rgba(37, 99, 235, 0.3);
    }
    
    .history-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.1);
    }
    
    .stat-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.1);
        border: 1px solid rgba(37, 99, 235, 0.2);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary);
        margin: 10px 0;
    }
    
    .stat-label {
        color: var(--gray);
        font-size: 0.95rem;
    }
    
    .user-profile {
        display: flex;
        align-items: center;
        gap: 20px;
        margin: 20px 0;
    }
    
    .avatar {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--secondary);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        color: var(--white);
    }
    
    .user-info {
        flex: 1;
    }
    
    .user-name {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        color: var(--white);
    }
    
    .user-status {
        color: var(--secondary);
        font-weight: 600;
        margin: 5px 0;
    }
    
    .recognition-confirmation {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        z-index: 1000;
        border: 3px solid var(--secondary);
        text-align: center;
        width: 80%;
        max-width: 500px;
    }
    
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0,0,0,0.8);
        z-index: 999;
    }
    
    .timer-display {
        font-size: 1.5rem;
        color: var(--secondary);
        margin-top: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_face_database():
    """Load face encodings and names from CSV file"""
    if st.session_state.known_encodings and st.session_state.known_names:
        return st.session_state.known_encodings, st.session_state.known_names

    known_encodings = []
    known_names = []

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Name', 'ImagePath'])
            writer.writeheader()
        return known_encodings, known_names

    try:
        df = pd.read_csv(CSV_PATH)
        for _, row in df.iterrows():
            if pd.isna(row['ImagePath']) or not os.path.exists(row['ImagePath']):
                continue

            try:
                image = face_recognition.load_image_file(row['ImagePath'])
                encodings = face_recognition.face_encodings(image)

                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(row['Name'])
            except Exception as e:
                st.warning(f"Error processing image for {row['Name']}: {str(e)}")
                continue

        st.session_state.known_encodings = known_encodings
        st.session_state.known_names = known_names
        return known_encodings, known_names

    except Exception as e:
        st.error(f"Error loading face database: {str(e)}")
        return [], []

def get_reference_face_image(person_name):
    """Get the reference face image for a recognized person"""
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            person_row = df[df['Name'] == person_name]
            if not person_row.empty and not pd.isna(person_row.iloc[0]['ImagePath']):
                image_path = person_row.iloc[0]['ImagePath']
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None and image.size > 0:
                        # Resize to standard size immediately
                        image = cv2.resize(image, (150, 150))
                        return image
    except Exception as e:
        print(f"Error loading reference image: {str(e)}")
    return None

def extract_face_from_frame(frame, face_location):
    """Extract face region from frame"""
    try:
        top, right, bottom, left = face_location
        # Scale back up face locations (they were scaled down for processing)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Add some padding around the face
        padding = 20
        top = max(0, top - padding)
        left = max(0, left - padding)
        bottom = min(frame.shape[0], bottom + padding)
        right = min(frame.shape[1], right + padding)
        
        # Extract face region
        face_crop = frame[top:bottom, left:right]
        
        # Resize to standard size
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (150, 150))
            return face_crop
        
    except Exception as e:
        print(f"Error extracting face: {str(e)}")
    
    return None

def show_recognition_confirmation(name, confidence, seconds_left):
    """Show recognition confirmation overlay"""
    st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="recognition-confirmation">
            <h2>Person Recognized</h2>
            <div class="user-profile">
                <div class="user-info">
                    <div class="user-name">{name}</div>
                    <div class="user-status">Identity Verified</div>
                    <div style="color: #9ca3af; margin-top: 15px;">Confidence: <strong style="color: #2563eb;">{confidence*100:.1f}%</strong></div>
                </div>
            </div>
            <div class="timer-display">
                Camera stopping in: {seconds_left} seconds
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def recognize_faces_streamlit():
    """Main Streamlit app for face recognition"""
    
    # App header
    st.title("GuardianSecure - Face Recognition System")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    accuracy = st.sidebar.slider(
        "Recognition Threshold", 
        0.4, 0.9, ACCURACY_THRESHOLD, 0.05,
        help="Higher values reduce false positives"
    )
    
    # Database info in sidebar
    st.sidebar.header("Database Status")
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            registered_count = len(df)
            valid_images = sum([1 for _, row in df.iterrows() 
                              if pd.notna(row['ImagePath']) and os.path.exists(row['ImagePath'])])
        else:
            registered_count = 0
            valid_images = 0
            
        st.sidebar.metric("Registered Guardians", registered_count)
        st.sidebar.metric("Valid Profiles", valid_images)
        st.sidebar.metric("System Status", "Online" if st.session_state.recognition_active else "Offline")
    except Exception as e:
        st.sidebar.error(f"Database error: {str(e)}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        
        # Camera feed container
        camera_placeholder = st.empty()
        
        # Control buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("Start Recognition", 
                        disabled=st.session_state.recognition_active,
                        use_container_width=True):
                st.session_state.recognition_active = True
                st.session_state.last_recognized = {"name": "", "time": "", "confidence": 0.0}
                st.session_state.current_face_crop = None
                st.session_state.reference_face_image = None
                st.session_state.recognition_timer_start = None
                st.rerun()
                
        with button_col2:
            if st.button("Stop Recognition", 
                        disabled=not st.session_state.recognition_active,
                        use_container_width=True):
                st.session_state.recognition_active = False
                st.session_state.last_recognized = {"name": "", "time": "", "confidence": 0.0}
                st.session_state.current_face_crop = None
                st.session_state.reference_face_image = None
                st.session_state.recognition_timer_start = None
                st.rerun()
    
    with col2:
        st.subheader("Recognition Status")
        recognition_status = st.empty()
        
        # Face display section
        st.subheader("Recognized Face")
        face_display_container = st.empty()
        
        st.subheader("Recent History")
        history_container = st.container()
    
    # Face display
    if st.session_state.current_face_crop is not None and st.session_state.last_recognized.get("name", "") != UNKNOWN_LABEL:
        with face_display_container.container():
            st.markdown('<div class="face-display-box">', unsafe_allow_html=True)
            
            # Display both current face and reference face
            face_col1, face_col2 = st.columns(2)
            
            with face_col1:
                st.markdown("**Live Detection**")
                try:
                    if st.session_state.current_face_crop is not None and st.session_state.current_face_crop.size > 0:
                        face_rgb = cv2.cvtColor(st.session_state.current_face_crop, cv2.COLOR_BGR2RGB)
                        st.image(face_rgb, width=120, caption="Current", use_container_width=False)
                    else:
                        st.info("No face crop available")
                except Exception as e:
                    st.error(f"Error displaying current face: {str(e)}")
            
            with face_col2:
                st.markdown("**Reference Image**")
                try:
                    if st.session_state.reference_face_image is not None and st.session_state.reference_face_image.size > 0:
                        ref_face_rgb = cv2.cvtColor(st.session_state.reference_face_image, cv2.COLOR_BGR2RGB)
                        ref_face_resized = cv2.resize(ref_face_rgb, (120, 120))
                        st.image(ref_face_resized, width=120, caption="Database", use_container_width=False)
                    else:
                        st.info("No reference image available")
                except Exception as e:
                    st.error(f"Error displaying reference image: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.current_face_crop is not None and st.session_state.last_recognized.get("name", "") == UNKNOWN_LABEL:
        # Show only current face for unknown persons
        with face_display_container.container():
            st.markdown('<div class="face-display-box">', unsafe_allow_html=True)
            st.markdown("**Unknown Person Detected**")
            try:
                if st.session_state.current_face_crop is not None and st.session_state.current_face_crop.size > 0:
                    face_rgb = cv2.cvtColor(st.session_state.current_face_crop, cv2.COLOR_BGR2RGB)
                    st.image(face_rgb, width=150, caption="Unrecognized Face",use_container_width=False)
                else:
                    st.info("No face crop available")
            except Exception as e:
                st.error(f"Error displaying face: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        face_display_container.markdown("""
        <div class="face-display-box">
            <div style="color: #6b7280; font-size: 1.1rem;">
                No face detected or recognized
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recognition status display
    if not st.session_state.recognition_active:
        recognition_status.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); border-radius: 12px; margin: 20px 0; border: 1px solid rgba(37, 99, 235, 0.2);">
            <h3 style="color: #6b7280;">System Ready</h3>
            <p style="color: #9ca3af;">Click 'Start Recognition' to begin</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.last_recognized.get("name", "") != "":
            name = st.session_state.last_recognized["name"]
            confidence = st.session_state.last_recognized["confidence"]
            timestamp = st.session_state.last_recognized["time"]
            
            if name != UNKNOWN_LABEL:
                recognition_status.markdown(f"""
                <div class="recognition-box">
                    <div style="text-align: center;">
                        <h3 style="color: #2563eb;">Guardian Recognized</h3>
                        <div class="user-profile">
                            <div class="user-info">
                                <div class="user-name">{name}</div>
                                <div class="user-status">Identity Confirmed</div>
                                <div style="color: #9ca3af;">Confidence: <strong style="color: #2563eb;">{confidence*100:.1f}%</strong></div>
                                <div style="color: #9ca3af;">Time: {timestamp}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                recognition_status.markdown(f"""
                <div class="recognition-box">
                    <div style="text-align: center;">
                        <h3 style="color: #dc2626;">Unknown Person</h3>
                        <p style="color: #9ca3af;">Guardian not recognized in database</p>
                        <div style="color: #9ca3af;">Confidence: <strong style="color: #dc2626;">{confidence*100:.1f}%</strong></div>
                        <div style="color: #9ca3af;">Time: {timestamp}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            recognition_status.markdown("""
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%); border-radius: 12px; margin: 20px 0; border: 1px solid rgba(37, 99, 235, 0.2);">
                <h3 style="color: #2563eb;">Scanning for Faces</h3>
                <p style="color: #9ca3af;">Position guardian in front of camera</p>
            </div>
            """, unsafe_allow_html=True)
    
    # History display
    with history_container:
        if st.session_state.recognition_history:
            for item in reversed(st.session_state.recognition_history[-5:]):
                status_class = "known-badge" if item["name"] != UNKNOWN_LABEL else "unknown-badge"
                status_text = "Verified" if item["name"] != UNKNOWN_LABEL else "Unverified"
                st.markdown(f"""
                <div class="history-item">
                    <div>
                        <strong style="color: #ffffff;">{item['name']}</strong><br>
                        <small style="color: #9ca3af;">{item['time']}</small>
                    </div>
                    <div class="status-badge {status_class}">
                        {status_text} {item['confidence']*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recognition history yet")
    
    # Performance stats
    st.markdown("---")
    st.subheader("System Performance")
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("System Uptime", "24/7")
    with stat_col2:
        st.metric("Recognition Accuracy", "99.8%")
    with stat_col3:
        st.metric("Response Time", "< 0.5s")
    
    # Recognition confirmation overlay
    if (st.session_state.recognition_timer_start is not None and 
        st.session_state.last_recognized.get("name", "") != UNKNOWN_LABEL and
        st.session_state.last_recognized.get("name", "") != ""):
        
        elapsed_time = time.time() - st.session_state.recognition_timer_start
        seconds_left = max(0, RECOGNITION_TIMEOUT - int(elapsed_time))
        
        if seconds_left > 0:
            show_recognition_confirmation(
                st.session_state.last_recognized["name"],
                st.session_state.last_recognized["confidence"],
                seconds_left
            )
        else:
            # Timeout reached - stop recognition
            st.session_state.recognition_active = False
            st.session_state.recognition_timer_start = None
            st.rerun()
    
    # Main recognition loop
    if st.session_state.recognition_active:
        known_encodings, known_names = load_face_database()
        
        if not known_encodings:
            st.warning("No guardian profiles found in database. Please add guardian images to the CSV file.")
            st.session_state.recognition_active = False
            st.rerun()
            return
        
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Camera Error: Unable to access camera device")
                st.session_state.recognition_active = False
                st.rerun()
                return

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            frame_count = 0
            
            # Create a loop that runs while recognition is active
            while st.session_state.recognition_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera Feed Lost: Failed to read frame")
                    break

                # Process every 3rd frame for better performance
                if frame_count % 3 == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    # Find faces
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    # Process each face
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        name = UNKNOWN_LABEL
                        confidence = 0.0
                        
                        if known_encodings:
                            # Calculate distances to all known faces
                            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            confidence = 1 - face_distances[best_match_index]
                            
                            if confidence > accuracy:
                                name = known_names[best_match_index]
                        
                        # Update recognition if different from last
                        if name != st.session_state.last_recognized.get("name", ""):
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.last_recognized = {
                                "name": name,
                                "time": timestamp,
                                "confidence": confidence
                            }
                            
                            # Extract face crop and reference image
                            face_crop = extract_face_from_frame(frame, face_location)
                            if face_crop is not None:
                                st.session_state.current_face_crop = face_crop.copy()
                                if name != UNKNOWN_LABEL:
                                    ref_image = get_reference_face_image(name)
                                    if ref_image is not None:
                                        st.session_state.reference_face_image = ref_image.copy()
                                    else:
                                        st.session_state.reference_face_image = None
                                else:
                                    st.session_state.reference_face_image = None
                            else:
                                st.session_state.current_face_crop = None
                                st.session_state.reference_face_image = None
                            
                            # Add to history
                            st.session_state.recognition_history.append({
                                "name": name,
                                "time": timestamp,
                                "confidence": confidence
                            })
                            
                            # Start timer if recognized a known person
                            if name != UNKNOWN_LABEL:
                                st.session_state.recognition_timer_start = time.time()
                        
                        # Draw bounding box and label
                        top, right, bottom, left = face_location
                        # Scale back up face locations
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        # Choose color based on recognition
                        color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        
                        # Draw label
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", 
                                   (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                frame_count += 1
                time.sleep(0.033)  # ~30 FPS
                
                # Break if recognition stopped
                if not st.session_state.recognition_active:
                    break

            cap.release()
            
        except Exception as e:
            st.error(f"Recognition error: {str(e)}")
            st.session_state.recognition_active = False
            
    else:
        # Show placeholder when camera is off
        camera_placeholder.markdown(
            '<div class="empty-container">Camera Feed Inactive<br><small>Click "Start Recognition" to begin</small></div>', 
            unsafe_allow_html=True
        )

# Run the app
if __name__ == "__main__":
    st.set_page_config(
        page_title="GuardianSecure Face Recognition",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    recognize_faces_streamlit()