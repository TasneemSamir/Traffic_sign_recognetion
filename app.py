# ============================================================
# Traffic Sign Recognition - Streamlit Web Demo
# ============================================================
# This is a web application that loads our trained CNN model
# and provides an interactive interface to classify traffic signs.
#
# To run: streamlit run app.py

# ============ IMPORTS ============
import streamlit as st           # Web framework for ML demos
import tensorflow as tf          # Deep learning framework (to load our model)
import numpy as np               # Numerical operations
import cv2                       # Image processing
from PIL import Image            # Image loading from uploads
import json                      # Load class names from JSON
import plotly.graph_objects as go  # Beautiful interactive charts
import matplotlib.pyplot as plt  # For Grad-CAM visualization
import os                        # File system operations

# ============================================================
# PAGE CONFIGURATION
# ============================================================
# This MUST be the first Streamlit command in the script.
# It sets up the page title, icon, and layout.
st.set_page_config(
    page_title="Traffic Sign Recognition AI",
    page_icon="🚦",
    layout="wide",  # Use full screen width
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
# Make the app look professional with custom CSS.
# Streamlit allows injecting CSS via st.markdown with unsafe_allow_html.
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-top: 0;
    }
    /* Prediction box styling */
    .prediction-box {
        background-color: #f0f9ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1e3a8a;
        margin: 10px 0;
    }
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & METADATA (Cached for performance)
# ============================================================
# @st.cache_resource ensures the model loads only ONCE,
# not on every user interaction. Critical for performance!

@st.cache_resource
def load_model():
    """Load the trained CNN model from disk."""
    model = tf.keras.models.load_model('traffic_sign_model_final.keras')
    # Build the model's graph by calling it once (needed for Grad-CAM)
    _ = model(np.zeros((1, 32, 32, 3), dtype=np.float32))
    return model

@st.cache_data
def load_class_names():
    """Load the class ID → name mapping from JSON."""
    with open('class_names.json', 'r') as f:
        # JSON keys are strings, convert back to int
        return {int(k): v for k, v in json.load(f).items()}

@st.cache_data
def load_metrics():
    """Load the project metrics from JSON."""
    with open('metrics_summary.json', 'r') as f:
        return json.load(f)

# Load everything once at startup
model = load_model()
class_names = load_class_names()
metrics = load_metrics()

# ============================================================
# IMAGE PREPROCESSING (must match training!)
# ============================================================
# CRITICAL: Use the EXACT same preprocessing as during training,
# otherwise the model will give wrong predictions.

def apply_clahe(img):
    """Apply CLAHE for contrast enhancement (same as training)."""
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l_enhanced = clahe.apply(l)
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

def preprocess_image(image):
    """
    Preprocess an uploaded image for the model.
    
    Args:
        image: PIL Image from user upload
    
    Returns:
        Tuple of (preprocessed_array, display_image)
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Handle grayscale images (convert to RGB)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    # Handle RGBA images (drop alpha channel)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize to 32x32 (model's expected input size)
    img_resized = cv2.resize(img_array, (32, 32), interpolation=cv2.INTER_AREA)
    
    # Apply CLAHE
    img_enhanced = apply_clahe(img_resized)
    
    # Normalize to [0, 1] and add batch dimension
    img_normalized = img_enhanced.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, img_enhanced

# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================
# Same Grad-CAM logic from our training notebook,
# adapted for the demo.

def make_gradcam_heatmap(img_array, model):
    """
    Generate Grad-CAM heatmap using manual layer iteration.
    
    This bulletproof version doesn't access model.input or model.output,
    avoiding the "layer has never been called" error in Keras 3.x.
    
    Approach:
    - Phase 1: Pass input through layers 0 to last_conv_idx, save conv output
    - Phase 2: Continue through remaining layers to get final predictions
    - Compute gradients between class score and the saved conv output
    """
    # Step 1: Find the index of the last Conv2D layer
    last_conv_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i
    
    # Step 2: Convert numpy array to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Step 3: Use GradientTape to track operations for gradient calculation
    with tf.GradientTape() as tape:
        # Phase 1: Forward pass through layers 0 to last conv layer
        x = img_tensor
        for i in range(last_conv_idx + 1):
            x = model.layers[i](x)
        
        # Save conv layer output - this is what we'll compute gradients for
        conv_outputs = x
        tape.watch(conv_outputs)
        
        # Phase 2: Continue forward pass through remaining layers
        for i in range(last_conv_idx + 1, len(model.layers)):
            x = model.layers[i](x)
        
        predictions = x
        
        # Get the predicted class (highest probability)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Step 4: Compute gradients of class score w.r.t. conv layer output
    # This tells us how much each feature map contributed to the prediction
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Step 5: Average gradients across spatial dimensions (batch, H, W)
    # This gives us the importance weight for each feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Step 6: Weight feature maps by their importance and sum
    conv_outputs = conv_outputs[0]  # Remove batch dimension
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)  # Remove single dimensions
    
    # Step 7: Apply ReLU and normalize to [0, 1] for visualization
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.5):
    """
    Overlay the Grad-CAM heatmap on the original image.
    
    Args:
        img: Original image, shape (H, W, 3) with values in [0, 1] or [0, 255]
        heatmap: Grad-CAM heatmap, shape (h, w) with values in [0, 1]
        alpha: Transparency of heatmap overlay (0=invisible, 1=fully opaque)
    
    Returns:
        Combined image as numpy array (H, W, 3) with values 0-255 uint8
    """
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap from [0,1] to [0,255] uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply 'jet' colormap: blue (cool) -> green -> yellow -> red (hot)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # OpenCV uses BGR by default, convert to RGB for display
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Convert original image to uint8 if needed
    if img.max() <= 1.0:
        img_uint8 = np.uint8(255 * img)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Blend the heatmap with the original image
    # alpha controls how visible the heatmap is vs the original
    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed
# ============================================================
# SIDEBAR (Navigation)
# ============================================================
# The sidebar contains project info and navigation between pages.
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1535/1535067.png", width=100)
    st.title("🚦 Traffic Sign AI")
    st.markdown("---")
    
    # Page navigation using radio buttons
    page = st.radio(
        "Navigation",
        ["🎯 Live Demo", "📊 Project Metrics", "🧠 About the Model", "📚 About the Project"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
    st.metric("Classes", metrics['num_classes'])
    st.metric("Parameters", f"{metrics['total_parameters']:,}")
    
    st.markdown("---")
    st.markdown("**Course:** Image Processing & AI")
    st.markdown("**Dataset:** GTSRB (43 classes)")

# ============================================================
# PAGE 1: LIVE DEMO
# ============================================================
if page == "🎯 Live Demo":
    st.markdown('<h1 class="main-title">🚦 Traffic Sign Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload a traffic sign image and let the AI identify it!</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # File uploader widget
    uploaded_file = st.file_uploader(
        "📤 Upload a traffic sign image",
        type=['jpg', 'jpeg', 'png', 'ppm'],
        help="Upload a clear image of a traffic sign for classification"
    )
    
    # When a file is uploaded
    if uploaded_file is not None:
        # Open the image with PIL
        image = Image.open(uploaded_file)
        
        # Two-column layout: original image | predictions
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### 🤖 AI Prediction")
            
            # Show a spinner while processing
            with st.spinner("Analyzing image..."):
                # Preprocess and predict
                img_batch, img_processed = preprocess_image(image)
                predictions = model.predict(img_batch, verbose=0)[0]
                
                # Get top prediction
                top_class = int(np.argmax(predictions))
                top_confidence = float(predictions[top_class]) * 100
                top_name = class_names[top_class]
            
            # Display the prediction in a styled box
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color:#1e3a8a; margin:0;">{top_name}</h2>
                <h3 style="color:#10b981; margin:0;">Confidence: {top_confidence:.2f}%</h3>
                <p style="color:#6b7280; margin:5px 0 0 0;">Class ID: {top_class}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence indicator
            if top_confidence > 90:
                st.success("✅ Very high confidence prediction")
            elif top_confidence > 70:
                st.warning("⚠️ Moderate confidence - image may be unclear")
            else:
                st.error("❌ Low confidence - image quality may be poor")
        
        st.markdown("---")
        
        # ============ TOP 5 PREDICTIONS ============
        st.markdown("### 📊 Top 5 Predictions")
        
        # Get indices of top 5 classes
        top5_indices = np.argsort(predictions)[-5:][::-1]
        top5_names = [class_names[i] for i in top5_indices]
        top5_confidences = [predictions[i] * 100 for i in top5_indices]
        
        # Create an interactive Plotly bar chart
        fig = go.Figure(go.Bar(
            x=top5_confidences,
            y=top5_names,
            orientation='h',  # Horizontal bars
            text=[f"{c:.2f}%" for c in top5_confidences],
            textposition='outside',
            marker=dict(
                color=top5_confidences,
                colorscale='Blues',
                showscale=False
            )
        ))
        
        fig.update_layout(
            xaxis_title="Confidence (%)",
            xaxis=dict(range=[0, 105]),
            yaxis=dict(autorange="reversed"),  # Highest at top
            height=350,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### 🔥 Grad-CAM: Where the AI Looked")
        st.markdown(
            "This heatmap shows which regions of the image were most important for the prediction. "
            "**Red areas** = high importance, **Blue areas** = low importance."
        )
        
        with st.spinner("Generating Grad-CAM heatmap..."):
            # Generate the heatmap
            heatmap = make_gradcam_heatmap(img_batch, model)
            # Overlay on the processed image
            overlay = overlay_gradcam(img_processed.astype('float32') / 255.0, heatmap, alpha=0.5)
        
        # Display in three columns: processed image | heatmap | overlay
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("**Processed Image (32x32)**")
            st.image(img_processed, use_container_width=True)
        
        with col_b:
            st.markdown("**Heatmap (raw)**")
            # Apply jet colormap to raw heatmap
            heatmap_display = cv2.applyColorMap(
                np.uint8(255 * cv2.resize(heatmap, (32, 32))),
                cv2.COLORMAP_JET
            )
            heatmap_display = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2RGB)
            st.image(heatmap_display, use_container_width=True)
        
        with col_c:
            st.markdown("**Overlay**")
            st.image(overlay, use_container_width=True)
    
    else:
        # No file uploaded yet - show instructions
        st.info("👆 Upload an image above to get started!")
        st.markdown("### 💡 Tips for best results:")
        st.markdown("""
        - Use clear, well-lit images of traffic signs
        - Ensure the sign occupies most of the image
        - Avoid heavy occlusion or extreme angles
        - The model recognizes 43 different German traffic sign classes
        """)

# ============================================================
# PAGE 2: PROJECT METRICS
# ============================================================
elif page == "📊 Project Metrics":
    st.markdown('<h1 class="main-title">📊 Project Metrics</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Display key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.2f}%")
    with col2:
        st.metric("Validation Accuracy", f"{metrics['best_val_accuracy']*100:.2f}%")
    with col3:
        st.metric("Test Images", f"{metrics['total_test_images']:,}")
    with col4:
        st.metric("Misclassified", f"{metrics['misclassified']}")
    
    st.markdown("---")
    
    # Display the visualizations from training
    st.markdown("### 📈 Training Curves")
    if os.path.exists("assets/training_curves.png"):
        st.image("assets/training_curves.png", use_container_width=True)
    
    st.markdown("### 🎯 Confusion Matrix")
    if os.path.exists("assets/confusion_matrix.png"):
        st.image("assets/confusion_matrix.png", use_container_width=True)
    
    st.markdown("### 📊 Per-Class Accuracy")
    if os.path.exists("assets/per_class_accuracy.png"):
        st.image("assets/per_class_accuracy.png", use_container_width=True)
    
    st.markdown("### 🔥 Grad-CAM Examples")
    if os.path.exists("assets/gradcam_visualization.png"):
        st.image("assets/gradcam_visualization.png", use_container_width=True)

# ============================================================
# PAGE 3: ABOUT THE MODEL
# ============================================================
elif page == "🧠 About the Model":
    st.markdown('<h1 class="main-title">🧠 About the Model</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🏗️ Architecture: Custom CNN")
    st.markdown(f"""
    The model is a **Convolutional Neural Network** with:
    - **3 convolutional blocks** (32 → 64 → 128 filters)
    - **Batch Normalization** for stable training
    - **Dropout** layers for regularization
    - **Fully connected layers** for final classification
    - **Total parameters:** {metrics['total_parameters']:,}
    """)
    
    st.markdown("### ⚙️ Training Details")
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Input size | 32×32×3 (RGB) |
    | Output classes | 43 |
    | Optimizer | Adam (lr=0.001) |
    | Loss function | Categorical Crossentropy |
    | Batch size | 64 |
    | Epochs trained | {metrics['epochs_trained']} |
    | Data augmentation | Rotation, shift, zoom, shear |
    | Preprocessing | CLAHE + normalization |
    """)
    
    st.markdown("### 🔬 Image Processing Techniques Used")
    st.markdown("""
    1. **Resizing** – Standardize all images to 32×32
    2. **CLAHE** – Contrast Limited Adaptive Histogram Equalization for lighting normalization
    3. **Color space conversion** – BGR → RGB → LAB → RGB
    4. **Normalization** – Pixel values scaled to [0, 1]
    5. **Data augmentation** – Random transformations for robustness
    """)

# ============================================================
# PAGE 4: ABOUT THE PROJECT
# ============================================================
elif page == "📚 About the Project":
    st.markdown('<h1 class="main-title">📚 About the Project</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Objective
    Build an AI system that can automatically recognize traffic signs from images,
    combining classical **image processing** techniques with modern **deep learning**.
    
    ### 🚗 Why It Matters
    Traffic Sign Recognition is a critical component of:
    - **Autonomous vehicles** (self-driving cars)
    - **ADAS** (Advanced Driver Assistance Systems)
    - **Smart traffic management**
    - **Accessibility tools** for visually impaired drivers
    
    ### 📊 Dataset: GTSRB
    - **Name:** German Traffic Sign Recognition Benchmark
    - **Classes:** 43 different traffic signs
    - **Training images:** ~39,000
    - **Test images:** ~12,600
    - **Source:** IJCNN 2011 Competition
    
    ### 🛠️ Technologies Used
    - **Python** – Programming language
    - **TensorFlow/Keras** – Deep learning framework
    - **OpenCV** – Image processing
    - **NumPy & Pandas** – Data handling
    - **Matplotlib & Seaborn** – Visualization
    - **Streamlit** – Web demo
    - **Plotly** – Interactive charts
    
    ### 🏆 Results
    - **99.03% test accuracy** on 12,630 unseen images
    - **99.97% validation accuracy** during training
    - **Grad-CAM** visualization for model interpretability
    
    ### 🔮 Future Improvements
    - Train on real-time video for sign detection (not just classification)
    - Deploy on edge devices (Raspberry Pi, mobile phones)
    - Add support for traffic signs from other countries
    - Implement ensemble methods to reduce errors further
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "Built with ❤️ using TensorFlow & Streamlit | Image Processing & AI Course Project"
    "</p>",
    unsafe_allow_html=True
)