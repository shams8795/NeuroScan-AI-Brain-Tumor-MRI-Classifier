import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

# ======== Page Configuration ========
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="⚕️",

    layout="wide"
)

# ======== Dark Mode CSS ========
st.markdown("""
    <style>
    /* Dark Background */
    .stApp {
        background: #0a0e27;
    }
    
    /* Main containers */
    .main {
        padding: 2rem;
        background: #0a0e27;
    }
    
    /* Title */
    h1 {
        text-align: center;
        color: #7dd3fc;
        text-shadow: 0 0 25px rgba(125, 211, 252, 0.5);
        font-size: 2.8rem;
        font-weight: 800;
        margin: 2rem auto 3rem auto;
        padding: 1.5rem 0;
    }
    
    /* Section headers */
    h2 {
        color: #7dd3fc;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 2rem;
        margin-top: 1.5rem;
        border-left: 5px solid #d97a3a;
        padding-left: 1.2rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(217, 122, 58, 0.2);
    }
    
    h3 {
        color: #00e5ff;
    }
    
    /* Dark cards */
    .dark-card {
        background: rgba(20, 30, 60, 0.6);
        border: 1px solid rgba(0, 229, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
    }
    
    /* Prediction result card */
    .prediction-card {
    background: rgba(20, 30, 55, 0.55);
    border: 2px solid rgba(125, 211, 252, 0.35);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
    box-shadow: 0 0 16px rgba(125, 211, 252, 0.15);
}

.prediction-card h2 {
    color: #7dd3fc;
    font-size: 2.1rem;
    font-weight: 700;
    text-shadow: none;
}


    
    /* Confidence card */
    .confidence-card {
    background: rgba(20, 30, 55, 0.55);
    border: 2px solid rgba(199, 210, 254, 0.35);
    border-radius: 15px;
    padding: 1.4rem;
    text-align: center;
    margin: 1rem 0;
    box-shadow: 0 0 14px rgba(199, 210, 254, 0.15);
}

.confidence-card h3 {
    color: #c7d2fe;
    font-size: 1.4rem;
    font-weight: 600;
    text-shadow: none;
}

    /* Metric cards */
    .metric-card {
        background: rgba(15, 20, 40, 0.8);
        border: 2px solid rgba(0, 229, 255, 0.3);
        border-left: 4px solid #00e5ff;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-left: 4px solid #ff6b35;
        box-shadow: 0 0 15px rgba(255, 107, 53, 0.3);
    }
    
    .metric-label {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .metric-value {
        color: #00ff9f;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Buttons */
    
    .stButton>button {
    width: 100%;
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    color: white;
    padding: 1rem 2rem;
    font-size: 1.15rem;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    box-shadow: 0 5px 20px rgba(139, 92, 246, 0.45);
    transition: all 0.3s ease;
    margin: 1rem 0;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.65);
    transform: translateY(-2px);
}

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 20, 40, 0.6);
        border: 3px dashed rgba(0, 229, 255, 0.5);
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #7dd3fc !important;
        font-size: 1rem !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f1419;
        border-right: 2px solid rgba(217, 122, 58, 0.3);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00e5ff;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: #94a3b8;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: rgba(255, 159, 28, 0.1);
        border: 1px solid rgba(255, 159, 28, 0.3);
        border-radius: 10px;
        color: #ff9f1c;
    }
    
    /* Text colors */
    p, span, div {
        color: #94a3b8;
    }
    
    /* Image container */
    .stImage {
        border-radius: 12px;
        border: 3px solid rgba(0, 229, 255, 0.4);
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 229, 255, 0.2);
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0 1.5rem;
    }
    
    /* Add spacing between sections */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ======== Load Model ========
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('resnet_brain_tumor.keras')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ======== Preprocessing ========
def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ======== Class Names ========
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model
model = load_model()

# ======== Title ========
st.markdown("<h1>  🩻Brain Tumor MRI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ======== Sidebar Info ========
with st.sidebar:
    st.markdown("## 📊 Classification Types")
    st.markdown("""
    - **Glioma**: Brain tissue tumor
    - **Meningioma**: Membrane tumor  
    - **No Tumor**: Healthy scan
    - **Pituitary**: Gland tumor
    """)
    
    st.markdown("---")
    
    st.markdown("## 💡 How to Use")
    st.markdown("""
    1. Upload an MRI scan
    2. Click 'Analyze Image'
    3. View the results
    """)
    
    st.markdown("---")
    st.warning("⚠️ For educational purposes only")

# ======== Main Layout ========
col1, col2, col3 = st.columns([1, 1, 1])

# ======== Column 1: Upload ========
with col1:
    st.markdown("<h2> 📤Upload MRI Scan</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analyze Image"):
            st.session_state['analyze'] = True
            st.session_state['image'] = image_pil
    else:
        st.info("📁 Please upload an MRI image to begin")

# ======== Column 2: Results ========
with col2:
    st.markdown("<h2>🧠 Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'analyze' in st.session_state and st.session_state['analyze'] and model:
        with st.spinner('🔄 Analyzing...'):
            try:
                image_pil = st.session_state['image']
                processed_img = preprocess_image(image_pil)
                predictions_array = model.predict(processed_img, verbose=0)[0]
                
                predictions = {
                    CLASS_NAMES[i]: float(predictions_array[i] * 100) 
                    for i in range(len(CLASS_NAMES))
                }
                
                predicted_class = CLASS_NAMES[np.argmax(predictions_array)]
                confidence = predictions[predicted_class]
                
                class_display = predicted_class.replace('notumor', 'No Tumor').title()
                
                st.session_state['predictions'] = predictions
                st.session_state['predicted_class'] = class_display
                st.session_state['confidence'] = confidence
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    if 'predicted_class' in st.session_state:
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🔬 {st.session_state['predicted_class']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="confidence-card">
            <h3>⚡ {st.session_state['confidence']:.2f}% Confidence</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #7dd3fc; margin-top: 2rem; margin-bottom: 1rem; font-size: 1.2rem;'>📋 Detailed Scores</h3>", unsafe_allow_html=True)
        
        for class_name, score in st.session_state['predictions'].items():
            display_name = class_name.replace('notumor', 'No Tumor').title()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{display_name}</div>
                <div class="metric-value">{score:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📊 Results will appear here after analysis")

# ======== Column 3: Chart ========
with col3:
    st.markdown("<h2> 📈 Probability Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'predictions' in st.session_state:
        classes = [name.replace('notumor', 'No Tumor').title() for name in CLASS_NAMES]
        scores = [st.session_state['predictions'][name] for name in CLASS_NAMES]
        
        # Create gradient colors
        colors = []
        for i, name in enumerate(classes):
            if name == st.session_state['predicted_class']:
                colors.append('#00ff9f')
            else:
                colors.append('#2a2e45')
        
        # Bar Chart
        fig_bar = go.Figure(data=[
            go.Bar(
                x=classes,
                y=scores,
                marker=dict(
                    color=colors,
                    line=dict(color='#d97a3a', width=2)
                ),
                text=[f'{s:.1f}%' for s in scores],
                textposition='outside',
                textfont=dict(size=12, color='#00e5ff', weight='bold')
            )
        ])
        
        fig_bar.update_layout(
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            xaxis=dict(
                title=dict(text="Tumor Type", font=dict(color='#7dd3fc', size=13)),
                tickfont=dict(color='#7dd3fc'),
                gridcolor='rgba(0, 229, 255, 0.1)',
                showgrid=True
            ),
            yaxis=dict(
                title=dict(text="Probability (%)", font=dict(color='#7dd3fc', size=13)),
                tickfont=dict(color='#7dd3fc'),
                gridcolor='rgba(0, 229, 255, 0.1)',
                range=[0, 105],
                showgrid=True
            ),
            height=350,
            showlegend=False,
            margin=dict(t=20, b=50, l=50, r=20)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Pie Chart
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=classes,
                values=scores,
                hole=0.6,
                marker=dict(
                    colors=['#00ff9f' if c == st.session_state['predicted_class'] else '#2a2e45' for c in classes],
                    line=dict(color='#d97a3a', width=2)
                ),
                textinfo='label+percent',
                textfont=dict(color='#7dd3fc', size=11, weight='bold')
            )
        ])
        
        fig_pie.update_layout(
            plot_bgcolor='#0a0e27',
            paper_bgcolor='#0a0e27',
            height=300,
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(
                text=f'{st.session_state["confidence"]:.1f}%',
                x=0.5, y=0.5,
                font=dict(size=28, color='#e6a84e', weight='bold'),
                showarrow=False
            )]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        if st.button("🔄 Analyze Another Image"):
            for key in ['analyze', 'image', 'predictions', 'predicted_class', 'confidence']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.info("📈 Charts will appear here after analysis")