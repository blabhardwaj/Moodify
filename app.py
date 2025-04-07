import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import uuid

# Set page configuration
st.set_page_config(
    page_title="Moodify My Sketch 🎨",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("🎨 Moodify My Sketch")
st.write(
    "Express your mood through drawing! Once you're done, save your sketch, "
    "and we'll match it to a song that fits your vibe."
)

# Layout for controls
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        # Stroke color picker
        stroke_color = st.color_picker("🎨 Pick a stroke color", "#000000")

        # Brush size slider
        stroke_width = st.slider("✏️ Brush size", min_value=1, max_value=25, value=5)

    with col2:
        # Background color picker
        bg_color = st.color_picker("🌈 Background color", "#FFFFFF")

# Drawing canvas
st.write("🖌️ **Start Drawing Below:**")
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Not used in free draw mode
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Save sketch button and functionality
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype("uint8"))

    # Add spacing and save button
    st.write("")
    if st.button("💾 Save Drawing"):
        filename = f"sketch_{uuid.uuid4().hex[:8]}.png"
        img.save(filename)
        st.success(f"✅ Your sketch has been saved as `{filename}`!")
