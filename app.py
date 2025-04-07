import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import uuid

st.title("🎨 Moodify My Sketch")

st.write("Draw something that shows your mood. We'll match a song to it!")

# Stroke color
stroke_color = st.color_picker("Pick a stroke color", "#000000")

# Brush size
stroke_width = st.slider("Brush size", min_value=1, max_value=25, value=5)

# Background color (optional)
bg_color = st.color_picker("Background color", "#FFFFFF")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Not used in free draw
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

# Save sketch
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype("uint8"))

    if st.button("💾 Save Drawing"):
        filename = f"sketch_{uuid.uuid4().hex[:8]}.png"
        img.save(filename)
        st.success(f"Saved as {filename}")
