import streamlit as st

st.set_page_config(
    page_title="CE440 Laboratory #2",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


cols = st.columns([2,1])
with cols[0]:
    """
    We will use [pillow](https://python-pillow.org/) and 
    [ImageMagick](https://imagemagick.org/) for basic image processing.
    """

with cols[1]:
    """
    > **References:**
    > - The ImageMagick Development Team. (2021). ImageMagick. Retrieved from https://imagemagick.org
    > - Clark, A. (2015). Pillow (PIL Fork) Documentation. readthedocs. 
    """