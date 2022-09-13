import streamlit as st

st.set_page_config(
    page_title="[NU CEE440] Lab 2 - References",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

cols = st.columns([2, 1])
with cols[0]:
    """
    We will use [pillow](https://python-pillow.org/) and 
    [ImageMagick](https://imagemagick.org/) for basic image processing.

    Check [Dallmann et al. (2020)](https://doi.org/10.1029/2019wr027010) for details on how to calculate 
    bed sediment dynamics based on image processing.

    """

with cols[1]:
    """
    > **References:**
    > - The ImageMagick Development Team. (2021). ImageMagick. Retrieved from https://imagemagick.org
    > - Clark, A. (2015). Pillow (PIL Fork) Documentation. readthedocs. 
    >
    > - [Dallmann et al. (2020)](https://doi.org/10.1029/2019wr027010). Impacts of suspended clay particle deposition on Sand-Bed morphodynamics. Water Resources Research, 56(8).
    """
