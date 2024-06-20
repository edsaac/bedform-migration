import streamlit as st

st.set_page_config(
    page_title="[NU CEE440] Lab 2 - References",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/style.css") as f:
    css = f.read()
    st.markdown(
        f"""
        <style>
            {css}
        </style>
        """, unsafe_allow_html=True
    )

"""
# 🗄️ References
"""

st.video("assets/timelapse.mp4", autoplay=True, muted=True)

"****"

cols = st.columns([1, 1])
with cols[0]:
    """
    **Info:**

    We used [pillow](https://python-pillow.org/) and 
    [ImageMagick](https://imagemagick.org/) for image processing.

    Interactive plots were built using [Plotly](https://plotly.com/python/). 
    Static plots were rendered using [Matplotlib](https://matplotlib.org/)

    Data sets manipulation was done using [Pandas](https://pandas.pydata.org/docs/user_guide/index.html#user-guide). 

    [NumPy](https://numpy.org/doc/stable/) was used for array programming and calculation and [SciPy](https://docs.scipy.org/doc/scipy/) was used for signal processing. 

    This site was built using [Streamlit](https://streamlit.io/).

    Check [Dallmann et al. (2020)](https://doi.org/10.1029/2019wr027010) for details on how to calculate 
    bed sediment dynamics based on image processing.

    """

with cols[1]:
    """
    > **References:**
    > - [Dallmann et al. (2020)](https://doi.org/10.1029/2019wr027010). Impacts of suspended clay particle deposition on Sand-Bed morphodynamics. Water Resources Research, 56(8).

    > - The ImageMagick Development Team. (2021). ImageMagick. Retrieved from https://imagemagick.org
    > - [Clark, A. (2015)](https://pillow.readthedocs.io/en/stable/index.html). Pillow (PIL Fork) Documentation. readthedocs. 
    > - [Hunter J. D. (2007)](https://doi.org/10.1109/MCSE.2007.55), "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95.
    > - Plotly Technologies Inc. (2015). Plotly. Retrieved from https://plotly.com/python/
    > - [McKinney, W. (2010)](https://doi.org/10.25080/Majora-92bf1922-00a). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference, pp. 56-61.
    > - [Harris, C.R., et al. (2020)](https://doi.org/10.1038/s41586-020-2649-2) Array programming with NumPy. Nature 585, 357–362.
    > - [Virtanen, P. et al. (2020)](https://doi.org/10.1038/s41592-019-0686-2) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
    > - Streamlit - A faster way to build and share data apps (2022). Retrieved from https://streamlit.io/
    """
