import streamlit as st

st.set_page_config(
    page_title="CE440 Laboratory #2",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

r"""
# CE440 Laboratory #2 - Bed sediment morphodynamics
"""

st.video("assets/timelapse.mp4")

"""
## Objectives 

> The primary objectives of this lab demonstration are for you to 
increase your conceptual understanding of flow interactions with 
granular sediment beds, and to learn methods for scientific image 
capture and analysis. You will measure morphodynamic changes in a 
sand bed (i.e., bedform motion), process the images to obtain the bed 
surface topography, and calculate statistics for bedform morphology 
and motion
"""

r"""
## Instruments

> You will use an imaging setup that is already in place for observing 
sediment, bed, and solute (dye) motion in the flume in LG17. The flow 
conditions will be similar to those used in Lab 1, but with continuous 
sediment motion. Cameras collect images of the bedforms through the 
transparent side-wall of the flume. This provides clear images of the 
surface of the sand bed at the contact point with the flume wall.
"""

r"""
## Data to be acquired

> You will see how images are taken and process them using python codes 
in order to determine bedform morphodynamic quantities (height, length, 
celerity and sediment flux). You will be able to change some input 
parameters in the code that dictate how well the water-sediment interface 
is extracted. You will not need to write any additional code, other 
than showing the results of your analysis.
"""
