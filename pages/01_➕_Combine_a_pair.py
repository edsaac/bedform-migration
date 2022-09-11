import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from st_lg17cam import *

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title = "[NU CEE440] - Processing a pair of photos", 
    page_icon  = "➕",
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

"""
# ➕ Combine a pair of images
"""
graph = generateProcessGraph(whichStep="merge")
st.graphviz_chart(graph,use_container_width=True)

###################################
## Session state management
###################################
if "uploadedFilesCheck" not in st.session_state.keys():
    st.session_state.uploadedFilesCheck = True

if "globalParameters" not in st.session_state.keys():
    st.session_state.globalParameters = dict()

if "restarted_btn" not in st.session_state.keys():
    st.session_state.restarted_btn = False

if "save_and_leave" in st.session_state.keys():
    if st.session_state.save_and_leave:
        switch_page("Color classification")

if st.session_state.restarted_btn:
    del st.session_state.leftPic
    del st.session_state.rightPic

###################################
## Start by adding pictures
###################################
with st.form(key="filesForm", clear_on_submit= True):
    """
    ## ▶️ Upload a pair of photos
    """
    col1, col2 = st.columns(2)
    with col1: 
        st.file_uploader("Left photo","JPG",False,key="leftPic")
    with col2: 
        st.file_uploader("Right photo","JPG",False,key="rightPic")
    st.form_submit_button("Process!")

###################################
## Start processing
###################################
if st.session_state.leftPic and st.session_state.rightPic and not st.session_state.restarted_btn:

    leftBytes = st.session_state.leftPic
    rightBytes = st.session_state.rightPic

    with st.sidebar:
        "#### Image processing parameters"

    """
    *****
    ## 0️ Uploaded images
    """
    with st.expander("🔵 Uploaded images:",expanded=True):
        st.warning(
            """
            The timestamp of both pictures **should not 
            differ** by more than a few seconds.
            """,
            icon="📷")

        raw_imgs = [Image.open(leftBytes), Image.open(rightBytes)]
        
        fig = show_two_imgs(raw_imgs,addTimestamp=True)
        st.pyplot(fig, transparent=True)

    """
    *****
    ## 1️⃣ Correct distortions
    """
    with st.expander("🔵 Barrel correction:",expanded=True):
        with st.echo():
            ## Parameters used for distortion
            BARREL_CORRECTIONS = (0.000, -0.015, 0.000)
        
        barr_imgs = [fixBarrelDistortion(img,BARREL_CORRECTIONS) for img in raw_imgs]
        fig = show_two_imgs(barr_imgs)
        st.pyplot(fig, transparent=True)

    with st.expander("🔵 4-point perspective correction:",expanded=True):
        with st.echo():
            ## Parameters used for distortion
            PERS_ORIGIN_LEFT  = {'x':[0,1162,1162,0]  ,'y':[533,532,99,87]}
            PERS_TARGET_LEFT  = {'x':[0,1162,1162,0]  ,'y':[515,515,75,75]}
            PERS_ORIGIN_RIGHT = {'x':[84,1200,1200,84],'y':[495,500,64,49]}
            PERS_TARGET_RIGHT = {'x':[84,1200,1200,84],'y':[515,515,75,75]}

        pers_imgs = [fixPerspective(img, orig, target) for img, orig, target 
            in zip(
                barr_imgs,
                [PERS_ORIGIN_LEFT, PERS_ORIGIN_RIGHT],
                [PERS_TARGET_LEFT, PERS_TARGET_RIGHT])]

        fig = show_two_imgs(pers_imgs)
        st.pyplot(fig, transparent=True)

    """ 
    *****
    ## 2️⃣ Additional edits
    """
    with st.expander("🔵 Crop top and bottom:",expanded=True):
        crop_imgs = [img.crop((0, 250, img.size[0], 500)) for img in pers_imgs]
        fig = show_two_imgs(crop_imgs)
        st.pyplot(fig, transparent=True)

    with st.expander("🔵 Equalize",expanded=True):
        np_imgs = [np.asarray(img.convert('L')) for img in crop_imgs]
        equa_imgs = [np.interp(img, (img.min(), img.max()), (0, 255)) for img in np_imgs]
        inte_imgs = [Image.fromarray(img.astype(np.uint8),"L") for img in equa_imgs]
        
        fig = show_two_imgs(inte_imgs,imshow_kwargs={'cmap':'Greys_r'})
        st.pyplot(fig, transparent=True)

    """ 
    *****
    ## 3️⃣ Overlap and merge

    """
    with st.expander("Overlap",expanded=True):
        
        cols = st.columns([2,1])
        with cols[0]:
            st.info("In order to merge both images in one, find where they should overlap.", icon="✴️")
        
        with cols[1]:
            XSHIFT = st.number_input("Where in the first picture does the second begin?", 500, 1200, 1078, 1, key="XSHIFT")
        
        x, y = np.meshgrid(np.arange(inte_imgs[0].width),np.arange(inte_imgs[0].height,0,-1))
        
        fig,ax= plt.subplots(1,1,figsize=[20,6])
        
        # Overlap region
        ax.add_patch(
            mpatches.Rectangle(
                (XSHIFT,0), 
                inte_imgs[0].width - XSHIFT, 
                inte_imgs[0].height + inte_imgs[1].height + 20, 
                ec="purple",lw=3,fill=False,zorder=3))
        
        # Right picture
        ax.pcolormesh(
            x + XSHIFT,
            y,
            equa_imgs[1],
            cmap='Greys_r', shading='nearest', zorder=1)

        # Left picture
        ax.pcolormesh(
            x, 
            y + inte_imgs[0].height + 20, 
            equa_imgs[0],
            cmap='Greys_r',shading='nearest',alpha=1,zorder=2)
        
        ax.set_aspect('equal')
        st.pyplot(fig,transparent=True)

        # Show merged picture
        (width1, height1) = inte_imgs[0].size
        (width2, height2) = inte_imgs[1].size

        result_width = width1 + XSHIFT
        result_height = max(height1, height2)

        joined_img = Image.new('L', (result_width, result_height))
        joined_img.paste(im=inte_imgs[1], box=(XSHIFT, 0))
        joined_img.paste(im=inte_imgs[0], box=(0, 0))
        
        st.image(joined_img,caption="Merged pictures")
    
    """ 
    *****
    ## 4️⃣ Save progress
    """
    
    ## Save globalParameters configuration
    st.session_state.globalParameters["BARREL_CORRECTIONS"] = BARREL_CORRECTIONS
    st.session_state.globalParameters["PERS_ORIGIN_LEFT"] = PERS_ORIGIN_LEFT
    st.session_state.globalParameters["PERS_ORIGIN_RIGHT"] = PERS_ORIGIN_RIGHT
    st.session_state.globalParameters["PERS_TARGET_LEFT"] = PERS_TARGET_LEFT
    st.session_state.globalParameters["PERS_TARGET_RIGHT"] = PERS_TARGET_RIGHT
    st.session_state.globalParameters["XSHIFT"] = st.session_state.XSHIFT
    st.session_state.joined_img = joined_img
    
    with st.sidebar:
        with st.expander("Image combination",expanded=True):
            st.metric("XSHIFT", st.session_state.globalParameters["XSHIFT"])

    cols = st.columns([1.5,1,1])
    
    with cols[1]:
        st.button("🔙 I want to try another pair of photos", key="restarted_btn")
    
    with cols[2]:
        st.button("🎨 Save and go to color classification", key="save_and_leave")