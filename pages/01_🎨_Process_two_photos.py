import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.signal import savgol_filter, find_peaks

from st_lg17cam import *

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title = "[NU CEE440] Lab2 - Image processing", 
    page_icon  = "🎨",
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

title = st.container()
placeholder = st.empty()

###################################
## Session state management
###################################

if "globalParameters" not in st.session_state.keys():
    st.session_state.globalParameters = dict()
else:
    buildSidebar()

if 'tempImages' not in st.session_state.keys():
    st.session_state.tempImages = dict()

if "save_and_continue" in st.session_state.keys():
    if st.session_state.save_and_continue:
        switch_page("Process all photos")

if 'page' not in st.session_state.keys(): 
    st.session_state.page = -1

def prevPage(): st.session_state.page -= 1
def nextPage(): st.session_state.page += 1
def firstPage(): st.session_state.page = 0

if "restarted_btn" not in st.session_state.keys():
    st.session_state.restarted_btn = False

if st.session_state.restarted_btn:
    st.session_state.globalParameters = dict()
    st.session_state.page = -1
    del st.session_state.leftPic
    del st.session_state.rightPic

###################################
## Start Streamlit App
###################################

with title:
    """
    ## 🎨 Identify the bed sediment - water interface
    """

    with st.expander("Image processing flowchart",
                    expanded = (st.session_state.page == -1)):
        graph = generateProcessGraph()
        st.graphviz_chart(graph,use_container_width=True)

    with st.expander("In case you don't have photos, download these pictures to run the app!",
                    expanded = (st.session_state.page == -1)):

        col1, col2 = st.columns(2)

        with col1:
            with open("assets/2021/left/DSC_0980.JPG",'rb') as photo:
                st.image(photo.read())
                st.download_button("Download left pic", photo, "left.JPG")
        with col2: 
            with open("assets/2021/right/DSC_0980.JPG", 'rb') as photo:
                st.image(photo.read())
                st.download_button("Download right pic", photo, "right.JPG")

    with st.expander("▶️ Upload a pair of photos",
                    expanded = (st.session_state.page == -1)):
        
        with st.form(key="filesForm", clear_on_submit= True):

            col1, col2 = st.columns(2)
            with col1: 
                st.file_uploader("Left photo","JPG",False,key="leftPic")
            with col2: 
                st.file_uploader("Right photo","JPG",False,key="rightPic")
            st.form_submit_button("Process!", on_click=nextPage)

###################################
## Start processing
###################################

if ("leftPic" in st.session_state.keys()) \
    and ("rightPic" in st.session_state.keys()):
    
    if st.session_state.leftPic \
        and st.session_state.rightPic \
        and not st.session_state.restarted_btn:

        ############# Page 0 ###################
        # Read the photos as PIL Images
        ########################################
        if st.session_state.page == 0:

            leftBytes = st.session_state.leftPic
            rightBytes = st.session_state.rightPic

            with placeholder.container():
                """
                *****
                # 0️⃣ Uploaded images
                """
                
                with st.expander("🔵",expanded=True):
                    st.warning(
                        """
                        The timestamp of both pictures **should not 
                        differ** by more than a few seconds.
                        """,
                        icon="📷")

                    raw_imgs = [Image.open(leftBytes), Image.open(rightBytes)]
                    fig = show_two_imgs(raw_imgs,addTimestamp=True)
                    st.pyplot(fig, transparent=True)

                    st.session_state.tempImages["raw"] = raw_imgs
                    st.button("➡️ Go to next step", on_click=nextPage)

        ############# Page 1 ###################
        # Correct distortions with ImageMagick
        ########################################
        elif st.session_state.page == 1:
            
            with placeholder.container():
                """
                *****
                # 1️⃣ Correct distortions
                """
                with st.expander("🔵  Barrel correction:", expanded=True):
                    
                    with st.echo():
                        ## Parameters used for distortion
                        BARREL_CORRECTIONS = (0.000, -0.015, 0.000)
            
                    barr_imgs = [fixBarrelDistortion(img,BARREL_CORRECTIONS) for img in st.session_state.tempImages["raw"]]
                    fig = show_two_imgs(barr_imgs)
                    st.pyplot(fig, transparent=True)

                with st.expander("🔵  4-point perspective correction:", expanded=True):
            
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

                st.session_state.globalParameters["BARREL_CORRECTIONS"] = BARREL_CORRECTIONS
                st.session_state.globalParameters["PERS_ORIGIN_LEFT"] = PERS_ORIGIN_LEFT
                st.session_state.globalParameters["PERS_ORIGIN_RIGHT"] = PERS_ORIGIN_RIGHT
                st.session_state.globalParameters["PERS_TARGET_LEFT"] = PERS_TARGET_LEFT
                st.session_state.globalParameters["PERS_TARGET_RIGHT"] = PERS_TARGET_RIGHT

                st.session_state.tempImages["pers"] = pers_imgs
                st.button("➡️ Go to next step", on_click=nextPage)

        ############# Page 2 ###################
        # Crop and equalize using PIL
        ########################################
        elif st.session_state.page == 2:
            
            with placeholder.container():
                """
                *****
                # 2️⃣ Additional edits
                """
                CROP_RANGE = st.slider(
                    "Cropping range",
                    min_value = 0,
                    max_value = st.session_state.tempImages["pers"][0].height,
                    value= (250,500),
                    step = 1,
                    key = "CROP_RANGE")

                with st.expander("🔵 Crop top and bottom:",expanded=True):
                    crop_imgs = [img.crop((0, CROP_RANGE[0], img.size[0], CROP_RANGE[1])) for img in st.session_state.tempImages["pers"]]
                    fig = show_two_imgs(crop_imgs)
                    st.pyplot(fig, transparent=True)

                with st.expander("🔵 Equalize",expanded=True):
                    np_imgs = [np.asarray(img.convert('L')) for img in crop_imgs]
                    equa_imgs = [np.interp(img, (img.min(), img.max()), (0, 255)) for img in np_imgs]
                    inte_imgs = [Image.fromarray(img.astype(np.uint8),"L") for img in equa_imgs]
                    
                    fig = show_two_imgs(inte_imgs,imshow_kwargs={'cmap':'Greys_r'})
                    st.pyplot(fig, transparent=True)

                st.session_state.globalParameters["CROP_RANGE"] = CROP_RANGE
                st.session_state.tempImages["equa"] = equa_imgs
                st.session_state.tempImages["inte"] = inte_imgs
                st.button("➡️ Go to next step", on_click=nextPage)

        ############# Page 3 ##############
        elif st.session_state.page == 3:
            
            with placeholder.container():
                """
                *****
                # 3️⃣ Overlap and merge
                """
        
                with st.expander("Overlap",expanded=True):
            
                    cols = st.columns([2,1])
                    
                    with cols[0]:
                        st.info("In order to merge both images in one, find where they should overlap.", icon="✴️")
            
                    with cols[1]:
                        XSHIFT = st.number_input(
                            "Where in the first picture does the second begin?", 
                            min_value = 500, 
                            max_value = 1200, 
                            value = 1078, 
                            step = 1,
                            key="XSHIFT")
            
                    x, y = np.meshgrid(
                        np.arange(st.session_state.tempImages["inte"][0].width),
                        np.arange(st.session_state.tempImages["inte"][0].height,
                        0, -1))
            
                    fig, ax = plt.subplots(1,1,figsize=[20,6])
            
                    # Overlap region
                    ax.add_patch(
                        mpatches.Rectangle(
                            (XSHIFT,0), 
                            st.session_state.tempImages["inte"][0].width - XSHIFT, 
                            st.session_state.tempImages["inte"][0].height + st.session_state.tempImages["inte"][1].height + 20, 
                            ec="purple",lw=3,fill=False,zorder=3))
                    
                    # Right picture
                    ax.pcolormesh(
                        x + XSHIFT,
                        y,
                        st.session_state.tempImages["equa"][1],
                        cmap='Greys_r', shading='nearest', zorder=1)

                    # Left picture
                    ax.pcolormesh(
                        x, 
                        y + st.session_state.tempImages["inte"][0].height + 20, 
                        st.session_state.tempImages["equa"][0],
                        cmap='Greys_r',shading='nearest',alpha=1,zorder=2)
                    
                    ax.set_aspect('equal')
                    st.pyplot(fig,transparent=True)

                    # Show merged picture
                    (width1, height1) = st.session_state.tempImages["inte"][0].size
                    (width2, height2) = st.session_state.tempImages["inte"][1].size

                    result_width = width1 + XSHIFT
                    result_height = max(height1, height2)

                    joined_img = Image.new('L', (result_width, result_height))
                    joined_img.paste(im=st.session_state.tempImages["inte"][1], box=(XSHIFT, 0))
                    joined_img.paste(im=st.session_state.tempImages["inte"][0], box=(0, 0))
            
                    st.image(joined_img,caption="Merged pictures")

                    st.session_state.globalParameters["XSHIFT"] = XSHIFT
                    st.session_state.tempImages["join"] = joined_img
                    st.button("➡️ Go to next step", on_click=nextPage)
                        
        ############# Page 4 ##############
        elif st.session_state.page == 4:
        
            with placeholder.container():
                """
                *****
                # 4️⃣ Picture histogram & color classification
                """

                cols = st.columns(2)
    
                with cols[0]:
                    MASKING_THRESHOLD = st.slider("Masking threshold",0,254,90,1,key="MASKING_THRESHOLD")
                    
                    with st.expander("What does this mean?",expanded=True):
                        """
                        |Threshold|Interpretation|Classification|
                        |----|----|----|
                        |Below|Darker than|Sand|
                        |Above|Brighter than|Water|

                        *****
                        """
                        
                with cols[1]:
                    hits = st.session_state.tempImages["join"].histogram()

                    fig,ax = plt.subplots(figsize=[8,4])
                    ax.vlines(np.arange(len(hits)),np.ones_like(hits),hits,color='purple')
                    ax.axvline(x = MASKING_THRESHOLD)
                    ax.set(
                        ylim = [1,1.0E6],
                        yscale = 'log',
                        ylabel = "Pixel count", 
                        xlabel = "Pixel value   [ 0  = Black | 255 = White ]")
                    
                    st.pyplot(fig, transparent=True)

                masked = np.ma.masked_greater(st.session_state.tempImages["join"],MASKING_THRESHOLD)
                ycoord = np.ma.count_masked(masked,axis=0)
                masked[np.logical_not(masked.mask)] = 1

                fig,axs = plt.subplots(2,1,figsize=[20,8],sharex=True,gridspec_kw={'hspace':0.01})
                ax = axs[0]
                ax.imshow(masked,cmap='Greys_r')

                ax.text(40, 200, "Sand", 
                    fontsize=14, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9))
                
                ax.text(40, 10, "Water", 
                    fontsize=14, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9))
        
                ax = axs[1]
                ax.imshow(st.session_state.tempImages["join"],cmap='Greys_r')
                ax.plot(ycoord,c='orange',lw=2,label='Sediment - water interface')
                ax.legend()
        
                st.pyplot(fig,transparent=True)

                st.session_state.globalParameters["MASKING_THRESHOLD"] = MASKING_THRESHOLD
                st.session_state.tempImages["ycoord"] = ycoord
                st.button("➡️ Go to next step", on_click=nextPage)        

        ############# Page 5 ##############
        elif st.session_state.page == 5:
        
            with placeholder.container():

                """
                *****
                # 5️⃣ Filtering and smoothing
                """

                cols = st.columns(2)

                with cols[0]:
                    """
                    How is the original line smoothed out?

                    Using a [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter),
                    which sequentially fits a polynomial to adjacent points.

                    **Learn more:**
                    - [📎](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) `scipy.signal.savgol_filter` 
                    """
        
                with cols[1]:
                    WINDOW_LENGHT = st.slider(
                        "The length of the filter window (i.e., the number of coefficients)",
                        min_value = 1,
                        max_value = 500,
                        value = 171,
                        step = 1,
                        key="WINDOW_LENGHT")

                    POLYORDER = st.slider(
                        "Order of the polynomial used to fit the samples",
                        min_value = 1,
                        max_value = 5,
                        value = 4,
                        step = 1,
                        key="POLYORDER")

                    SAVGOL_FILTER_PARAMS = {
                        'window_length': WINDOW_LENGHT,
                        'polyorder': POLYORDER}
                    
                ysmoothed = savgol_filter(
                    st.session_state.tempImages["ycoord"],
                    **SAVGOL_FILTER_PARAMS)
                
                xtemp = np.arange(len(ysmoothed))

                fig,axs = plt.subplots(2,1,figsize=[20,10],sharex=True,
                    gridspec_kw = {
                        'height_ratios':[1,1.5],
                        'hspace' : 0.01
                        })

                ax = axs[0]

                ax.plot(xtemp,st.session_state.tempImages["ycoord"],
                    lw=3,c='yellow',label='Smoothed',zorder=2)
                
                ax.plot(xtemp,st.session_state.tempImages["ycoord"],
                    lw=2,c='orange',label='Original')
                ax.imshow(st.session_state.tempImages["join"],
                    cmap='Greys_r',zorder=1)
                ax.set(ylabel="Distance Z [px]")
                ax.legend(loc='lower right')

                ax = axs[1]
                ax.plot(xtemp,ysmoothed,
                    lw=3,c='orange',alpha=0.9,label='Smoothed',zorder=2)
                ax.plot(xtemp,st.session_state.tempImages["ycoord"],
                    lw=1,c='k',label='Original',zorder=1)
                ax.set(
                    xlabel="Distance X [px]",
                    ylabel="Distance Z [px]",
                    xlim=[0,st.session_state.tempImages["join"].width],
                    ylim=[st.session_state.tempImages["join"].height, 0])
                ax.legend()
                st.pyplot(fig, transparent=True)

                st.session_state.globalParameters["WINDOW_LENGHT"] = WINDOW_LENGHT
                st.session_state.globalParameters["POLYORDER"] = POLYORDER
                st.session_state.tempImages["ysmoothed"] = ysmoothed

                st.button("➡️ Go to next step", on_click=nextPage)
                
        ############# Page 6 ##############
        elif st.session_state.page == 6:
        
            with placeholder.container():

                """
                *****

                # 6️⃣ Peaks and troughs identification
                """

                cols = st.columns(2)

                with cols[0]:
                    """
                    **How to systematically identify the peaks and troughs?**
                    
                    A peak or trough is located wherever the slope of the curve is zero.
                    For a signal, it suffices to compare each point to its neighbours.

                    **References & docs:**
                    - [📎](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html) `scipy.signal.find_peaks` 
                    """
        
                with cols[1]:
                    MINIMAL_DISTANCE = st.slider(
                        "Minimal distance between peaks/troughs",
                        min_value = 1,
                        max_value = 200,
                        value = 100,
                        step = 1,
                        key="MINIMAL_DISTANCE")
                    
                    PROMINENCE = st.slider(
                        "Prominence of peaks/troughs",
                        min_value = 0.1,
                        max_value = 20.0,
                        value = 10.0,
                        step = 0.1, 
                        key="PROMINENCE")
    
                TROUGH_FINDER_PARAMS = dict(distance=MINIMAL_DISTANCE,height=15,prominence=PROMINENCE)
                PEAK_FINDER_PARAMS   = dict(distance=MINIMAL_DISTANCE,prominence=PROMINENCE)

                whereTroughs,_ = find_peaks(
                    st.session_state.tempImages["ysmoothed"],
                    **TROUGH_FINDER_PARAMS)
                
                troughs_df = pd.DataFrame({
                    'X' : whereTroughs,
                    'Z' : [ st.session_state.tempImages["ysmoothed"][w] for w in whereTroughs ]
                    })
                
                troughs_df["Type"] = "Trough"
    
                wherePeaks,_ = find_peaks(
                    -st.session_state.tempImages["ysmoothed"],
                    **PEAK_FINDER_PARAMS)
                
                peaks_df = pd.DataFrame({
                    'X' : wherePeaks,
                    'Z' : [ st.session_state.tempImages["ysmoothed"][w] for w in wherePeaks ]
                    })
                    
                peaks_df["Type"] = "Peak"

                xtemp = np.arange(len(st.session_state.tempImages["ysmoothed"]))
                
                fig,axs = plt.subplots(2,1,figsize=[20,10],sharex=True,
                    gridspec_kw = {
                        'height_ratios':[1,1.5],
                        'hspace' : 0.01
                        })
                
                ax = axs[0]
                
                ax.plot(xtemp,st.session_state.tempImages["ysmoothed"],
                    lw=3,c='yellow',label='Smoothed',zorder=2)
                
                ax.scatter(troughs_df['X'],troughs_df['Z'],
                    c='r',s=100,zorder=3,label='Trough')
                
                ax.scatter(peaks_df['X'],peaks_df['Z'],
                    c='b',s=100,zorder=3,label='Peak')
                
                ax.imshow(st.session_state.tempImages["join"],
                    cmap='Greys_r',zorder=1)
                
                ax.set(
                    xlabel="Distance X [px]",
                    ylabel="Distance Z [px]")
                
                ax.legend(loc='lower right')

                ax = axs[1]
                
                ax.plot(xtemp,st.session_state.tempImages["ysmoothed"],
                    lw=3,c='orange',alpha=0.9,label='Smoothed',zorder=2)
                
                ax.scatter(troughs_df['X'],troughs_df['Z'],
                    c='r',s=100,zorder=3,label='Trough')
                
                ax.scatter(peaks_df['X'],peaks_df['Z'],
                    c='b',s=100,zorder=3,label='Peak')
                
                ax.set(
                    xlabel="Distance X [px]",
                    ylabel="Distance Z [px]",
                    xlim=[0,st.session_state.tempImages["join"].width],
                    ylim=[st.session_state.tempImages["join"].height, 0])

                ax.legend()
                st.pyplot(fig, transparent=True)

                st.session_state.globalParameters["MINIMAL_DISTANCE"] = MINIMAL_DISTANCE
                st.session_state.globalParameters["PROMINENCE"] = PROMINENCE
                
                st.session_state.tempImages["whereTroughs"] = whereTroughs
                st.session_state.tempImages["wherePeaks"] = wherePeaks
                st.session_state.tempImages["troughs_df"] = troughs_df
                st.session_state.tempImages["peaks_df"] = peaks_df
                st.button("➡️ Go to next step", on_click=nextPage)

        ############# Page 7 ##############
        elif st.session_state.page == 7:
        
            with placeholder.container():

                """
                *****
                # 7️⃣ Summary of info extracted from a pair pictures
                """

                endimg = Image.new('RGB', st.session_state.tempImages["join"].size)
                endimg.paste(im=st.session_state.tempImages["join"], box=(0, 0))

                lineimg = ImageDraw.Draw(endimg)
                lineimg.line(
                    list(
                        zip(
                            np.arange(st.session_state.tempImages["join"].width),
                            st.session_state.tempImages["ysmoothed"]
                            )
                        ),
                    fill ="orange", 
                    width = 7)

                r = 10
                for wt in st.session_state.tempImages["whereTroughs"]:
                    lineimg.ellipse([
                        (
                            wt - r,
                            st.session_state.tempImages["ysmoothed"][wt] - r
                        ),
                        (
                            wt + r,
                            st.session_state.tempImages["ysmoothed"][wt] + r
                        )],
                        outline = 'white', 
                        fill = 'red', 
                        width = 1)

                for wt in st.session_state.tempImages["wherePeaks"]:
                    lineimg.ellipse([
                        (
                            wt - r,
                            st.session_state.tempImages["ysmoothed"][wt]-r
                        ),
                        (
                            wt + r,
                            st.session_state.tempImages["ysmoothed"][wt]+r
                        )], 
                        outline = 'white', 
                        fill = 'blue', 
                        width = 1)
    
                """
                ### 🔵 The location of the sediment-bed interface
                """
                st.image(endimg, caption="Merged photo with sediment/water interface, peaks and troughs")
                
                """
                ### 🔵 A list of the peaks and troughs locations
                """
                st.dataframe(
                    pd.concat([
                        st.session_state.tempImages["troughs_df"],
                        st.session_state.tempImages["peaks_df"]]
                    ))

                cols = st.columns([2,1,1])

                with cols[0]:
                    st.info(
                        """
                        Check other pictures to make sure that the parameters you used for 
                        image classification and processing are adecuate for other pairs of photos. 
                        """, icon="🎞️")

                with cols[1]:
                    st.button("🔙 I want to try another pair of photos", key="restarted_btn", on_click=firstPage)

                with cols[2]:
                    st.button("🎥 I'm ready to process all my photos!", key="save_and_continue")

