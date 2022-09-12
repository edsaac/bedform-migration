import pickle
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from PIL import Image, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

from st_lg17cam import *

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title = "[NU CEE440] - Color classification", 
    page_icon  = "🎨",
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

"""
# 🎨 Color classification
"""
graph = generateProcessGraph(whichStep="identify")
st.graphviz_chart(graph,use_container_width=True)

###################################
## Session state management
###################################
if "restarted_btn2" in st.session_state.keys():
    if st.session_state.restarted_btn2:
        del st.session_state.uploadedFilesCheck
        switch_page("Combine a pair")

if "save_and_continue2" in st.session_state.keys():
    if st.session_state.save_and_continue2:
        switch_page("Process all photos")

if "joined_img" not in st.session_state.keys() \
    or "globalParameters" not in st.session_state.keys():
    
    "*******"
    cols = st.columns([3,1])
    joined_img = Image.open("./assets/250.jpg")
    
    with cols[0]:
        st.image(joined_img, caption="🙀 I should not be here")
    
    with cols[1]:
        st.error("It seems like you skipped something!", icon="⛔")        
        if st.button("🔙 Take me back"): switch_page("Combine a pair")

###################################
## Processing colors
###################################
else:
    joined_img = st.session_state.joined_img 
    
    with st.sidebar: 
        "#### Image processing parameters"
        with st.expander("Image combination",expanded=True):
            st.metric("XSHIFT", st.session_state.globalParameters["XSHIFT"])

    cols = st.columns([3,1])
    with cols[0]:
        st.image(joined_img, caption="Combined photo that will be processed here")
    
    with cols[1]:
        if st.button("🔙 Go back"):
            switch_page("Combine a pair")

    """
    *****
    ## 1️⃣ Picture histogram & color classification
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

        hits = joined_img.histogram()

        fig,ax = plt.subplots(figsize=[8,4])
        ax.vlines(np.arange(len(hits)),np.ones_like(hits),hits,color='purple')
        ax.axvline(x = MASKING_THRESHOLD)
        ax.set(
            ylim = [1,1.0E6],
            yscale = 'log',
            ylabel = "Pixel count", 
            xlabel = "Pixel value   [ 0  = Black | 255 = White ]")
        
        st.pyplot(fig, transparent=True)

    masked = np.ma.masked_greater(joined_img,MASKING_THRESHOLD)
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
    ax.imshow(joined_img,cmap='Greys_r')
    ax.plot(ycoord,c='orange',lw=2,label='Sediment - water interface')
    ax.legend()
    
    st.pyplot(fig,transparent=True)

    """
    *****
    ## 2️⃣ Filtering and smoothing
    """

    cols = st.columns(2)

    with cols[0]:
        """
        How is the original line smoothed out?
        Using a [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter),
        which sequentially fits a polynomial to adjacent points.

        **References & docs:**
        - [📎](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) `scipy.signal.savgol_filter` 
        """
        
    with cols[1]:
        WINDOW_LENGHT = st.slider("The length of the filter window (i.e., the number of coefficients)",1,500,171,1,key="WINDOW_LENGHT")
        POLYORDER = st.slider("Order of the polynomial used to fit the samples",1,5,4,1,key="POLYORDER")

        SAVGOL_FILTER_PARAMS = {
            'window_length': WINDOW_LENGHT,
            'polyorder': POLYORDER}
        
        ysmoothed = savgol_filter(ycoord,**SAVGOL_FILTER_PARAMS)
        xtemp = np.arange(len(ysmoothed))

    fig,axs = plt.subplots(2,1,figsize=[20,10],sharex=True,
        gridspec_kw = {
            'height_ratios':[1,1.5],
            'hspace' : 0.01
            })

    ax = axs[0]
    ax.plot(xtemp,ysmoothed,
        lw=3,c='yellow',label='Smoothed',zorder=2)
    ax.plot(xtemp,ycoord,
        lw=2,c='orange',label='Original')
    ax.imshow(joined_img,
        cmap='Greys_r',zorder=1)
    ax.set(ylabel="Distance Z [px]")
    ax.legend(loc='lower right')

    ax = axs[1]
    ax.plot(xtemp,ysmoothed,
        lw=3,c='orange',alpha=0.9,label='Smoothed',zorder=2)
    ax.plot(xtemp,ycoord,
        lw=1,c='k',label='Original',zorder=1)
    ax.set(
        xlabel="Distance X [px]",
        ylabel="Distance Z [px]",
        ylim=[200,70],
        xlim=[0,joined_img.width])
    ax.legend()
    st.pyplot(fig, transparent=True)

    """
    *****

    ## 3️⃣ Peaks and troughs identification
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
        MINIMAL_DISTANCE = st.slider("Minimal distance between peaks/troughs", 1,200,100,1, key="MINIMAL_DISTANCE")
        PROMINENCE = st.slider("Prominence of peaks/troughs", 0.1,20.0,10.0,0.1, key="PROMINENCE")
    
    TROUGH_FINDER_PARAMS = dict(distance=MINIMAL_DISTANCE,height=15,prominence=PROMINENCE)
    PEAK_FINDER_PARAMS   = dict(distance=MINIMAL_DISTANCE,prominence=PROMINENCE)

    whereTroughs,_ = find_peaks(ysmoothed,**TROUGH_FINDER_PARAMS)
    Troughs_df = pd.DataFrame({'X':whereTroughs,'Z':[ysmoothed[w] for w in whereTroughs]})
    Troughs_df["Type"] = "Trough"
    
    wherePeaks,_ = find_peaks(-ysmoothed,**PEAK_FINDER_PARAMS)
    Peaks_df = pd.DataFrame({'X':wherePeaks,'Z':[ysmoothed[w] for w in wherePeaks]})
    Peaks_df["Type"] = "Peak"

    fig,axs = plt.subplots(2,1,figsize=[20,10],sharex=True,
        gridspec_kw = {
            'height_ratios':[1,1.5],
            'hspace' : 0.01
            })
    
    ax = axs[0]
    ax.plot(xtemp,ysmoothed,
        lw=3,c='yellow',label='Smoothed',zorder=2)
    ax.scatter(Troughs_df['X'],Troughs_df['Z'],
        c='r',s=100,zorder=3,label='Trough')
    ax.scatter(Peaks_df['X'],Peaks_df['Z'],
        c='b',s=100,zorder=3,label='Peak')
    ax.imshow(joined_img,
        cmap='Greys_r',zorder=1)
    ax.set(
        xlabel="Distance X [px]",
        ylabel="Distance Z [px]")
    ax.legend(loc='lower right')

    ax = axs[1]
    ax.plot(xtemp,ysmoothed,
        lw=3,c='orange',alpha=0.9,label='Smoothed',zorder=2)
    ax.scatter(Troughs_df['X'],Troughs_df['Z'],
        c='r',s=100,zorder=3,label='Trough')
    ax.scatter(Peaks_df['X'],Peaks_df['Z'],
        c='b',s=100,zorder=3,label='Peak')
    ax.set(
        xlabel="Distance X [px]",
        ylabel="Distance Z [px]",
        ylim=[200,60],
        xlim=[0,joined_img.width])
    ax.legend()
    st.pyplot(fig, transparent=True)
    
    """
    *****
    ## 4️⃣ Summary of info extracted from a pair pictures
    """

    endimg = Image.new('RGB', joined_img.size)
    endimg.paste(im=joined_img, box=(0, 0))

    lineimg = ImageDraw.Draw(endimg)
    lineimg.line(list(zip(np.arange(joined_img.size[0]),ysmoothed)), fill ="orange", width = 7)

    r = 10
    for wt in whereTroughs:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)],
                        outline = 'white', fill = 'red', width = 1)

    for wt in wherePeaks:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)], 
                        outline = 'white', fill = 'blue', width = 1)
    """
    ### 🔵 The location of the sediment-bed interface
    """
    st.image(endimg, caption="Merged photo with sediment/water interface, peaks and troughs")
    
    """
    ### 🔵 A list of the peaks and troughs locations
    """
    st.dataframe(pd.concat([Troughs_df,Peaks_df]))

    cols = st.columns([2,1,1])

    with cols[0]:
        st.info(
            """
            Check other pictures to check that the parameters you used for image classification 
            and processing are adecuate for other pairs of pictures. 
            """, icon="🎞️")

    
    with cols[1]:
        st.button("🔙 I want to try another pair of photos", key="restarted_btn2")

    with cols[2]:
        st.button("🎥 I'm ready to process all my photos!", key="save_and_continue2")
            
    ## Save globalParameters configuration
    st.session_state.globalParameters["MASKING_THRESHOLD"] = st.session_state.MASKING_THRESHOLD
    st.session_state.globalParameters["WINDOW_LENGHT"] = st.session_state.WINDOW_LENGHT
    st.session_state.globalParameters["POLYORDER"] = st.session_state.POLYORDER
    st.session_state.globalParameters["MINIMAL_DISTANCE"] = st.session_state.MINIMAL_DISTANCE
    st.session_state.globalParameters["PROMINENCE"] = st.session_state.PROMINENCE
    
    with st.sidebar: 
        with st.expander("Color classification:",expanded=True):
            st.metric("MASKING_THRESHOLD", st.session_state.globalParameters["MASKING_THRESHOLD"])
        
        with st.expander("Smoothing and filtering:",expanded=True):
            cols = st.columns(2)
            with cols[0]: st.metric("WINDOW_LENGHT", st.session_state.globalParameters["WINDOW_LENGHT"])
            with cols[1]: st.metric("POLYORDER", st.session_state.globalParameters["POLYORDER"])
        
        with st.expander("Peak identification:",expanded=True):
            cols = st.columns(2)
            with cols[0]: st.metric("MINIMAL_DISTANCE", st.session_state.globalParameters["MINIMAL_DISTANCE"])
            with cols[1]: st.metric("PROMINENCE", st.session_state.globalParameters["PROMINENCE"])

    # with open("assets/globalParameters.pkl",'wb') as f:
    #     pickle.dump(st.session_state.globalParameters,f)
    
        
