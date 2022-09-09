import streamlit as st
from tempfile import NamedTemporaryFile

from PIL import Image, ImageDraw
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

st.set_page_config(
    page_title = None, 
    page_icon  = None,
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

"""
# Color classification
"""

if "joined_img" not in st.session_state.keys():
    st.error("Seems like you didn't finish!")
    joined_img = Image.open("./assets/250.jpg")
    st.image(joined_img)
else:
    joined_img = st.session_state.joined_img 
    st.image(joined_img)
    
    """
    ## Picture histogram
    """

    MASKING_THRESHOLD = 90

    hits = joined_img.histogram()

    fig,axs = plt.subplots(1,1,sharex=True,sharey=True,gridspec_kw={'wspace':0.01})
    axs.vlines(np.arange(len(hits)),np.ones_like(hits),hits,color='grey')
    axs.axvline(x=MASKING_THRESHOLD)
    axs.set(ylim=[1,1.0E6],yscale='log')
    axs.set(ylabel = "Pixel count", xlabel="Pixel value   [ 0  = Black | 255 = White ]")
    st.pyplot(fig)

    masked = np.ma.masked_greater(joined_img,MASKING_THRESHOLD)
    ycoord = np.ma.count_masked(masked,axis=0)
    masked[np.logical_not(masked.mask)] = 1

    fig,axs = plt.subplots(3,1,figsize=[20,8],sharex=True,gridspec_kw={'hspace':0.01})
    axs[0].imshow(joined_img,alpha=1.0)
    axs[1].imshow(masked,cmap='Greys_r')
    axs[2].imshow(joined_img,cmap='Greys_r')
    axs[2].plot(ycoord,c='orange',lw=5)
    st.pyplot(fig)

    """
    ## Filtering and smoothing
    """

    fig,ax = plt.subplots(figsize=[20,8])
    ax.plot(np.arange(len(ycoord)),ycoord,lw=1,c='orange',label='PixelCount')
    ax.set(xlabel="Distance X [px]",ylabel="Distance Z [px]")
    ax.imshow(joined_img,cmap='Greys_r')
    ax.legend()
    st.pyplot(fig)

    SAVGOL_FILTER_PARAMS = dict(window_length=171,polyorder=4)
    ysmoothed = savgol_filter(ycoord,**SAVGOL_FILTER_PARAMS)

    fig,ax = plt.subplots(figsize=[20,8])
    xtemp = np.arange(len(ysmoothed))
    ax.plot(xtemp,ysmoothed,lw=3,c='yellow',label='Filtered',zorder=2)
    ax.plot(xtemp,ycoord,lw=2,c='orange',label='PixelCount')
    ax.imshow(joined_img,cmap='Greys_r',zorder=1)
    ax.set(xlabel="Distance X [px]",ylabel="Distance Z [px]")
    ax.legend(loc='lower right')
    st.pyplot(fig)

    fig,ax = plt.subplots(figsize=[20,5])
    xtemp = np.arange(len(ysmoothed))
    ax.plot(xtemp,ysmoothed,lw=3,c='orange',alpha=0.9,label='Filtered',zorder=2)
    ax.plot(xtemp,ycoord,lw=1,c='k',label='PixelCount',zorder=1)
    ax.set(xlabel="Distance X [px]",ylabel="Distance Z [px]",ylim=[200,70],xlim=[0,joined_img.width])
    ax.legend()
    st.pyplot(fig)

    """
    ## Peaks and through identification
    """

    TROUGH_FINDER_PARAMS = dict(distance=100,height=15,prominence=10)
    PEAK_FINDER_PARAMS   = dict(distance=100,prominence=10)

    whereTroughs,_ = find_peaks(ysmoothed,**TROUGH_FINDER_PARAMS)
    Troughs_df = pd.DataFrame({'X':whereTroughs,'ZT':[ysmoothed[w] for w in whereTroughs]})
    Troughs_df.set_index('X',inplace=True)

    wherePeaks,_ = find_peaks(-ysmoothed,**PEAK_FINDER_PARAMS)
    Peaks_df = pd.DataFrame({'X':wherePeaks,'ZP':[ysmoothed[w] for w in wherePeaks]})
    Peaks_df.set_index('X',inplace=True)

    fig,ax = plt.subplots(figsize=[20,8])
    ax.plot(xtemp,ysmoothed,lw=3,c='yellow',label='Filtered',zorder=2)
    ax.scatter(Troughs_df.index,Troughs_df['ZT'],c='r',s=100,zorder=3,label='Trough')
    ax.scatter(Peaks_df.index,Peaks_df['ZP'],c='b',s=100,zorder=3,label='Peak')
    ax.imshow(joined_img,cmap='Greys_r',zorder=1)
    ax.set(xlabel="Distance X [px]",ylabel="Distance Z [px]")
    ax.legend(loc='lower right')
    st.pyplot(fig)

    """
    ## Info extracted from pictures
    """
    fig,ax = plt.subplots(1,1,figsize=[20,5])
    ax.plot(xtemp,ysmoothed,lw=3,c='orange',alpha=0.9,label='Filtered',zorder=2)
    ax.scatter(Troughs_df.index,Troughs_df['ZT'],c='r',s=100,zorder=3,label='Trough')
    ax.scatter(Peaks_df.index,Peaks_df['ZP'],c='b',s=100,zorder=3,label='Peak')
    ax.set(xlabel="Distance X [px]",ylabel="Distance Z [px]",ylim=[200,60],xlim=[0,joined_img.width])
    ax.legend()
    st.pyplot(fig)
    
    """
    ## Final result
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

    st.image(endimg)
    st.dataframe(Troughs_df)
    st.dataframe(Peaks_df)