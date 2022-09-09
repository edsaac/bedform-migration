import streamlit as st
from tempfile import NamedTemporaryFile

from PIL import Image
import subprocess

import numpy as np
import matplotlib.pyplot as plt

print(" --------------------- ")

st.set_page_config(
    page_title = None, 
    page_icon  = None,
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

BARREL_CORRECTIONS = (0.000, -0.015, 0.000)

def fixBarrelDistortion(img, params = BARREL_CORRECTIONS):
    
    a,b,c = params

    with (
        NamedTemporaryFile(prefix="in",  suffix=".jpg") as in_img,
        NamedTemporaryFile(prefix="out", suffix=".jpg") as out_img
        ):

        ## Writes the input photo to a temporal file
        img.save(in_img.name)

        ## Execute ImageMagick
        ms = subprocess.run(
            [
                "convert", 
                in_img.name, 
                "-virtual-pixel", 
                "black", 
                "-distort", 
                "Barrel",
                f"{a:.4f} {b:.4f} {c:.4f}",
                out_img.name
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        st.write(ms)
       
        return Image.open(out_img.name)


PERS_ORIGIN_LEFT  = {'x':[0,1162,1162,0]  ,'y':[533,532,99,87]}
PERS_ORIGIN_RIGHT = {'x':[84,1200,1200,84],'y':[495,500,64,49]}
PERS_TARGET_LEFT  = {'x':[0,1162,1162,0]  ,'y':[515,515,75,75]}
PERS_TARGET_RIGHT = {'x':[84,1200,1200,84],'y':[515,515,75,75]}

def fixPerspective(img,pointsFrom,pointsTarget):
    
    originPairs = list(zip(pointsFrom['x'],pointsFrom['y']))
    targetPairs = list(zip(pointsTarget['x'],pointsTarget['y']))

    with (
        NamedTemporaryFile(prefix="in",  suffix=".jpg") as in_img,
        NamedTemporaryFile(prefix="out", suffix=".jpg") as out_img
        ):
    
        img.save(in_img.name)

        listCoords = ""
        for o,t in zip(originPairs,targetPairs):
            listCoords += f"{o[0]},{o[1]} {t[0]},{t[1]}  "

        ## Execute ImageMagick
        ms = subprocess.run([
            "convert",
            in_img.name,
            "-matte",
            "-virtual-pixel",
            "black",
            "-distort",
            "Perspective",
            f'{listCoords}',
            out_img.name
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        st.write(ms)

        return Image.open(out_img.name)

def get_timestamp(img):
    return img.getexif().get(306)

def show_imgs(imgs,imshow_kwargs={}):
    
    shared_kw = dict(figsize=[18,18/1.5/2],sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0.01))
    fig,axs = plt.subplots(1,2,**shared_kw)
    
    for ax,img in zip(axs, imgs):
        ax.imshow(img,**imshow_kwargs)
        ax.grid()
    
    return fig

"""
# Combine a pair of images to extract the beg geometry
"""

"""
We will use ImageMagick. You can recreate this on your computer

if on Debian/Ubuntu: apt install imagemagick
if on Windows: ???
if on iOS: don't
"""

col1, col2 = st.columns(2)

with col1:
    leftBytes = st.file_uploader("Left photo","JPG",False,key="leftPic")

with col2:
    rightBytes = st.file_uploader("Right photo","JPG",False,key="rightPic")

if leftBytes and rightBytes:
    
    ## Open images
    raw_imgs = [Image.open(leftBytes), Image.open(rightBytes)]

    """
    ## Raw images
    """
    for img in raw_imgs: st.write(get_timestamp(img))
    st.pyplot(show_imgs(raw_imgs))

    """
    ## Distortions
         
    ### Barrel correction
    """
    barr_imgs = [fixBarrelDistortion(img) for img in raw_imgs]
    st.pyplot(show_imgs(barr_imgs))

    """ 
    ### Perspective correction
    """
    pers_imgs = [fixPerspective(img, orig, target) for img, orig, target 
        in zip(
            barr_imgs,
            [PERS_ORIGIN_LEFT, PERS_ORIGIN_RIGHT],
            [PERS_TARGET_LEFT, PERS_TARGET_RIGHT])]
    
    st.pyplot(show_imgs(pers_imgs))

    """ 
    ## Miscelaneous

    ### Crop pictures
    """
    crop_imgs = [img.crop((0, 250, img.size[0], 500)) for img in pers_imgs]
    st.pyplot(show_imgs(crop_imgs))

    """ 
    ### Equalize pictures
    """
    st.header("Equalize pictures")
    np_imgs = [np.asarray(img.convert('L')) for img in crop_imgs]
    equa_imgs = [np.interp(img, (img.min(), img.max()), (0, 255)) for img in np_imgs]
    inte_imgs = [Image.fromarray(img.astype(np.uint8),"L") for img in equa_imgs]

    st.pyplot(show_imgs(inte_imgs,imshow_kwargs={'cmap':'Greys_r'}))

    ## Overlap
    st.header("Overlap")
    XSHIFT = 1162 - 84  # Overlap between pictures
    x, y = np.meshgrid(np.arange(inte_imgs[0].width),np.arange(inte_imgs[0].height,0,-1))
    
    fig,ax= plt.subplots(1,1,figsize=[20,6])
    ax.pcolormesh(x+XSHIFT,y,equa_imgs[1],cmap='Greys_r',shading='nearest',zorder=1)
    ax.pcolormesh(x,y,equa_imgs[0],cmap='Greys_r',shading='nearest',alpha=1,zorder=2)
    ax.set_aspect('equal')
    st.pyplot(fig)

    ## Save merged
    (width1, height1) = inte_imgs[0].size
    (width2, height2) = inte_imgs[1].size

    result_width = width1 + XSHIFT
    result_height = max(height1, height2)

    joined_img = Image.new('L', (result_width, result_height))
    joined_img.paste(im=inte_imgs[1], box=(XSHIFT, 0))
    joined_img.paste(im=inte_imgs[0], box=(0, 0))
    
    st.header("Merged")
    st.image(joined_img)
    joined_img.save("joined_img.jpg")

    ## Finish
    if st.button("Save progress."):
        st.session_state.joined_img = joined_img
        st.balloons()
