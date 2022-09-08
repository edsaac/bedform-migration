from dataclasses import dataclass
import streamlit as st
from tempfile import NamedTemporaryFile
from PIL import Image
import os 

import numpy as np
import matplotlib.pyplot as plt

print(" --------------------- ")

oneimg_kw = dict(figsize=[14,14/1.5])

PERS_ORIGIN_LEFT  = {'x':[0,1162,1162,0]  ,'y':[533,532,99,87]}
PERS_ORIGIN_RIGHT = {'x':[84,1200,1200,84],'y':[495,500,64,49]}
PERS_TARGET_LEFT  = {'x':[0,1162,1162,0]  ,'y':[515,515,75,75]}
PERS_TARGET_RIGHT = {'x':[84,1200,1200,84],'y':[515,515,75,75]}
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
        os.system(f"""convert {in_img.name} -virtual-pixel black -distort Barrel "{a:.4f} {b:.4f} {c:.4f}" {out_img.name}""")
       
        return Image.open(out_img.name)

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
        os.system(f"""convert {in_img.name} -matte -virtual-pixel black -distort Perspective '{listCoords}' {out_img.name}""")
    
        return Image.open(out_img.name)

def get_timestamp(img):
    return img.getexif().get(306)

def show_imgs(imgs):
    
    shared_kw = dict(figsize=[18,18/1.5/2],sharex=True,sharey=True,gridspec_kw=dict(hspace=0,wspace=0.01))
    fig,axs = plt.subplots(1,2,**shared_kw)
    
    for ax,img in zip(axs, imgs):
        ax.imshow(img)
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
    raw_imgs = (Image.open(leftBytes), Image.open(rightBytes))

    ## Read their datestamp
    for img in raw_imgs:
        st.write(get_timestamp(img))

    # Plot them side by side
    st.pyplot(show_imgs(raw_imgs))

    ## Barrel correction
    barr_imgs = (fixBarrelDistortion(img) for img in raw_imgs)

    ## Plot them side by side
    st.pyplot(show_imgs(barr_imgs))

    ## Perspective correction
    
    pers_imgs = (fixPerspective(barr_imgs[0],PERS_ORIGIN_LEFT,PERS_TARGET_LEFT),
        fixPerspective(barr_imgs[1],PERS_ORIGIN_RIGHT,PERS_TARGET_RIGHT))

    st.pyplot(pers_imgs.plotly_fig())

    CROP_DIMENSIONS = (0, 250, 1200, 500)

    crop_imgs = ImagePair(
        pers_imgs.left.crop((CROP_DIMENSIONS)),
        pers_imgs.right.crop((CROP_DIMENSIONS))
        ) 
    
    st.pyplot(crop_imgs.plotly_fig())

    ## Equalize

    np_imgs = {k:np.asarray(v.convert('L')) for k,v in crop_imgs.items()}
equa_imgs = {k:np.interp(v, (v.min(), v.max()), (0, 255)) for k,v in np_imgs.items()}
inte_imgs = {k:Image.fromarray(v.astype(np.uint8),"L") for k,v in equa_imgs.items()}

fig,axs = plt.subplots(1,2,**shared_kw)
axs[0].imshow(inte_imgs['left'],cmap='Greys_r')
axs[1].imshow(inte_imgs['right'],cmap='Greys_r')
for ax in axs: ax.grid(which='both',lw=0.5,color='w')
plt.show()



