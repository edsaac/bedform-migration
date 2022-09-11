from functools import cache
from PIL import Image, ImageDraw
import subprocess
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import pandas as pd 
import streamlit as st 
import io
import numpy as np 
import graphviz 

@st.experimental_singleton
def generateProcessGraph(whichStep="merge"):

    parent = graphviz.Digraph(name="Parent")
    g = graphviz.Digraph(name="cluster_Combine",comment="Combine")
    h = graphviz.Digraph(name="cluster_Identify")
    
    ## Formatting and styling
    parent.graph_attr["rank"] = "same"
    parent.graph_attr['rankdir'] = 'LR'
    parent.graph_attr['bgcolor'] = '#F8F8FF'
    
    if whichStep == "merge":
        g.graph_attr["rank"] = "same"

        g.graph_attr["label"] = 'What we will do in this step'
        g.graph_attr["style"] = 'filled'
        g.graph_attr["color"] = 'lightgrey'

        g.node_attr["style"] = "filled"
        g.node_attr["fillcolor"] = "lightskyblue"
        g.node_attr["color"] = "navy"

        h.graph_attr["style"] = 'invis'

        h.node_attr["style"] = "filled, dashed"
        h.node_attr["fillcolor"] = "honeydew"
        h.node_attr["color"] = "honeydew4"

    elif whichStep == "identify":
        h.graph_attr["rank"] = "same"

        h.graph_attr["label"] = 'What we will do in this step'
        h.graph_attr["style"] = 'filled'
        h.graph_attr["color"] = 'lightgrey'
        
        h.node_attr["style"] = "filled"
        h.node_attr["fillcolor"] = "lightskyblue"
        h.node_attr["color"] = "navy"

        g.graph_attr["style"] = 'invis'
        g.node_attr["style"] = "filled, dashed"
        g.node_attr["fillcolor"] = "honeydew"
        g.node_attr["color"] = "honeydew4"
    
    else: pass

    ## Topology
    g.node("left","Left-side\nphoto")
        # style = "filled", fillcolor="lightskyblue", shape = "egg", color = "navy")

    g.node("right","Right-side\nphoto")
    g.node("merged","Merged\nphoto")
    g.edge("left","merged")
    g.edge("right","merged")

    h.node("interface","Sediment-water\ninterface")
    h.node("peaks","Peaks\nlocation")
    h.node("troughs","Troughs\nlocation")
    # g.edge("merged","interface")
    h.edge("interface","peaks")
    h.edge("interface","troughs")

    parent.subgraph(h)
    parent.subgraph(g)
    parent.edge("merged","interface")

    return parent


def fixBarrelDistortion(img, params):
    '''
    Uses imagemagick to fix barrel/pincushion distortion
    
    Parameters
    ----------
    img : PIL Image
    params : (float,float,float)  three parameters to fix barrel distortion
    
    Returns
    -------
    PIL Image
        Distorted image

    Notes
    -----
    Check https://imagemagick.org/Usage/distorts/#barrel
    
    '''
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

        print(ms)
       
        return Image.open(out_img.name)


def fixPerspective(img,pointsFrom,pointsTarget):
    '''
    Uses imagemagick to get a 4-point perspective distortion
    
    Parameters
    ----------
    img : PIL Image
        Input image
    pointsFrom : dict
        {'x':[x1,x2,x3,x4],'y':[y1,y2,y3,y4]}
        Location of four points in the input image
    pointsTarget : dict
        {'x':[x1,x2,x3,x4],'y':[y1,y2,y3,y4]}
        Location of the four points in the target image
    
    Returns
    -------
    PIL Image
        Output image

    Notes
    -----
    Check https://imagemagick.org/Usage/distorts/#perspective
    
    '''
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

        print(ms)

        return Image.open(out_img.name)

def get_timestamp(img):
    '''
    Returns datetime of photo from its metadata

    Parameters
    ----------
    img : PIL Image
        Input image
    
    Returns
    -------
    str
        EXIF 306 tag DateTime

    Notes
    -----
    Check https://pillow.readthedocs.io/en/stable/reference/ExifTags.html
    
    '''
    return img.getexif().get(306)

def checkTimeStamps(timestamps):
    '''
    Checks that two timestamps are close 

    Parameters
    ----------
    imgs : (str, str)
        Tuple of strings from PIL.Image.getexif().get(306)
    
    Returns
    -------    
    None
    
    Raises
    ------
    st.error
        If the two timestamps are not close

    Notes
    -----
    
    
    '''

    timefmt = '%Y:%m:%d %H:%M:%S'
    dates = [pd.to_datetime(f,format=timefmt) for f in timestamps]
    deltat = abs(dates[0] - dates[1])

    if deltat > pd.to_timedelta("00:00:30"):
        st.error(
            f"""
            Looks like there are timestamps that do not coincide!
            {dates[0]} and {dates[1]}
            """)

def show_two_imgs(imgs,addTimestamp=False,imshow_kwargs={}):
    '''
    Plots the right and left photos side by side using plotly

    Parameters
    ----------
    imgs : (PIL.Image, PIL.Image)
        Tuple of input images

    imshow_kwargs : dict
        kwargs passed to imshow
    
    Returns
    -------
    plt.Figure
        Matplotlib figure

    Notes
    -----
    Check https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
    
    '''
    shared_kw = dict(
        figsize = [18,18/1.5/2],
        sharex = True,
        sharey = True,
        gridspec_kw = dict(
            hspace = 0,
            wspace = 0.01)
        )
    
    fig,axs = plt.subplots(1, 2,**shared_kw)
    
    for ax,img in zip(axs, imgs):
        ax.imshow(img,**imshow_kwargs)
        ax.grid()
    
        if addTimestamp:
            ax.set_title(get_timestamp(img))

    return fig

def getDatabasePics(uploadedFiles):
    '''
    Lists pictures and extracts their timestamps

    Parameters
    ----------
    imgs : list[st.UploadedFile]
        List of uploaded files
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe holding the filename and time of photo
    
    list[PIL.Image]
        An ordered list of the PIL Images.

    Notes
    -----
    
    '''
    
    nameImg = []
    dateImg = []
    pilImgs = []

    for f in uploadedFiles:
        img = Image.open(f)
        date = img.getexif().get(306)
        if isinstance(date, bytes): date = date.decode()
        
        nameImg.append(f.name)
        dateImg.append(date)
        pilImgs.append(img)
        
    db = pd.DataFrame({'File':nameImg, 'Time':dateImg, 'Imgs':pilImgs})
    db['Time'] = pd.to_datetime(db['Time'], format='%Y:%m:%d %H:%M:%S')
    db.sort_values(by=['Time'], inplace=True, ignore_index=True)
    
    return db[['Time','File']], db['Imgs']

@cache
def processPair(imgs: tuple, parameters: dict):
    '''
    Streamlines the processing pair process

    Parameters
    ----------
    imgs : (st.UploadedFile)
        Tuple of uploaded files
    
    parameters: dict()
        Dictionary with all the tweaking parameters

    Returns
    -------    
    PIL.Image
        A processed image.

    Notes
    -----
    
    
    '''

    from scipy.signal import savgol_filter, find_peaks

    BARREL_CORRECTIONS = parameters["BARREL_CORRECTIONS"]
    PERS_ORIGIN_LEFT = parameters["PERS_ORIGIN_LEFT"]
    PERS_ORIGIN_RIGHT = parameters["PERS_ORIGIN_RIGHT"]
    PERS_TARGET_LEFT = parameters["PERS_TARGET_LEFT"]
    PERS_TARGET_RIGHT = parameters["PERS_TARGET_RIGHT"]
    XSHIFT = parameters["XSHIFT"]
    MASKING_THRESHOLD = parameters["MASKING_THRESHOLD"]
    WINDOW_LENGHT = parameters["WINDOW_LENGHT"]
    POLYORDER = parameters["POLYORDER"]
    MINIMAL_DISTANCE = parameters["MINIMAL_DISTANCE"]
    PROMINENCE = parameters["PROMINENCE"]

    raw_imgs = [f for f in imgs]
    raw_imgs_timestamps = [get_timestamp(f) for f in raw_imgs] 
    checkTimeStamps(raw_imgs_timestamps)

    # Fix barrel distortion
    barr_imgs = [fixBarrelDistortion(img,BARREL_CORRECTIONS) for img in raw_imgs]
    
    # Fix perspective distortion
    pers_imgs = [fixPerspective(img, orig, target) for img, orig, target 
            in zip(
                barr_imgs,
                [PERS_ORIGIN_LEFT, PERS_ORIGIN_RIGHT],
                [PERS_TARGET_LEFT, PERS_TARGET_RIGHT])]
    
    # Crop to conserve only water/sediment
    crop_imgs = [img.crop((0, 250, img.size[0], 500)) for img in pers_imgs]

    # Normalize both pictures
    np_imgs = [np.asarray(img.convert('L')) for img in crop_imgs]
    equa_imgs = [np.interp(img, (img.min(), img.max()), (0, 255)) for img in np_imgs]
    inte_imgs = [Image.fromarray(img.astype(np.uint8),"L") for img in equa_imgs]
    
    # Overlap images
    (width1, height1) = inte_imgs[0].size
    (width2, height2) = inte_imgs[1].size

    result_width = width1 + XSHIFT
    result_height = max(height1, height2)

    joined_img = Image.new('L', (result_width, result_height))
    joined_img.paste(im=inte_imgs[1], box=(XSHIFT, 0))
    joined_img.paste(im=inte_imgs[0], box=(0, 0))
         
    # Color classification for edge detection
    masked = np.ma.masked_greater(joined_img,MASKING_THRESHOLD)
    ycoord = np.ma.count_masked(masked,axis=0)

    # Filter and smooth the edge profile
    SAVGOL_FILTER_PARAMS = {
            'window_length': WINDOW_LENGHT,
            'polyorder': POLYORDER}
    
    ysmoothed = savgol_filter(ycoord,**SAVGOL_FILTER_PARAMS)

    # Locate peaks and throughs
    TROUGH_FINDER_PARAMS = dict(distance=MINIMAL_DISTANCE,height=15,prominence=PROMINENCE)
    PEAK_FINDER_PARAMS   = dict(distance=MINIMAL_DISTANCE,prominence=PROMINENCE)
    whereTroughs,_ = find_peaks(ysmoothed,**TROUGH_FINDER_PARAMS)
    wherePeaks,_   = find_peaks(-ysmoothed,**PEAK_FINDER_PARAMS)

    # Save the extracted peak/through data
    timestamp = raw_imgs_timestamps[0]
    troughs = {
        'Timestamp' : timestamp,
        'X(px)'     : np.array([round(w,ndigits=1) for w in whereTroughs]),
        'Z(px)'     : np.array([round(ysmoothed[w],ndigits=2) for w in whereTroughs])
    }

    peaks = {
        'Timestamp' : timestamp,
        'X(px)'     : np.array([round(w,ndigits=1) for w in wherePeaks]),
        'Z(px)'     : np.array([round(ysmoothed[w],ndigits=2) for w in wherePeaks])
    }

    ## Create Line over image
    endimg = Image.new('RGB', joined_img.size)
    endimg.paste(im=joined_img, box=(0, 0))

    lineimg = ImageDraw.Draw(endimg)
    lineimg.line(list(zip(np.arange(joined_img.size[0]),ysmoothed)),
                 fill ="orange", width = 7)

    ## Create Points over image
    r = 10 
    for wt in whereTroughs:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)],
                        outline = 'white', fill = 'red', width = 1)

    for wt in wherePeaks:
        lineimg.ellipse([(wt-r,ysmoothed[wt]-r),(wt+r,ysmoothed[wt]+r)], 
                        outline = 'white', fill = 'blue', width = 1)
    
    return endimg, troughs, peaks

def createVideo(imgs:list):

    # so now we can process it with OpenCV functions

    initImg = imgs[0]

    with io.BytesIO() as videoFile:
        initImg.save(videoFile, format="WebP", append_images = imgs[1:], duration=1000, loop=0)
        

def main():
    pass

if __name__ == "__main__":
    pass