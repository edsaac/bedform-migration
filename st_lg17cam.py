
from PIL import Image, ImageDraw
import subprocess
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import io
import numpy as np
import graphviz
import pickle


@st.experimental_singleton
def generateProcessGraph():

    parent = graphviz.Digraph(name="Parent")
    g_pair = graphviz.Digraph(name="cluster_Pair")
    g_calc = graphviz.Digraph(name="cluster_Calculate")

    ## Ranking and order
    for g in [g_pair, g_calc, parent]:
        g.graph_attr["rank"] = "same"
        g.graph_attr['rankdir'] = 'LR'

    ## Formatting and styling
    parent.graph_attr['bgcolor'] = '#F8F8FF'

    # Graph/Cluster styling
    g_pair.graph_attr["label"] = 'Processing a single pair of photos'
    g_pair.graph_attr["style"] = 'dotted'
    g_pair.graph_attr["color"] = 'grey'

    g_calc.graph_attr["label"] = 'Geomorphodynamics'
    g_calc.graph_attr["style"] = 'dotted'
    g_calc.graph_attr["color"] = 'grey'

    # Node styling
    g_pair.node_attr["shape"] = "box"
    g_pair.node_attr["style"] = "filled"
    g_pair.node_attr["fillcolor"] = "lightskyblue"
    g_pair.node_attr["color"] = "navy"

    g_calc.node_attr["shape"] = "box"
    g_calc.node_attr["style"] = 'filled'
    g_calc.node_attr["fillcolor"] = "paleturquoise1"
    g_calc.node_attr["color"] = 'grey'

    # Topology
    g_pair.node("left", "Left-side\nphoto")
    g_pair.node("right", "Right-side\nphoto")
    g_pair.node("merged", "Merged\nphoto")
    g_pair.node("interface", "Sediment-water\ninterface")
    g_pair.node("peaks", "Peaks\nlocation")
    g_pair.node("troughs", "Troughs\nlocation")

    g_pair.edge("left", "merged")
    g_pair.edge("right", "merged")
    g_pair.edge("merged", "interface")
    g_pair.edge("interface", "peaks")
    g_pair.edge("interface", "troughs")

    g_calc.node("celerity", "Celerity")
    g_calc.node("height", "Bedform\nheight")
    g_calc.node("lenght", "Bedform\nlenght")
    g_calc.node("flux", "Sediment\nflux")

    parent.subgraph(g_calc)
    parent.subgraph(g_pair)

    parent.node("common", "Repeat for all\npairs of photos",
                shape="folder")

    parent.edge("troughs", "common")
    parent.edge("peaks", "common")
    parent.edge("common", "celerity")
    parent.edge("common", "height")
    parent.edge("common", "lenght")
    parent.edge("common", "flux")

    parent.subgraph(g_calc)

    return parent


def buildSidebar():
    with st.sidebar:
        st.write("Parameters:")
        for k, v in st.session_state.globalParameters.items():
            if any(t in k for t in ["PERS", "BARR"]):
                pass
            elif any(t in k for t in ["CROP"]):
                st.metric(k, f"{v[0]} - {v[1]}")
            else:
                st.metric(k, v)


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
    a, b, c = params

    with (
        NamedTemporaryFile(prefix="in",  suffix=".jpg") as in_img,
        NamedTemporaryFile(prefix="out", suffix=".jpg") as out_img
    ):

        # Writes the input photo to a temporal file
        img.save(in_img.name)

        # Execute ImageMagick
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


def fixPerspective(img, pointsFrom, pointsTarget):
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
    originPairs = list(zip(pointsFrom['x'], pointsFrom['y']))
    targetPairs = list(zip(pointsTarget['x'], pointsTarget['y']))

    with (
        NamedTemporaryFile(prefix="in",  suffix=".jpg") as in_img,
        NamedTemporaryFile(prefix="out", suffix=".jpg") as out_img
    ):

        img.save(in_img.name)

        listCoords = ""
        for o, t in zip(originPairs, targetPairs):
            listCoords += f"{o[0]},{o[1]} {t[0]},{t[1]}  "

        # Execute ImageMagick
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
    dates = [pd.to_datetime(f, format=timefmt) for f in timestamps]
    deltat = abs(dates[0] - dates[1])

    if deltat > pd.to_timedelta("00:00:30"):
        st.error(
            f"""
            Looks like there are timestamps that do not coincide!
            {dates[0]} and {dates[1]}
            """)


def show_two_imgs(imgs, addTimestamp=False, imshow_kwargs={}):
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
        figsize=[18, 18/1.5/2],
        sharex=True,
        sharey=True,
        gridspec_kw=dict(
            hspace=0,
            wspace=0.01)
    )

    fig, axs = plt.subplots(1, 2, **shared_kw)

    for ax, img in zip(axs, imgs):
        ax.imshow(img, **imshow_kwargs)
        ax.grid()

        if addTimestamp:
            ax.set_title(get_timestamp(img))

    return fig


def plotPeaksOrTroughsOverTime(df, title=""):
    '''
    Plots the location of all the identified peaks or troughs 
    over time

    Parameters
    ----------
    df : pd.DataFrame
        dataframe generated with timestamp and location

    Returns
    -------
    plt.Figure
        Matplotlib figure

    Notes
    -----

    '''
    fig, ax = plt.subplots(figsize=[12, 5])

    for _, row in df.iterrows():
        ax.scatter(
            [pd.to_datetime(row["Timestamp"], format=r'%Y:%m:%d %H:%M:%S') for _ in row["X(px)"]],
            row["X(px)"],
            c='purple',
            s=200)

    ax.set(
        xlabel="Timestamp",
        ylabel="X [px]",
        title=title
    )

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
        if isinstance(date, bytes):
            date = date.decode()

        nameImg.append(f.name)
        dateImg.append(date)
        pilImgs.append(img)

    db = pd.DataFrame({'File': nameImg, 'Time': dateImg, 'Imgs': pilImgs})
    db['Time'] = pd.to_datetime(db['Time'], format='%Y:%m:%d %H:%M:%S')
    db.sort_values(by=['Time'], inplace=True, ignore_index=True)

    return db[['Time', 'File']], db['Imgs']


def sanitizeDataframe(df):
    '''
    Converts the CSV dataframe into np.arrays and datetimes

    Parameters
    ----------
    df : pd.DataFrame
        Must have the columns Timestamp, X(px) and Z(px) as str objects

    Returns
    -------    
    pd.DataFrame
        The same but easy to calculate on

    '''

    df_clean = pd.DataFrame()
    df_clean["Timestamp"] = pd.to_datetime(df["Timestamp"], format=r'%Y:%m:%d %H:%M:%S')
    df_clean["X(px)"] = [np.array(x.replace("[", "").replace("]", "").split()).astype(np.float) for x in df["X(px)"]]
    df_clean["Z(px)"] = [np.array(x.replace("[", "").replace("]", "").split()).astype(np.float) for x in df["Z(px)"]]

    return df_clean


def expandPeakOrTroughDf(df):

    df_expanded = pd.DataFrame(columns=["Timestamp", "X(px)", "Z(px)"])

    for _, row in df.iterrows():
        small = pd.DataFrame(
            {"X(px)": row["X(px)"],
             "Z(px)": row["Z(px)"]})
        small["Timestamp"] = row["Timestamp"]

        df_expanded = pd.concat([df_expanded, small])

    return df_expanded


def processPair(imgs: tuple):
    '''
    Streamlines the processing pair process using the parameters in 
    the session state.

    Parameters
    ----------
    imgs : tuple[st.UploadedFile]
        Tuple of uploaded files

    Returns
    -------    
    PIL.Image
        A processed image.

    Notes
    -----


    '''

    from scipy.signal import savgol_filter, find_peaks

    if "globalParameters" in st.session_state.keys():
        BARREL_CORRECTIONS = st.session_state.globalParameters["BARREL_CORRECTIONS"]
        PERS_ORIGIN_LEFT = st.session_state.globalParameters["PERS_ORIGIN_LEFT"]
        PERS_ORIGIN_RIGHT = st.session_state.globalParameters["PERS_ORIGIN_RIGHT"]
        PERS_TARGET_LEFT = st.session_state.globalParameters["PERS_TARGET_LEFT"]
        PERS_TARGET_RIGHT = st.session_state.globalParameters["PERS_TARGET_RIGHT"]

        XSHIFT = st.session_state.globalParameters["XSHIFT"]
        CROP_RANGE = st.session_state.globalParameters["CROP_RANGE"]

        MASKING_THRESHOLD = st.session_state.globalParameters["MASKING_THRESHOLD"]

        WINDOW_LENGHT = st.session_state.globalParameters["WINDOW_LENGHT"]
        POLYORDER = st.session_state.globalParameters["POLYORDER"]

        MINIMAL_DISTANCE = st.session_state.globalParameters["MINIMAL_DISTANCE"]
        PROMINENCE = st.session_state.globalParameters["PROMINENCE"]

    else:
        st.warning("Using default parameters. Go to the previous steps if you'd like to change one of the parameters", icon="🐜")
        with open("assets/globalParameters.pkl", 'rb') as f:
            st.session_state.globalParameters = pickle.load(f)

    raw_imgs = [f for f in imgs]
    raw_imgs_timestamps = [get_timestamp(f) for f in raw_imgs]
    checkTimeStamps(raw_imgs_timestamps)

    # Fix barrel distortion
    barr_imgs = [fixBarrelDistortion(img, BARREL_CORRECTIONS) for img in raw_imgs]

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
    inte_imgs = [Image.fromarray(img.astype(np.uint8), "L") for img in equa_imgs]

    # Overlap images
    (width1, height1) = inte_imgs[0].size
    (width2, height2) = inte_imgs[1].size

    result_width = width1 + XSHIFT
    result_height = max(height1, height2)

    joined_img = Image.new('L', (result_width, result_height))
    joined_img.paste(im=inte_imgs[1], box=(XSHIFT, 0))
    joined_img.paste(im=inte_imgs[0], box=(0, 0))

    # Color classification for edge detection
    masked = np.ma.masked_greater(joined_img, MASKING_THRESHOLD)
    ycoord = np.ma.count_masked(masked, axis=0)

    # Filter and smooth the edge profile
    SAVGOL_FILTER_PARAMS = {
        'window_length': WINDOW_LENGHT,
        'polyorder': POLYORDER}

    ysmoothed = savgol_filter(ycoord, **SAVGOL_FILTER_PARAMS)

    # Locate peaks and throughs
    TROUGH_FINDER_PARAMS = dict(distance=MINIMAL_DISTANCE, height=15, prominence=PROMINENCE)
    PEAK_FINDER_PARAMS = dict(distance=MINIMAL_DISTANCE, prominence=PROMINENCE)
    whereTroughs, _ = find_peaks(ysmoothed, **TROUGH_FINDER_PARAMS)
    wherePeaks, _ = find_peaks(-ysmoothed, **PEAK_FINDER_PARAMS)

    # Save the extracted peak/through data
    timestamp = raw_imgs_timestamps[0]
    troughs = {
        'Timestamp': timestamp,
        'X(px)': np.array([round(w, ndigits=1) for w in whereTroughs]),
        'Z(px)': np.array([round(ysmoothed[w], ndigits=2) for w in whereTroughs])
    }

    peaks = {
        'Timestamp': timestamp,
        'X(px)': np.array([round(w, ndigits=1) for w in wherePeaks]),
        'Z(px)': np.array([round(ysmoothed[w], ndigits=2) for w in wherePeaks])
    }

    # Create Line over image
    endimg = Image.new('RGB', joined_img.size)
    endimg.paste(im=joined_img, box=(0, 0))

    lineimg = ImageDraw.Draw(endimg)
    lineimg.line(list(zip(np.arange(joined_img.size[0]), ysmoothed)),
                 fill="orange", width=7)

    # Create Points over image
    r = 10
    for wt in whereTroughs:
        lineimg.ellipse([(wt-r, ysmoothed[wt]-r), (wt+r, ysmoothed[wt]+r)],
                        outline='white', fill='red', width=1)

    for wt in wherePeaks:
        lineimg.ellipse([(wt-r, ysmoothed[wt]-r), (wt+r, ysmoothed[wt]+r)],
                        outline='white', fill='blue', width=1)

    return endimg, troughs, peaks


def createVideo(imgs: list):

    # so now we can process it with OpenCV functions

    initImg = imgs[0]

    with io.BytesIO() as videoFile:
        initImg.save(videoFile, format="WebP", append_images=imgs[1:], duration=1000, loop=0)


def main():
    pass


if __name__ == "__main__":
    pass
