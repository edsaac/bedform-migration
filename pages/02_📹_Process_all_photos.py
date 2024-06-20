from multiprocessing import Pool
import tempfile
import cv2
import subprocess
import matplotlib.pyplot as plt
import streamlit as st
from st_lg17cam import buildSidebar, getDatabasePics, processPair, plotPeaksOrTroughsOverTime

from streamlit_extras.switch_page_button import switch_page

import pickle
import numpy as np
import pandas as pd

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title="[NU CEE440] Lab 2 - Process all photos",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None)

with open("assets/style.css") as f:
    css = f.read()
    st.markdown(
        f"""
        <style>
            {css}
        </style>
        """, unsafe_allow_html=True
    )

title = st.container()
placeholder = st.empty()

###################################
# Session state management
###################################
if "globalParameters" not in st.session_state.keys():
    st.warning(
        """Using default parameters. Go to the previous steps if 
        you'd like to change the processing parameters""", icon="🐜")

    with open("assets/globalParameters.pkl", 'rb') as f:
        st.session_state.globalParameters = pickle.load(f)

else:
    if not all(k in st.session_state.globalParameters.keys()
               for k in [
            "XSHIFT",
            "CROP_RANGE",
            "MASKING_THRESHOLD",
            "WINDOW_LENGHT",
            "POLYORDER",
            "MINIMAL_DISTANCE",
            "PROMINENCE"]
    ):

        st.warning(
            """Using default parameters. Go to the previous steps if 
            you'd like to change the processing parameters""", icon="🐜")

        with open("assets/globalParameters.pkl", 'rb') as f:
            st.session_state.globalParameters = pickle.load(f)

buildSidebar()

if "go_to_page_1" in st.session_state.keys():

    if st.session_state.go_to_page_1:
        st.session_state.pagx = -1
        st.session_state.page = -1
        switch_page("Process two photos")

if "go_to_page_3" in st.session_state.keys():

    if st.session_state.go_to_page_3:
        st.session_state.pagx = -1
        st.session_state.page = -1
        switch_page("Bed transport")

if 'pagx' not in st.session_state.keys():
    st.session_state.pagx = -1

if 'tempPagx' not in st.session_state.keys():
    st.session_state.tempPagx = {}


def prevPage(): st.session_state.pagx -= 1
def nextPage(): st.session_state.pagx += 1
def firstPage(): st.session_state.pagx = -1


def checkUploads():
    if all(pic in st.session_state.keys() for pic in ["leftPicFolder", "rightPicFolder"]):
        if st.session_state.leftPicFolder and st.session_state.rightPicFolder:
            if len(st.session_state.leftPicFolder) == len(st.session_state.rightPicFolder):
                nextPage()
            else:
                st.error(
                    """**⚠️⚠️ There must be the same number of photos at both sides ⚠️⚠️**""")
                firstPage()
        else:
            st.error("**⚠️⚠️ Upload pictures for both sides before continuing ⚠️⚠️**", icon="🖼️")
            firstPage()
    else:
        firstPage()


def buildNav():
    cols = st.columns(3, gap="small")
    with cols[0]:
        st.button("🔁 Restart", on_click=firstPage)
    with cols[1]:
        st.button("🔙 Go to previous step", on_click=prevPage)
    with cols[2]:
        st.button("🔜 Go to next step", on_click=nextPage)

########################################
# Header
########################################


with title:
    """
    # 📹 Process all the photos
    """

    cols = st.columns([3, 1])

    with cols[0]:
        with st.expander("👈 Check in the sidebar or below that these are the correct parameters you decided on"):
            st.json(st.session_state.globalParameters, expanded=True)

    with cols[1]:
        if st.button("🔙 Go to the beginning"):
            st.session_state.page = -1
            st.session_state.pagx = -1
            switch_page("Process two photos")

    with st.expander("In case you don't have a set of photos, download these to run the app!",
                     expanded=st.session_state.pagx == -1):

        col1, col2 = st.columns(2)

        with col1:
            with open('assets/2021/left.zip', 'rb') as myzip:
                st.download_button("📥 Download left-side pictures", myzip, "left.ZIP")

        with col2:
            with open('assets/2021/right.zip', 'rb') as myzip:
                st.download_button("📥 Download right-side pictures", myzip, "right.ZIP")


############# Page -1 ###################
# Upload photos
########################################
if st.session_state.pagx == -1:

    with placeholder.container():
        """
        ****
        ## 📤 Upload all your photos
        """

        with st.form(key="foldersForm", clear_on_submit=False):

            col1, col2 = st.columns(2)

            with col1:
                st.file_uploader("Left photos", "JPG", True, key="leftPicFolder")
            with col2:
                st.file_uploader("Right photos", "JPG", True, key="rightPicFolder")

            st.form_submit_button("🔜 Process!", on_click=checkUploads)

############# Page 0 ###################
# Picture database
########################################
elif st.session_state.pagx == 0:

    with placeholder.container():

        leftBytes = st.session_state.leftPicFolder
        rightBytes = st.session_state.rightPicFolder

        # Organize lefts and rights
        leftdb, leftPhotos = getDatabasePics(leftBytes)
        rightdb, rightPhotos = getDatabasePics(rightBytes)

        mergedb = leftdb.join(rightdb, lsuffix='_left', rsuffix='_right')
        deltaTime = leftdb['Time'] - rightdb['Time']
        mergedb["DeltaTime"] = [t.total_seconds() for t in deltaTime]

        """
        *****
        ## 1️⃣ Picture database

        Below is a list of the uploaded photos with their corresponding 
        timestamps. The column `DeltaTime` indicates how far in time are
        each couple of pictures. You will see a warning if a pair of
        photo timestamps differ by more than the threshoold.
        """

        def time_greater_than(cell_value, threshold=5.0):
            highlight = 'background-color: darkorange;'
            default = 'background-color: aquamarine;'
            _ = ''

            if abs(cell_value) >= threshold:
                return highlight
            else:
                return default

        threshold = st.slider("Time threshold", 0.0, 30.0, 5.0, 0.5, key="time_threshold")

        st.dataframe(mergedb.style.applymap(
            time_greater_than,
            subset=["DeltaTime"],
            threshold=threshold
        ))

        if np.any(np.greater_equal(np.absolute(mergedb["DeltaTime"]), threshold)):
            st.error(" At least a pair of photos do not correspond to the same instant",
                     icon="🐈‍⬛")

        st.session_state.tempPagx["mergedb"] = mergedb
        st.session_state.tempPagx["leftPhotos"] = leftPhotos
        st.session_state.tempPagx["rightPhotos"] = rightPhotos

        buildNav()

############# Page 1 ##############
elif st.session_state.pagx == 1:
    """
    *****
    ## 2️⃣ Process uploaded photos
    """

    with st.spinner(' 🚧 Processing your photos... 🚧'):
        nprocs = subprocess.check_output("nproc --all".split(" "))
        cols = st.columns(2)
        with cols[0]:
            st.metric("# of CPUs:", int(nprocs))
        with cols[1]:
            st.metric("# of photos:", st.session_state.tempPagx["mergedb"].shape[0])

        # To run using multiprocessing
        with Pool(processes=2) as pool:
            processedBlob = pool.map(
                processPair,
                zip(
                    st.session_state.tempPagx["leftPhotos"],
                    st.session_state.tempPagx["rightPhotos"]
                ))

    st.success("All images have been processed!", icon="🎊")

    allPhotos, allTroughs, allPeaks = map(list, zip(*processedBlob))

    st.session_state.tempPagx["allPhotos"] = allPhotos
    st.session_state.tempPagx["allTroughs"] = allTroughs
    st.session_state.tempPagx["allPeaks"] = allPeaks

    buildNav()

############# Page 2 ##############
elif st.session_state.pagx == 2:
    """
    *****
    ## 3️⃣ Visualize results
    """

    width, height = st.session_state.tempPagx["allPhotos"][0].size
    frame_fps = 5

    with (tempfile.NamedTemporaryFile(suffix=".mp4") as videoFile,
            tempfile.NamedTemporaryFile(suffix=".mp4") as convertedVideo):

        fourcc_mp4 = cv2.VideoWriter_fourcc(*'MP4V')
        out_mp4 = cv2.VideoWriter(videoFile.name, fourcc_mp4, frame_fps, (width, height), isColor=True)

        for photo in st.session_state.tempPagx["allPhotos"]:
            out_mp4.write(np.array(photo))
        out_mp4.release()

        subprocess.run(args=f"ffmpeg -y -i {videoFile.name} -c:v libx264 {convertedVideo.name}".split(" "))
        st.video(convertedVideo.name)

    buildNav()

############# Page 3 ##############
elif st.session_state.pagx == 3:
    """
    *****
    ## 4️⃣ Peaks and troughs location data
    """
    cols = st.columns(2)
    with cols[0]:
        "### 🕳️ List of troughs"
        df_troughs = pd.DataFrame(st.session_state.tempPagx["allTroughs"])
        st.dataframe(df_troughs)

    with cols[1]:
        "### ⛰️ List of peaks"
        df_peaks = pd.DataFrame(st.session_state.tempPagx["allPeaks"])
        st.dataframe(df_peaks)

    "### 📉 Plots over time"
    fig = plotPeaksOrTroughsOverTime(df_troughs, title="Troughs")
    st.pyplot(fig, transparent=True)

    fig = plotPeaksOrTroughsOverTime(df_peaks, title="Peaks")
    st.pyplot(fig, transparent=True)

    st.session_state.tempPagx["df_troughs"] = df_troughs
    st.session_state.tempPagx["df_peaks"] = df_peaks

    buildNav()

############# Page 4 ##############
elif st.session_state.pagx == 4:

    """
    *****
    ## 5️⃣ Summary and download data
    """

    cols = st.columns(2)

    with cols[0]:
        st.write("**💀 Something did not go as expected**")
        st.button("⏮️ Go to the very beginning", key="restart_3btn")
        st.button("🔙 Restart this page", on_click=firstPage)

    with cols[1]:
        st.write("**🛫 All seems fine!**")
        st.download_button(
            "🕳️  Click here to download the troughs as CSV",
            data=st.session_state.tempPagx["df_troughs"].to_csv().encode('utf-8'),
            file_name="df_troughs.csv")

        st.download_button(
            "⛰️  Click here to download the peaks as CSV",
            data=st.session_state.tempPagx["df_peaks"].to_csv().encode('utf-8'),
            file_name="df_peaks.csv")

    st.button("⏭️ _I'm ready to calculate geomorphodynamics!_ ⏭️",
              key="go_to_page_3")
