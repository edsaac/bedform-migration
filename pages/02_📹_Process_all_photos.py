from multiprocessing import Pool
import tempfile
import cv2
from PIL import ImageDraw
import subprocess
import plotly.express as px
import matplotlib.pyplot as plt

import streamlit as st
import extra_streamlit_components as stx
from st_lg17cam import *
from streamlit_extras.switch_page_button import switch_page
import pickle
import os 

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title = "[NU CEE440] Lab 2 - Process all photos", 
    page_icon  = "📹",
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)


###################################
## Session state management
###################################
if "globalParameters" not in st.session_state.keys(): 
    st.warning("Using default parameters. Go to the previous steps if you'd like to change one of the parameters", icon="🐜")

    with open("assets/globalParameters.pkl",'rb') as f:
        st.session_state.globalParameters = pickle.load(f)

else:
    if not all( k in st.session_state.globalParameters.keys() 
        for k in [
            "XSHIFT",
            "MASKING_THRESHOLD",
            "WINDOW_LENGHT",
            "POLYORDER",
            "MINIMAL_DISTANCE",
            "PROMINENCE"]
    ):

        st.warning("Using default parameters. Go to the previous steps if you'd like to change one of the parameters", icon="🐜")

        with open("assets/globalParameters.pkl",'rb') as f:
            st.session_state.globalParameters = pickle.load(f)

with st.sidebar:
    st.write("Parameters:")
    for k,v in st.session_state.globalParameters.items():
        if "PERS" not in k and "BARR" not in k:
            st.metric(k,v)

if "restart_3btn" not in st.session_state.keys():
    st.session_state.restart_3btn = False
else:
    if st.session_state.restart_3btn:
        st.session_state.pagx = -1
        st.session_state.page = -1
        switch_page("Process two photos")

if 'pagx' not in st.session_state.keys(): 
    st.session_state.pagx = -1

if 'tempPagx' not in st.session_state.keys():
    st.session_state.tempPagx = {}

def prevPage(): st.session_state.pagx -= 1
def nextPage(): st.session_state.pagx += 1
def firstPage(): st.session_state.pagx = 0

"""
# 📹 Process all the photos
"""

cols = st.columns([3,1])

with cols[0]:
    with st.expander("👈 Check in the sidebar or below that these are the correct parameters you decided on"):
        st.json(st.session_state.globalParameters,expanded=True)

with cols[1]:
    if st.button("🔙 Go to the beginning"):
        st.session_state.globalParameters = dict()
        st.session_state.page = -1
        st.session_state.pagx = -1
        switch_page("Process two photos")

###################################
## Processing all photos
###################################

with st.expander("In case you don't have a set of photos, download these to run the app!",
        expanded=st.session_state.pagx == -1):

    col1, col2 = st.columns(2)

    with col1:
        with open('assets/2021/left.zip', 'rb') as myzip:
            st.download_button("Download left-side pictures", myzip, "left.ZIP")

    with col2: 
        with open('assets/2021/right.zip', 'rb') as myzip:
            st.download_button("Download right-side pictures", myzip, "right.ZIP")

with st.expander("▶️ Upload all your photos",
        expanded=st.session_state.pagx == -1):

    with st.form(key="foldersForm", clear_on_submit= False):

            col1, col2 = st.columns(2)

            with col1: st.file_uploader("Left photos","JPG",True,key="leftPicFolder")
            with col2: rightBytes = st.file_uploader("Right photos","JPG",True,key="rightPicFolder")
            st.form_submit_button("Process!",on_click=nextPage)


if not st.session_state.restart_3btn \
    and "leftPicFolder" in st.session_state.keys() \
    and "rightPicFolder" in st.session_state.keys():

    leftBytes = st.session_state.leftPicFolder
    rightBytes = st.session_state.rightPicFolder
    
    if leftBytes and rightBytes:
        
        if len(leftBytes) != len(rightBytes):
            st.error("There must be the same number of photos at both sides")

        else:
            ############# Page 0 ##############
            if st.session_state.pagx == 0:
            

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

                def time_greater_than(cell_value, threshold = 5.0):
                    highlight = 'background-color: darkorange;'
                    default = 'background-color: aquamarine;'
                    donothing = ''

                    if abs(cell_value) >= threshold:
                        return highlight
                    else:
                        return default
            
                threshold = st.slider("Time threshold", 0.0, 30.0, 5.0, 0.5, key="time_threshold")

                st.dataframe(mergedb.style.applymap(
                    time_greater_than, 
                    subset=["DeltaTime"], 
                    threshold = threshold
                    ))
            
                if np.any(np.greater_equal(np.absolute(mergedb["DeltaTime"]), threshold)): 
                    st.error(" At least a pair of photos do not correspond to the same instant",
                        icon="🐈‍⬛")
                
                st.session_state.tempPagx["mergedb"] = mergedb
                st.session_state.tempPagx["leftPhotos"] = leftPhotos
                st.session_state.tempPagx["rightPhotos"] = rightPhotos

                st.button("➡️ Go to next step", on_click=nextPage)

            ############# Page 1 ##############
            elif st.session_state.pagx == 1:
                """
                *****
                ## 2️⃣ Process uploaded photos
                """

                with st.spinner(' 🚧 Processing your photos... 🚧'):
                    nprocs = subprocess.check_output("nproc --all".split(" "))
                    cols = st.columns(2)
                    with cols[0]: st.metric("# of CPUs:",int(nprocs))
                    with cols[1]: st.metric("# of photos:",st.session_state.tempPagx["mergedb"].shape[0])
                
                    ## To run using multiprocessing
                    with Pool() as pool:
                        processedBlob = pool.map(
                            processPair, 
                            zip(
                                st.session_state.tempPagx["leftPhotos"],
                                st.session_state.tempPagx["rightPhotos"]
                                ))

                st.success("All images have been processed!", icon="🎊")
                
                allPhotos, allTroughs, allPeaks = map(list,zip(*processedBlob)) 
                
                st.session_state.tempPagx["allPhotos"] = allPhotos
                st.session_state.tempPagx["allTroughs"] = allTroughs
                st.session_state.tempPagx["allPeaks"] = allPeaks

                st.button("➡️ Go to next step", on_click=nextPage)

            ############# Page 2 ##############
            elif st.session_state.pagx == 2:
                """
                *****
                ## 3️⃣ Visualize results
                """

                width,height  = st.session_state.tempPagx["allPhotos"][0].size
                frame_fps = 5
            
                with (tempfile.NamedTemporaryFile(suffix=".mp4") as videoFile,
                    tempfile.NamedTemporaryFile(suffix=".mp4") as convertedVideo):
                
                    fourcc_mp4 = cv2.VideoWriter_fourcc(*'MP4V')
                    out_mp4 = cv2.VideoWriter(videoFile.name, fourcc_mp4, frame_fps, (width, height),isColor = True)

                    for photo in st.session_state.tempPagx["allPhotos"]: out_mp4.write(np.array(photo))
                    out_mp4.release()
                
                    subprocess.run(args=f"ffmpeg -y -i {videoFile.name} -c:v libx264 {convertedVideo.name}".split(" "))
                    st.video(convertedVideo.name)

                st.button("➡️ Go to next step", on_click=nextPage)

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
                
                fig,ax = plt.subplots(figsize=[18,8])
                for _, row in df_troughs.iterrows():
                    ax.scatter(
                        [pd.to_datetime(row["Timestamp"],format=r'%Y:%m:%d %H:%M:%S') for _ in row["X(px)"]],
                        row["X(px)"], 
                        c='purple')

                ax.set(
                    xlabel="Timestamp",
                    ylabel="X [px]",
                    title="Troughs"
                )
                    
                st.pyplot(fig, transparent=True)

                    
                fig,ax = plt.subplots(figsize=[18,8])
                for _, row in df_peaks.iterrows():
                    ax.scatter(
                        [pd.to_datetime(row["Timestamp"],format=r'%Y:%m:%d %H:%M:%S') for _ in row["X(px)"]],
                        row["X(px)"], 
                        c='purple')

                ax.set(
                    xlabel="Timestamp",
                    ylabel="X [px]",
                    title="Peaks"
                )

                st.pyplot(fig, transparent=True)

                st.session_state.tempPagx["df_troughs"] = df_troughs
                st.session_state.tempPagx["df_peaks"] = df_peaks

                st.button("➡️ Go to next step", on_click=nextPage)
            
            ############# Page 4 ##############
            elif st.session_state.pagx == 4:
            
                """
                *****
                ## 5️⃣ Summary and download data
                """

                cols = st.columns(2)

                with cols[0]:
                    st.write("**💀 Something did not go as expected**")
                    st.button("🔙 I want to start over", key="restart_3btn")           

                with cols[1]:
                    st.write("**🛫 All seems fine!**")
                    st.download_button(
                        "🕳️  Click here to download the troughs as CSV",
                        data = st.session_state.tempPagx["df_troughs"].to_csv().encode('utf-8'),
                        file_name = "df_troughs.csv")
                    
                    st.download_button(
                        "⛰️  Click here to download this peaks as CSV",
                        data = st.session_state.tempPagx["df_peaks"].to_csv().encode('utf-8'),
                        file_name = "df_peaks.csv")