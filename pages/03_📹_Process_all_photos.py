import plotly.express as px
import matplotlib.pyplot as plt

import streamlit as st
from st_lg17cam import *
from streamlit_extras.switch_page_button import switch_page
import pickle

plt.style.use('assets/edwin.mplstyle')

st.set_page_config(
    page_title = "[NU CEE440] - Process all photos", 
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
    with st.sidebar: 
        "#### Image processing parameters"
        with st.expander("Color classification:",expanded=False):
            st.metric("XSHIFT", st.session_state.globalParameters["XSHIFT"])

        with st.expander("Image combination",expanded=False):
            st.metric("MASKING_THRESHOLD", st.session_state.globalParameters["MASKING_THRESHOLD"])
        
        with st.expander("Smoothing and filtering:",expanded=False):
            cols = st.columns(2)
            with cols[0]: st.metric("WINDOW_LENGHT", st.session_state.globalParameters["WINDOW_LENGHT"])
            with cols[1]: st.metric("POLYORDER", st.session_state.globalParameters["POLYORDER"])
        
        with st.expander("Peak identification:",expanded=False):
            cols = st.columns(2)
            with cols[0]: st.metric("MINIMAL_DISTANCE", st.session_state.globalParameters["MINIMAL_DISTANCE"])
            with cols[1]: st.metric("PROMINENCE", st.session_state.globalParameters["PROMINENCE"])

if "restart_3btn" not in st.session_state.keys():
    st.session_state.restart_3btn = False
else:
    if st.session_state.restart_3btn:
        switch_page("combine a pair")

"""
# 📹 Process all the photos
"""

cols = st.columns([3,1])

with cols[0]:
    with st.expander("0️⃣ Check in the sidebar or below that these are the correct parameters you decided on 👈"):
        st.json(st.session_state.globalParameters,expanded=False)

with cols[1]:
    if st.button("🔙 Go to the beginning"):
        switch_page("Combine a pair")

###################################
## Processing all photos
###################################
"****" 
with st.form(key="foldersForm", clear_on_submit= False):
    """
    ## ▶️ Upload all your photos
    """
    col1, col2 = st.columns(2)
    with col1: 
        st.file_uploader("Left photos","JPG",True,key="leftPicFolder")
    with col2: 
        rightBytes = st.file_uploader("Right photos","JPG",True,key="rightPicFolder")
    st.form_submit_button("Process!")

if not st.session_state.restart_3btn:

    leftBytes = st.session_state.leftPicFolder
    rightBytes = st.session_state.rightPicFolder
    
    if leftBytes and rightBytes:
        
        if len(leftBytes) != len(rightBytes):
            st.error("There must be the same number of photos at both sides")

        else:
            st.session_state.uploadedFolderCheck = False

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

            """
            *****
            ## 2️⃣ Process uploaded photos
            """

            with st.spinner(' 🚧 Processing your photos...'):
                processedBlob = [processPair((l, r), parameters=st.session_state.globalParameters) for l,r in zip(leftPhotos,rightPhotos)]
            
            st.success("All images have been processed!", icon="🎊")
            allPhotos, allTroughs, allPeaks = map(list,zip(*processedBlob))    
        
            """
            *****
            ## 3️⃣ Visualize results
            """

            allPhotosArray = np.stack(allPhotos)
            
            fig = px.imshow(allPhotosArray, 
                animation_frame = 0, 
                labels = {
                    "animation_frame":"slice"})
            
            fig.update_layout(
                showlegend=False,
                autosize=True,
                width=800,
                title={
                    'text': "🎥 Processed pictures",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis={
                    'title':"X (px)"},
                yaxis={
                    'title':"Z (px)"},
                font={
                    'size': 14}
                )

            st.plotly_chart(fig,use_container_width=True)

            """
            *****
            ## 4️⃣ Peaks and troughs location data
            """
            cols = st.columns(2)
            with cols[0]:
                "### 🕳️ List of troughs"
                df_troughs = pd.DataFrame(allTroughs)
                st.dataframe(df_troughs)
            
                fig,ax = plt.subplots()
                for _, row in df_troughs.iterrows():
                    ax.scatter(
                        [pd.to_datetime(row["Timestamp"],format=r'%Y:%m:%d %H:%M:%S') for _ in row["Z(px)"]],
                        row["Z(px)"], 
                        c='purple')

                ax.set_xlabel("X [px]")
                ax.set_ylabel("Z [px]")
                st.pyplot(fig, transparent=True)

            with cols[1]:
                "### ⛰️ List of peaks"
                df_peaks = pd.DataFrame(allPeaks)
                st.dataframe(df_peaks)

                
                fig,ax = plt.subplots()
                for _, row in df_peaks.iterrows():
                    ax.scatter(row["X(px)"],row["Z(px)"], c='purple')

                ax.set_xlabel("X [px]")
                ax.set_ylabel("Z [px]")
                st.pyplot(fig, transparent=True)

            """
            *****
            ## 5️⃣ Summary and download data
            """

            st.session_state["allTroughs"] = allTroughs
            st.session_state["allPeaks"] = allPeaks

            cols = st.columns(2)

            with cols[0]:
                st.write("**💀 Something did not go as expected**")
                st.button("🔙 I want to start over", key="restart_3btn")           

            with cols[1]:
                st.write("**🛫 All seems fine!**")
                st.download_button(
                    "🕳️  Click here to download the troughs as CSV",
                    data = df_troughs.to_csv().encode('utf-8'),
                    file_name = "df_troughs.csv")
                
                st.download_button(
                    "⛰️  Click here to download this peaks as CSV",
                    data = df_peaks.to_csv().encode('utf-8'),
                    file_name = "df_peaks.csv")