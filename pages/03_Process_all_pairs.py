from cmath import exp
import streamlit as st
from st_lg17cam import *
from streamlit_extras.switch_page_button import switch_page
import pickle

st.set_page_config(
    page_title = None, 
    page_icon  = None,
    layout = "wide", 
    initial_sidebar_state = "collapsed",
    menu_items=None)

"""
# Process all the images
"""

"****" 

if "uploadedFolderCheck" not in st.session_state.keys():
    st.session_state.uploadedFolderCheck = True

if "globalParameters" not in st.session_state.keys(): 
    
    st.warning("Using default parameters. Go to the previous steps if you'd like to change one of the parameters", icon="🐜")

    with open("assets/globalParameters.pkl",'rb') as f:
        st.session_state.globalParameters = pickle.load(f)

if True:

    with st.expander("0️⃣ Check that these are the correct parameters you decided on 👉"):
        st.json(st.session_state.globalParameters,expanded=False)

    with st.expander("▶️ Upload all your photos", expanded=st.session_state.uploadedFolderCheck):
        
        col1, col2 = st.columns(2)
        with col1: 
            leftBytes = st.file_uploader("Left photos","JPG",True,key="leftPicFolder")
        with col2: 
            rightBytes = st.file_uploader("Right photos","JPG",True,key="rightPicFolder")

    if leftBytes and rightBytes:
        if len(leftBytes) != len(rightBytes):
            st.error("There must be the same number of photos at both sides")

        else:
            st.session_state.uploadedFolderCheck = False

            # Organize lefts and rights
            leftdb, leftPhotos = getDatabasePics(leftBytes)
            rightdb, rightPhotos = getDatabasePics(rightBytes)
            
            mergedb = leftdb.join(rightdb, lsuffix='_left', rsuffix='_right')
            st.dataframe(mergedb)  # streamlit fails with datetimes

            # Process all photos
            with st.spinner('Processing your photos...'):
                processedBlob = [processPair((l, r), parameters=st.session_state.globalParameters) for l,r in zip(leftPhotos,rightPhotos)]
                st.balloons()

            allPhotos, allTroughs, allPeaks = map(list,zip(*processedBlob))
            
            with st.expander("Show all processed photos", expanded=False):
                for photo in allPhotos:
                    st.image(createVideo(allPhotos))

            with st.expander("Show animation", expanded=False):
                st.video()

            cols = st.columns(2)
            with cols[0]:
                "### 🕳️ List of troughs"
                df_troughs = pd.DataFrame(allTroughs)
                
                st.dataframe(df_troughs)
                st.download_button(
                    "🕳️  Click here to download the dataset as CSV",
                    data = df_troughs.to_csv().encode('utf-8'),
                    file_name = "df_troughs.csv")
            
            with cols[1]:
                "### ⛰️ List of peaks"
                df_peaks = pd.DataFrame(allPeaks)

                st.dataframe(df_peaks)
                st.download_button(
                    "⛰️  Click here to download this dataset as CSV",
                    data = df_peaks.to_csv().encode('utf-8'),
                    file_name = "df_peaks.csv")