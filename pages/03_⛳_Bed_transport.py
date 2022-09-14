import streamlit as st

st.set_page_config(
    page_title = "[NU CEE440] - Bed sediment transport", 
    page_icon  = "⛳",
    layout = "wide", 
    initial_sidebar_state = "auto",
    menu_items=None)

###################################
## Session state management
###################################
if "demodata" not in st.session_state.keys():
    st.session_state.demodata = True

###################################
## Theory
###################################