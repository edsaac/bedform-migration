import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from st_lg17cam import *


st.set_page_config(
    page_title="[NU CEE440] - Bed sediment transport",
    page_icon="⛳",
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

###################################
# Session state management
###################################
if "demodata" not in st.session_state.keys():
    st.session_state.demodata = dict()
    st.session_state.demodata["trough"] = pd.read_csv("assets/2021/df_troughs.csv")
    st.session_state.demodata["peak"] = pd.read_csv("assets/2021/df_peaks.csv")

###################################
# Dataframe processing
###################################

r"""
# 🏞️ **Bed sediment morphodynamics**

After processing all the images captured during your laboratory, 
you will end up with a couple of dataframes like the ones below.
"""

df_troughs_csv = st.session_state.demodata["trough"]
df_peaks_csv = st.session_state.demodata["peak"]

df_troughs_san = sanitizeDataframe(df_troughs_csv)
df_peaks_san = sanitizeDataframe(df_peaks_csv)

df_troughs = expandPeakOrTroughDf(df_troughs_san)
df_peaks = expandPeakOrTroughDf(df_peaks_san)


cols = st.columns(2)
with cols[0]:
    "### 🕳️ List of troughs"
    st.dataframe(df_troughs_san)

with cols[1]:
    "### ⛰️ List of peaks"
    st.dataframe(df_peaks_san)

for df in [df_peaks, df_troughs]:
    df["Time (s)"] = df["Timestamp"] - df["Timestamp"].iloc[0]
    df["Time (s)"] = [t.total_seconds() for t in df["Time (s)"]]

"""
*****

## Celerity $c$ [L/T]
"""
cols = st.columns([1.3, 1], gap="large")

with cols[1]:
    """
    The celerity is defined as the speed at which a bedform migrates. 

    Given the position over time plot, the celerity of each bedform can be 
    estimated as the change in position over time of the bedform, i.e., the
    slope of the line. 
    """

    st.info(
        """
        All bedforms have a different celerity.

        You can report your results as a [frequency distribution](https://en.wikipedia.org/wiki/Histogram), 
        indicating the mean and standard deviation.
        """, icon="☑️")

with cols[0]:

    fig, ax = plt.subplots(figsize=[5, 5])

    ax.scatter(
        df_troughs["Time (s)"],
        df_troughs["X(px)"],
        c='seagreen',
        s=200,
        alpha=0.5)

    ax.set(
        xlabel="Time since first photo (s)",
        ylabel="X [px]",
        title="Troughs over time"
    )

    line_kwarg = dict(
        c="black"
    )

    ax.axline((0, 500), (2100, 2000), **line_kwarg)
    ax.axline((500, 0), (3300, 2000), **line_kwarg)
    ax.axline((1900, 0), (4000, 2000), **line_kwarg)
    ax.axline((3000, 0), (4000, 1000), **line_kwarg)

    ax.text(1600, 1600,
            r"$\measuredangle c$",
            fontsize=20)

    st.pyplot(fig, transparent=True)

"""
## Lenght $L$ [L]
"""
cols = st.columns([1.3, 1], gap="large")

with cols[1]:
    """
    The lenght is the distance between two consecutive bedforms. 

    In the position over time plot, it corresponds to the vertical 
    distance between two consecutive trend lines.
    """

    st.info(
        """
        The **lenght** is not a constant for all bedforms.

        You can report your results as a [frequency distribution](https://en.wikipedia.org/wiki/Histogram), 
        indicating the mean and standard deviation.
        """, icon="☑️")

with cols[0]:

    fig, ax = plt.subplots(figsize=[5, 5])

    ax.scatter(
        df_troughs["Time (s)"],
        df_troughs["X(px)"],
        c='seagreen',
        s=200,
        alpha=0.5)

    ax.set(
        xlabel="Time since first photo (s)",
        ylabel="X [px]",
        title="Troughs over time"
    )

    line_kwarg = dict(
        c="grey",
        ls="dotted"
    )

    ax.axline((0, 500), (2100, 2000), **line_kwarg)
    ax.axline((500, 0), (3300, 2000), **line_kwarg)
    ax.axline((1900, 0), (4000, 2000), **line_kwarg)
    ax.axline((3000, 0), (4000, 1000), **line_kwarg)

    ax.annotate(
        r"$L$",
        xy=(1500, 1600),
        xycoords='data',
        xytext=(1500, 600),
        textcoords='data',
        fontsize=20,
        arrowprops=dict(arrowstyle="|-|"))

    st.pyplot(fig, transparent=True)


"""
## Height $H$ [L]
"""
cols = st.columns([1.3, 1], gap="large")

with cols[1]:
    """
    The bedform height is the difference in peak and trough elevations.

    To estimate $H$, we can plot all the elevations over time and determine
    a representative value.
    """

    st.warning(
        r"""
    Why do the troughs have a higher elevation than the peaks?
    """, icon="🤔")

    st.info(
        """
        Calculate the bedform height and assess the uncertainty on 
        your estimate.
        """, icon="☑️")

with cols[0]:

    fig, ax = plt.subplots(figsize=[5, 5])

    ax.scatter(
        df_troughs["Time (s)"],
        df_troughs["Z(px)"],
        c='salmon',
        s=200,
        alpha=0.5,
        label="Troughs")

    ax.axhline(y=df_troughs["Z(px)"].mean(),
               c="salmon")

    ax.scatter(
        df_peaks["Time (s)"],
        df_peaks["Z(px)"],
        c='violet',
        s=200,
        alpha=0.5,
        label="Peaks")

    ax.axhline(y=df_peaks["Z(px)"].mean(),
               c="violet")

    ax.set(
        xlabel="Time since first photo (s)",
        ylabel="Elevation Z [px]",
        title="Elevations over time",
        ylim=(50, 175)
    )

    ax.annotate(
        r"$H$",
        xy=(2000, 86),
        xycoords='data',
        xytext=(2000, 125),
        textcoords='data',
        fontsize=20,
        arrowprops=dict(arrowstyle="|-|"))

    ax.legend()

    st.pyplot(fig, transparent=True)

"""
## Bedload sediment flux $Q_s$
"""
cols = st.columns([1.3, 1], gap="large")

with cols[1]:
    r"""
    $$
        Q_s = \beta n c H
    $$

    Where $\beta$ is a shape factor around 0.5 for triangular
    bedforms, $n$ is the bed porosity, $c$ is the celerity and 
    $H$ is the bedform height. 
    """

with cols[0]:
    st.warning(
        r"""
    What are the units of $Q_s$?  Why is porosity involved here?
    """, icon="🤔")

    st.info(
        """
        Calculate $Q_s$ and discuss the uncertainty on 
        your estimate.
        """, icon="☑️")
