import streamlit as st
import numpy as np

#print(run_neural_network(np.array([1.0,1.0,0.0,0.22985206447339368,0.20799180327868855,0.0,1.0,0.5,1.0,0.0]))) # Should return [0]

st.set_page_config(
    page_title = "Project Aegle",
)

st.title("Project Aegle")
st.write("This website calculates your risk of stroke and gives you some suggestions to lower it!")
st.sidebar.success("Select a demo above.")
