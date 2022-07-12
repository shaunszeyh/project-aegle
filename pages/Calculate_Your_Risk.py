import streamlit as st

st.markdown(
'''
    # Calculate Your Risk
    Note: We will not be collecting your data so be rest assured when you key in your information.    
'''
)

with st.form("my_form"):
    st.write("test")
    submitted = st.form_submit_button("Submit My Data")
