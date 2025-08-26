import streamlit as st
import pickle
import pandas as pd

# load mô hình đã lưu
with open("career_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Career Level Classification")

# hàm reset form
def reset_form():
    defaults = {
        "title": "",
        "location": "",
        "description": "",
        "function": "",
        "industry": ""
    }
    for key, value in defaults.items():
        st.session_state[key] = value

# tạo form
with st.form(key="input_form"):
    title = st.text_input("Title", value=st.session_state.get("title", ""), key="title")
    location = st.text_input("Location", value=st.session_state.get("location", ""), key="location")
    description = st.text_area("Description", value=st.session_state.get("description", ""), key="description")
    function = st.text_input("Function", value=st.session_state.get("function", ""), key="function")
    industry = st.text_input("Industry", value=st.session_state.get("industry", ""), key="industry")

    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button(label="Dự đoán")
    with col2:
        reset_button = st.form_submit_button(label="Thử lại", on_click=reset_form)

if submit_button:
    input_df = pd.DataFrame({
        "title": [st.session_state["title"]],
        "location": [st.session_state["location"]],
        "description": [st.session_state["description"]],
        "function": [st.session_state["function"]],
        "industry": [st.session_state["industry"]]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"Phân loại của công việc trên: {prediction}")
