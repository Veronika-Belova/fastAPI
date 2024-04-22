import requests
import streamlit as st
import json
from PIL import Image

def main():
    st.title("Image and Text Classification")
    st.sidebar.title("Menu")
    choice = st.sidebar.selectbox("Choose Activity", ["Image Classification", "Text Sentiment Analysis"])
    
    if choice == "Image Classification":
        st.subheader("Image Classification")
        uploaded_image = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        if st.button("Classify Image") and uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image")
            files = {"file": uploaded_image.getvalue()}
            res = requests.post("http://127.0.0.1:8020/classify", files=files)
            st.write(json.loads(res.text)['prediction'])
            
    elif choice == "Text Sentiment Analysis":
        st.subheader("Text Sentiment Analysis")
        text_input = st.text_area("Enter the text you'd like to analyze for sentiment")
        
        if st.button("Analyze Sentiment"):
            if text_input != "":
                data = {"text": text_input}
                response = requests.post("http://127.0.0.1:8020/clf_text", json=data)
                
                if response.status_code == 200:
                    sentiment = response.json()['sentiment']
                    st.success(f"Sentiment: {sentiment}")
                else:
                    st.error("An error occurred during API request.")
            else:
                st.error("Please enter some text to analyze the sentiment.")

if __name__ == "__main__":
    main()