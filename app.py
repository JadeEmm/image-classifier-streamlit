import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image

def main():
    st.title('Image Classifier Using Machine Learning')
    st.text('Upload the Image')
    #Streamlit app code goes here
    uploaded_file = st.file_uploader('Choose an image...', type='jpg')
    model = pickle.load(open('img_model.p', 'rb'))
    if uploaded_file is not None:
      img = Image.open(uploaded_file)
      st.image(img, caption='Uploaded Image')
    if st.button('Predict'):
        categories =['pretty sunflower', 'tesla car', 'glass of water' ]
        #Model prediction code
        st.write('Result:')
        flat_data = []
        img = np.array(img)
        img_resized = resize(img,(150,150,3))
        flat_data.append(img_resized.flatten())
        flat_data = np.array(flat_data)
        y_out = model.predict(flat_data)
        y_out = categories[y_out[0]]
        st.title(f'Prediction: {y_out}')
        #Get percentage of prediction
        q = model.predict_proba(flat_data)
        for index, item in enumerate(categories):
          st.write(f'{item} : {q[0][index]*100:.4f}%')
if __name__ == '__main__':
    main()
