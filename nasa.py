import pandas as pd
import numpy as np
import joblib
import streamlit as st
import sklearn
import xgboost
from PIL import Image

model = joblib.load(r"C:\Users\srkar\GreatLearning\Self\algoshack\nasa.pkl")


def welcome():
    return 'Welcome all'

def prediction(loc,vg,i, v, d):
    sample =  pd.DataFrame({'loc':[loc] , 'v(g)':[vg], 
    'i':[i] , 'v':[v] , 'd':[d] })
    prediction = model.predict(sample)
    return prediction 


def main():
    st.title('Test Software for Defects')
    
    st.markdown('This web app can classify software as non-defective/ defective')
    
    html_description = """ 
    <h3>
    Model Used: XGB Classifier
    </h3>
     <table>
     <tr>
     <th> </th>
    <th>Precision</th>
    <th>Recall</th>
    <th>f1 Score</th>
    <th>Support</th>
    </tr>
    <tr>
    <td>No Defects</td>
    <td>0.91</td>
    <td>0.64</td>
    <td>0.75</td>
    <td>1755</td>
    </tr>
    <tr>
    <td>No Defects</td>
    <td>0.33</td>
    <td>0.72</td>
    <td>0.45</td>
    <td>421</td>
    </tr>
    <tr>
    <td>Accuracy</td>
    <td> </td>
    <td> </td>
    <td>0.66</td>
    <td>2176</td>
    </tr>
    </table> 
    """
    st.sidebar.markdown(html_description, unsafe_allow_html=True)
    
    st.sidebar.markdown('')
    st.sidebar.markdown('Confusion Matrix')
    
    image = Image.open(r'C:\Users\srkar\GreatLearning\Self\algoshack\output.png')

    st.sidebar.image(image, caption='Confusion Matrix')

    st.subheader('Parameters')

    loc = st.text_input('McCabe Line Count of Code')
    vg = st.text_input('McCabe Cyclomatic Complexity')
    i = st.text_input('Halstead Intelligence')
    v = st.text_input('Halstead Volume')
    d = st.text_input('Halstead Difficulty')
    
    result = ''

    st.markdown(' ')
    
    if st.button('Run'):
        result = prediction(loc,vg,i, v, d)
        if(result[0]==0):
            text = 'no defects'
        else:
            text = 'defects'
        st.success('The software has {}'.format(text))

if __name__=='__main__':
    main()




