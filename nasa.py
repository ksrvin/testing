import pandas as pd
import numpy as np
import joblib
import streamlit as st
import sklearn
import xgboost
from PIL import Image

model = joblib.load('nasa.pkl')


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
    st.write('Info about the data [Link] (https://github.com/ksrvin/testing/blob/main/about%20JM1%20Dataset.txt)')
    
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
    
    image = Image.open('output.png')

    st.sidebar.image(image, caption='Confusion Matrix')

    st.subheader('Parameters')

    loc = st.number_input('McCabe Line Count of Code',value = 0)
    vg = st.number_input('McCabe Cyclomatic Complexity',value= 0)
    i = st.number_input('Halstead Intelligence',value=0)
    v = st.number_input('Halstead Volume',value=0)
    d = st.number_input('Halstead Difficulty',value=0)
    
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




