import streamlit as st
import pandas as pd
import pickle

st.write("""
### Hello!
""")

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')



def get_input():
    #widgets
    p_sex = st.sidebar.radio('Sex', ['Male','Female','Infant'])
    p_gpaEng = st.sidebar.slider('GPA_Eng',0.00, 4.00, 1.00 )
    p_gpaMath = st.sidebar.slider('GPA_Math', 0.00, 4.00, 1.00 )
    p_gpaSci = st.sidebar.slider('GPA_Sci', 0.00, 4.00, 1.00 )
    p_gpaSoc = st.sidebar.slider('GPA_Sco', 0.00, 4.00, 1.00 )
    p_stuTH = st.sidebar.selectbox('Student_Th', [0,1])
    p_q3 = st.sidebar.selectbox('Q3', [0,1])
    p_q4 = st.sidebar.selectbox('Q4', [0,1])
    p_q6 = st.sidebar.selectbox('Q6', [0,1])
    p_q26 = st.sidebar.selectbox('Q26', [0,1])
    p_q28 = st.sidebar.selectbox('Q28', [0,1])

    

    if p_sex == 'Male': p_sex = 'M'
    elif p_sex == 'Female': p_sex = 'F'
    else: p_sex = 'I'

    #dictionary
    data = {'GPA_Eng': p_gpaEng,
            'GPA_Math': p_gpaMath,
            'GPA_Sci': p_gpaSci,
            'GPA_Sco': p_gpaSoc,
            'Sex': p_sex,
            'Student_Th': p_stuTH ,
            'Q3': p_q3,
            'Q4': p_q4,
            'Q6': p_q6,
            'Q26': p_q26,
            'Q28': p_q28,
    }

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('new_sample_tcas.xlsx', encoding='latin-1')
df = pd.concat([df, data_sample],axis=0)

cat_data = pd.get_dummies(df[['Sex']])

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)
#Drop un-used feature
X_new = X_new.drop(columns=['Sex'])


# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

# -- Reads the saved classification model
load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)