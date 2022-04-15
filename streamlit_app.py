import sqlite3
import streamlit as st

import pandas as pd
import numpy as np
import pickle
import plotly.express as px

from skimage.transform import resize

## App Title
st.markdown(
    """
    ## Machine Learning Classification Example
    ### Predict the type of fish based on its shape
    """)

## Script Parameter
data_path = './data/fish.csv'
db_path = './data/FishDB.sqlite'
class_names = ['Bream', 'Roach', 'Whitefish', 'Parkki', 'Perch', 'Pike', 'Smelt']

model_paths = {
                'Random Forest':'./models/rf_fish_trained.pkl',
                'Multinomal Logit':'./models/logit_fish_trained.pkl'
            }

model_table_names ={
                'Random Forest' : 'RF',
                'Multinomal Logit' : 'Logit'
                }

fish_color_palette =  {
    'Bream':'#a6cee3', 
    'Roach':'#1f78b4',
    'Whitefish':'#b2df8a', 
    'Parkki':'#33a02c',
    'Perch':'#fb9a99',
    'Pike':'# e31a1c',
    'Smelt':'#fdbf6f'
}

## Script Functions
@st.cache
def load_fish_data(data_path):
    """
        Import Original Fish Dataset and preprocess data
    """
    fish_df = pd.read_csv(data_path)
    # just keep one length column as AVG of original length columns
    fish_df['Length'] = fish_df.loc[: , ['Length1', 'Length2','Length3']].mean(axis=1)
    
    fish_df= fish_df.drop(['Length1', 'Length2','Length3'], axis =1)

    return fish_df

@st.cache
def load_predictions(db_path, features):
    """
        Import stored model predictions from sqlite DB
    """

    feature_cols_sql_string = ''
    for feature in features:
        feature_cols_sql_string = feature_cols_sql_string + 'f.' + feature + ','

    with sqlite3.connect(db_path) as con:
        predictions_df = pd.read_sql("""SELECT 
                                            p.Model AS Model,
                                            """ + feature_cols_sql_string + """
                                            p.Predicted_Species as Species
                                        FROM predictions p
                                        JOIN fishes f ON p.Fish_ID = f.Fish_ID
                                        """, con)
    
    return predictions_df

@st.cache
def load_fish_image():
    """
        Import original size fish image as array
    """
    
    with open('./img/fish_array.npy', 'rb') as f:
        fish_img_array = np.load(f)
    return fish_img_array


def scale_fish_to_input(orig_img_array, user_input_length, user_input_height):
    """
        Use original fish image from numpy array, 
            scale it to the widht and height of user input,
            add it to the fixed canvas size of image,
            return array of scaled fish on canvas.
    """
    # hard coded values as they should never change
    max_fish_length = 63.47
    max_fish_height = 18.96
    
    scale_y = user_input_height / max_fish_height
    scale_x = user_input_length / max_fish_length

    new_y = int(orig_img_array.shape[0] * scale_y)
    new_x = int(orig_img_array.shape[1] * scale_x)
    
    scaled_img = resize(orig_img_array, (new_y, new_x))
    scaled_img = (scaled_img * 255).astype(np.uint8)
    #scaled_img = cv2.resize(orig_img_array, (new_x, new_y))
    
    ## hard coded as shape of canvas should never change
    canvas = np.zeros((567, 1899), dtype=np.uint8)
    
    # make sure the smaller img is added to the middle
    img_max_y = scaled_img.shape[0]
    img_max_x = scaled_img.shape[1]
    
    canvas_y_idx = int((canvas.shape[0] - scaled_img.shape[0]) / 2)
    canvas_x_idx = int((canvas.shape[1] - scaled_img.shape[1]) / 2)
    
    # only add to slice of canvas
    canvas[canvas_y_idx:(canvas_y_idx + img_max_y), 
                   canvas_x_idx:(canvas_x_idx + img_max_x)] += scaled_img
    
    return canvas


## Data Import
fish_df = load_fish_data(data_path)
features = fish_df.drop('Species', axis=1).columns

predictions_df = load_predictions(db_path, features)

## User Input Fish
X_input_dict = {}

filter_elements = st.columns(len(features))

for i, feature in enumerate(features):
    slider_text = feature + " (in g)" if feature == 'Weight' else feature + " (in cm)"

    with filter_elements[i]:
        input_value = st.slider(slider_text, 
                                        min_value=0.1, 
                                        max_value=fish_df[feature].max(),
                                        value=float(fish_df[feature].median()))

    X_input_dict[feature] = [input_value]

X_input = pd.DataFrame(X_input_dict)


## Import Image
#https://openclipart.org/detail/213228/fish-icon
fish_img_array = load_fish_image()

scaled_fish = scale_fish_to_input(fish_img_array, 
                                    user_input_length = X_input.iloc[0, 3], 
                                    user_input_height = X_input.iloc[0, 1])

st.image(scaled_fish)


st.markdown("""
### Model Prediction
""")

model_class = st.selectbox("Prediction Algorithm", model_paths.keys())
model_path = model_paths[model_class]

## Prediction
# Load prediction model
prediction_model = pickle.load(open(model_path, 'rb'))

prediction_proba = prediction_model.predict_proba(X_input)[0]
proba_df = pd.DataFrame({
    'Species': class_names,
    'Probability': prediction_proba
})

## Prediction Probability Plot
fig = px.bar(proba_df, 
                x="Species", 
                y="Probability", 
                color="Species", 
                color_discrete_map=fish_color_palette,
                title="Predicted Species")
fig.update_layout(yaxis_range=[0,1])
st.plotly_chart(fig)


st.markdown("""
### Feature Exploration
"""
) 

feat_filter_1, feat_filter_2 = st.columns(2)
with feat_filter_1:
    feature_1 = st.selectbox("X-Axis", features)
with feat_filter_2:
    feature_2 = st.selectbox("Y-Axis", features[::-1])

model_table_name = model_table_names[model_class]
model_predictions = predictions_df.loc[predictions_df.Model == model_table_name]

## Feature Exploration Plot
fig = px.scatter(model_predictions, 
                    x=feature_1, 
                    y=feature_2, 
                    color='Species', 
                    color_discrete_map=fish_color_palette,
                    opacity=0.5,
                    title="Model Predictions on full Dataset")
fig.update_traces(marker_size=20)
st.plotly_chart(fig)
