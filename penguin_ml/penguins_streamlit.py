import streamlit as st
import pickle

st.write("""
## Penguin Classifier
##### This app uses 6 inputs to predict the species of penguin using a model built on the Palmer's Penguins dataset.
##### Use the form below to get started! 
""")

rf_pickle = open("random_forest_penguin.pickle", "rb") # the random forest classifier model
# rb indicates that the file is opened for reading in binary mode.
map_pickle = open("output_penguin.pickle", "rb") # the file which contains our output

rfc = pickle.load(rf_pickle) # load the random forest classifier model
unique_penguin_mapping = pickle.load(map_pickle) # load the file which contains our output

rf_pickle.close()
map_pickle.close()

#Let's print the files that we just loaded and check if they are the correct files.
# st.write(f"Model:\n\n {rfc}")
# st.write(f"Output values (Species):\n\n {unique_penguin_mapping.values}")
# Yes they are the correct files. We can comment out these code blocks.

# Add Streamlit functions to get the user input.
island = st.selectbox(label="Penguin Island", 
    options=["Biscoe", "Dream", "Torgerson"])
sex = st.selectbox(label="Sex", options=["Female", "Male"])
bill_length = st.number_input(label="Bill Length (mm)", min_value=0)
bill_depth = st.number_input(label="Bill Depth (mm)", min_value=0)
flipper_length = st.number_input(label="Flipper Length (mm)", min_value=0)
body_mass = st.number_input(label="Body Mass (g)", min_value=0)

st.write(f"#### The user inputs are:\n\n {[island, sex, bill_length, bill_depth, flipper_length, body_mass]} ")

# Now we need to put our sex and island variables into the correct format as in the penguins_ml.py file.
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == "Biscoe":
    island_biscoe = 1
elif island == "Dream":
    island_dream = 1
elif island == "Torgerson":
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == "Female":
    sex_female = 1
elif sex == "Male":
    sex_male = 1

# All of our data is in the correct format!
# The last step here is using the predict() function on our model with our new data.
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
    body_mass, island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[new_prediction][0]
st.write(f"### We predict your penguin is of the {prediction_species} species.")