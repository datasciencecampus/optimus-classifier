import pandas as pd
import numpy as np
import re
import fasttext as ft
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import save_model, load_model
import glob
from sklearn.metrics import confusion_matrix, classification_report
import os
import time
import sys
import json

#Processing data functions
#get a list of key words from a text file
#A function to clean a list
def clean_list(list_words):
    #clean a list of words (rather than file) removes punctuation,
    #specific strings, and makes words lowercase

    cleaned_descriptions = []
    pairs = {
    "[a-z]{2}[0-9]{2}[a-z]{3} ":" ",
    "[a-z]{2}[0-9]{2}[a-z]{3}$":" ",
    "[a-z]{2}[0-9]{3}[a-z]{2}":" ",
    "[a-z]{2}[0-9]{4}" : " ",
    "()/\"": " ",
    r"[^a-z\s]": " ",
    r"\b[a-z]\b": " ",
    "lofo": "",
    "mdso": "",
    "\s+": " ",
    "chilled": "c",
    "ambient": "a",
    "frozen": "f",
    "restart": "r"
    }


    for desc in list_words:
        desc = str(desc).lower()
        for this, that in pairs.items():
            desc = re.sub(this, that, desc).strip()

        print(desc)

        cleaned_descriptions.append(desc)

    return cleaned_descriptions

#training functions
#embed words for training
def embed(list_of_words):
    #loads model, get_word_vectors, embed in lsit of words then delete model

    ft_model = ft.load_model("./model/wiki.en.bin")

    embeddings = []

    for desc in list_of_words:
        w = ft_model.get_word_vector(desc)
        embeddings.append(w)

    del ft_model

    vectors = []

    for x in embeddings:
        vectors.append([*x])

    vectors = np.array(vectors)

    return vectors

#decode predictions back to assinged string
def label_decoder(preds):
    #decode predicts to categories

    output_argmax = []
    for x in preds:
        output_argmax.append(np.argmax(x))

    return output_argmax

#load the model
def load_model_pack(file):
    #load model and label encoders from folder

    model_p = "./trained_models/" + file
    model = load_model(model_p, compile = True)
    labels_path = "./label_encoders/" + file + ".csv"

    labels = pd.read_csv(labels_path)
    labels_list = labels.values.tolist()

    return model, labels_list

#predicting functions
#Generate prediction labels
def prediction_labels(df, model_name):
    #generate predictions by loading model, creating embdedings of preducitons, predictin and the decoding,
    #load model
    model, labels = load_model_pack('main_model_' + model_name)

    labels = list(labels)

    descriptions = df

    descr = list(descriptions['original'])
    desc = clean_list(descr)

    # get embedding vectors for cleaned descriptions
    v = embed(desc)

    preds = model.predict(v)

    res = label_decoder(preds)

    prediction_labels = []

    for x in res:

        prediction_labels.append(labels[x][0])

    output_df = pd.DataFrame(zip(descr, prediction_labels))
    output_df.columns = ['original', 'preds']
    return output_df

#separate certain labels from the main dataset
def seperate(df, label):
    df1 = df[df['preds'].isin([label])]
    df2 = df[~df['preds'].isin([label])]
    return (df1, df2)

#all prediction functions put into one function

def rulebase(data_frame):

    ## uses a json to make changes post classifications - can fix issues
    ## with predicting on items not in training set
    data = data_frame
    config = json.load(open("./processing/processing.json"))

    for key, value in config["string"].items():

        data['preds'] = pd.np.where(data.original.str.contains(key), value, data['preds'])

    return data

def remove_encodings(df, char):
    df = df.dropna(how = 'any')
    df['original_not_clean'] = df['original']
    df['original'] = df['original'].str.replace('-', '') #clean for processing

    df_filter = df[df['original'].str.startswith(char)] #seperate descriptions with character
    df_not = df[~df['original'].str.startswith(char)]

    df_filter['original'] =  [re.sub('.*/', '', str(word)) for word in df_filter['original']]

    frames = [df_not, df_filter]
    df_both = pd.concat(frames)

    df_white_space = df[df['original'].str.startswith(char)]

    df_not_white_space = df[~df['original'].str.startswith(char)]

    df_white_space['original'] = [re.sub('\d+', '', str(word)) for word in df_white_space['original']]
    if char == 'XA':
        df_white_space['original'] = df_white_space['original'].str[4:]
    else:
        df_white_space['original'] = df_white_space['original'].str[2:]

    frames2 = [df_white_space, df_not_white_space]
    df_complete = pd.concat(frames2)
    df_complete['original'] = df_complete['original'].str.replace('/', '')

    return df_complete

def predict(filepath):
    data = pd.read_csv(filepath, encoding = "UTF-8")
    #data = remove_encodings(data, 'XA')
    #data = remove_encodings(data, 'AF')

    chemicals_df = prediction_labels(data, "chemicals_vs_not_chemicals")
    chemicals_df, not_chemicals_df = seperate(chemicals_df, "Chemicals")

    petrol_df = prediction_labels(chemicals_df, "chemicals_vs_petrol")
    other_labels = prediction_labels(not_chemicals_df, "all_labels_dataset")

    all_labels = pd.concat([petrol_df, other_labels])

    animals_df, not_animals_df = seperate(all_labels, "Animals")
    animals_df = prediction_labels(animals_df, "animals_relabelled")
    all_labels = pd.concat([animals_df, not_animals_df])

    general_df, not_general_df = seperate(all_labels, "General")
    pallets_df = prediction_labels(general_df, "pallets_relabelled")
    all_labels = pd.concat([pallets_df, not_general_df])

    all_labels.to_csv(filepath[:-4] + "_predictions.csv", encoding = "UTF-8", quotechar = "\"")

    # apply the rulebase - and write out new output
    post_processed = rulebase(all_labels)

    post_processed.to_csv(filepath[:-4] + "_predictions_pp.csv", encoding = "UTF-8", quotechar = "\"")
