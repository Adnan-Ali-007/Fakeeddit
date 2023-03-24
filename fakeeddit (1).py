import re
import os
import numpy as np
from matplotlib import pyplot as plt
# Importing libraries
import numpy as np
import pandas as pd
import re
import nltk
from tensorflow import keras 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Activation, BatchNormalization, Dropout, Bidirectional
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import chardet 
#Lines to run the code on GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from keras.backend import set_session
from tensorflow.python.keras import backend as K
tf.autograph.set_verbosity(0)
sess = tf.compat.v1.Session()
K.set_session(sess)
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True
#set_session(tf.compat.v1.Session(config=config))
#inputing data
df_test=pd.read_csv ('/sda/rina_1921cs13/fakenewseddit/data/multimodal_test_public.tsv',sep = '\t', encoding='utf-8')
df_test.head()
#df=pd.read_csv ('/sda/rina_1921cs13/fakenewseddit/data/all_comments.tsv',sep = '\t')
#df.head()
df_train=pd.read_csv ('/sda/rina_1921cs13/fakenewseddit/data/multimodal_train.tsv',sep = '\t', encoding='utf-8')
df_train.head()
df_valid=pd.read_csv ('/sda/rina_1921cs13/fakenewseddit/data/multimodal_validate.tsv',sep = '\t', encoding='utf-8')
df_valid.head()
bert_preprocess = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')
# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='clean_title')
#merging the comments in 1 para comment input in bert
preprocessed_text = bert_preprocess(text_input)
#preprocessed_text.keys()
outputs = bert_encoder(preprocessed_text)#outputs for commnt
preprocessed_text['input_mask']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min', verbose=2, patience=5)
print(df_train['2_way_label'].unique())
print(df_train['3_way_label'].unique())
print(df_train['6_way_label'].unique())
# BERT + Bi_LSTM + No_Attention
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output']) #create outputs comment
l = tf.keras.layers.Reshape((1,768))(l)
#l = tf.keras.layers.Bidirectional(LSTM(256, return_sequences=True, activation='relu'))(l)
l = tf.keras.layers.LSTM(128, activation='tanh', name='lstm', return_sequences=False)(l) 
l = tf.keras.layers.Dense(32, activation='tanh', name='dense')(l) #take another by using m
#l = tf.keras.layers.Flatten()(l) #use connactation layer input with 2arg l and m
l = tf.keras.layers.Dense(2, activation='sigmoid', name='output')(l) #conncataneted rep here
# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])
model.summary()
#df_train = pd.read_csv("/sda/rina_1921cs13/fakenewseddit/data/multimodal_train.tsv",sep='\t', encoding='utf-8')
#df_test = pd.read_csv("/sda/rina_1921cs13/fakenewseddit/data/multimodal_test_public.tsv",sep='\t', encoding= 'utf-8')
#df_valid = pd.read_csv("/sda/rina_1921cs13/fakenewseddit/data/multimodal_validate.tsv",sep='\t',encoding= 'utf-8')
x_train = df_train['clean_title']
x_valid = df_valid['clean_title']
x_test = df_test['clean_title']
y_train = df_train['2_way_label']
y_valid = df_valid['2_way_label']
y_test = df_test['2_way_label']
x_train.head(4)
print(x_train.shape)
y_train.value_counts()
print(y_train.shape)
y_test.value_counts()
print("Shape of y_train:", y_train.shape)
print("Data type of y_train:", type(y_train))
print("First 10 elements of y_train:", y_train[:10])
import numpy as np
from tensorflow.keras.utils import to_categorical
print(np.any(np.isnan(y_train)))
print(np.all(np.isfinite(y_train)))
y_train = y_train.replace(np.nan, 0)
#y_train = y_train.dropna()
y_train = y_train.astype('int')
y_train = to_categorical(y_train)
y_valid = y_valid.replace(np.nan, 0)
#y_train = y_train.dropna()
y_valid = y_valid.astype('int')
y_valid = to_categorical(y_valid)
y_test = y_test.replace(np.nan, 0)
#y_train = y_train.dropna()
y_test = y_test.astype('int')
y_test = to_categorical(y_test)
#shape checking
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
input_shape = model.layers[0].input_shape
print("Expected input shape:", input_shape)
output_shape = model.layers[-1].output_shape
print("Expected output shape:", output_shape)
####################Compilation of the model###############################
from tensorflow.keras.optimizers import Adam
METRICS=[tf.keras.metrics.BinaryAccuracy(name="accuracy"),
         tf.keras.metrics.Precision(name="precision"), 
         tf.keras.metrics.Recall(name="recall"),
         tf.keras.metrics.BinaryAccuracy(name="val_accuracy"),
         tf.keras.metrics.BinaryCrossentropy(name="loss"),                   tf.keras.metrics.BinaryCrossentropy(name="val_loss")
         ]
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=METRICS)
history = model.fit(x_train, y_train,validation_split=0.3,batch_size=32, epochs=50,callbacks=[es],shuffle=True,verbose=2)
#model.evaluate(x_valid,y_valid,verbose=2)
from sklearn.metrics import confusion_matrix
y_predicted=model.predict(x_test)
y_predicted = np.argmax(y_predicted, axis=1)
print("y_ predicted:", y_predicted.shape)
#adjusting for confusion matrix shape
y_test = np.argmax(y_test, axis=1)
# y_predicted=y_predicted.flatten()
con=confusion_matrix(y_test, y_predicted)
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
precision = precision_score(y_test, y_predicted, average='micro')
recall = recall_score(y_test, y_predicted, average='micro')
f1 = f1_score(y_test, y_predicted, average='micro')
acc = accuracy_score(y_test, y_predicted)
print(precision)
print(recall)
print(f1)
print(acc)
from sklearn.metrics import classification_report
unique_labels = np.unique(y_test)
target_names = [str(label) for label in unique_labels]
print(classification_report(y_test, y_predicted, target_names=target_names))
import seaborn as sns
import matplotlib.pyplot as plt  
from pylab import savefig   
fig, ax = plt.subplots(figsize =(9, 9))
dens_mat = sns.heatmap(con, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['fake','real'],rotation=45); ax.yaxis.set_ticklabels(['0','1'],rotation=45);
figure = dens_mat.get_figure()    
figure.savefig('', dpi=400)
plt.figure(figsize=(5,4))
plt.plot(history.history['accuracy'],label = 'accuracy', color ='b')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy', color ='r')
plt.title('Model Accuracy Curves')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy','val_accuracy'], loc='upper left') 
#fixing error for path
import os
directory = "/sda/rina_1921cs13/fakenewseddit/data"
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig('/sda/rina_1921cs13/fakenewseddit/data/accuracy_curves.png', dpi=400)
plt.figure(figsize=(5,4))
plt.plot(history.history['loss'],label = 'loss', color ='b')
plt.plot(history.history['val_loss'],label = 'val_loss', color ='r')
plt.title('Model Loss Curves')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Loss','Val_Loss'], loc='upper left')
plt.savefig('/sda/rina_1921cs13/fakenewseddit/data/loss_curves.png', dpi=400)

