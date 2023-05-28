# streamlit4.py
import streamlit as st
import pandas as pd
import numpy as np
import time
# import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk                                         #Natural language processing tool-kit
from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

@st.cache(suppress_st_warning=True)
def data_clean1(data):
  # Ensure that 1st column is text column
  txt = data.iloc[:,0].name
  data = data.drop_duplicates(subset=[txt], ignore_index=True)
  data = data.dropna()
  data.iloc[:,1] = data.iloc[:,1].replace({'I': 1, 'II': 2,'III': 3,'IV': 4,'V': 5,'VI':6})
  return data

# @st.cache(suppress_st_warning=True)
def nlp_preprocess(sentence, stopwords,lemmer):
  import nltk
  import string
  import re
  sentence = sentence.lower()                 # Converting to lowercase
  sentence = re.sub(r'\d+', '', sentence)     # Removing numbers
  sentence = re.sub(r'[^\w\s]', '', sentence)
  #sentence = sentence.translate(string.maketrans('',''), string.punctuation) # Remove punctuations
  sentence = ' '.join(sentence.split()) # Remove whitespaces
  tokens = nltk.word_tokenize(sentence)  # Create tokens
  output = [i for i in tokens if not i in stopwords]  # Remove stop words
  words = [lemmer.lemmatize(word) for word in output] # Lemmatize words
  sentence = ' '.join(words)
  return sentence

# @st.cache(suppress_st_warning=True)
def preprocess_func(train_text1,test_text1):
  stop_words = set(stopwords.words('english'))
  lemmer = nltk.stem.WordNetLemmatizer()
  train_text2 = train_text1.apply(lambda x: nlp_preprocess(x,stop_words,lemmer))
  test_text2 = test_text1.apply(lambda x: nlp_preprocess(x,stop_words,lemmer))
  return train_text2, test_text2

def result_format(train_y,train_pred, test_y,test_pred):
  from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
  a = accuracy_score(train_y, train_pred) # Train accuracy
  b = accuracy_score(test_y, test_pred) # Test accuracy
  c = f1_score(test_y, test_pred, average='weighted',zero_division=0) # weighted F1
  d = f1_score(test_y, test_pred, average=None, zero_division=0) # list of F1s
  d = [ round(elem, 2) for elem in d ]
  e = precision_score(test_y, test_pred, average=None, zero_division=0) # Precision scores
  e = [ round(elem, 2) for elem in e ]
  f = recall_score(test_y, test_pred, average=None, zero_division=0) # Recall scores
  f = [ round(elem, 2) for elem in f ]

  return [a,b,c,d,e,f]

def get_results(x_train,x_test, train_y, test_y,mod,param):
#def get_results(train_y,train_pred, test_y,test_pred):
  from sklearn.model_selection import StratifiedShuffleSplit
  from sklearn.model_selection import RandomizedSearchCV
  try:
    #s_split = StratifiedShuffleSplit(n_splits=3,test_size=0.2,random_state=108)
    model = RandomizedSearchCV(mod,param_distributions=param,cv=5, scoring='f1_weighted', random_state=108)
  except:
    model = mod
  model.fit(x_train, train_y)
  train_pred = model.predict(x_train)
  test_pred = model.predict(x_test)
  scores = result_format(train_y,train_pred,test_y,test_pred)
  return scores, model


# @st.cache(suppress_st_warning=True)
def r_f(x_train,x_test, y_train, y_test):
  from sklearn.ensemble import RandomForestClassifier
  runtime = time.time()
  model = RandomForestClassifier(max_depth=4,max_features=4)
  param = {'class_weight':['balanced'],
              'random_state':[108],
              'n_estimators':[100,200,300],
              'max_depth':[5,7,9,11]
            }
  #rf.fit(x_train,y_train)
  #scores = get_results(y_train,rf.predict(x_train),y_test,rf.predict(x_test))
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def LogR(x_train,x_test, y_train, y_test):
  from sklearn.linear_model import LogisticRegression
  param = {'penalty':['l1','l2'],
            'C':[0.01,0.1,1,10,100],
            'solver':['saga']}
  model = LogisticRegression(max_iter=1000)
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def GNB(x_train,x_test, y_train, y_test):
  from sklearn.naive_bayes import GaussianNB
  model = GaussianNB()
  param = {'var_smoothing':[10**-8,10**-9,10**-10]}
  runtime = time.time()
  x_train = x_train.toarray()
  x_test = x_test.toarray()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def KNC(x_train,x_test, y_train, y_test):
  from sklearn.neighbors import KNeighborsClassifier
  model = KNeighborsClassifier()
  param = {'n_neighbors':[5,9,13,17,21,25,29,33,35],
            'weights':['distance']}
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def SVC(x_train,x_test, y_train, y_test):
  from sklearn.svm import SVC
  model = SVC()
  param = {'C':range(1,100,10),
          'gamma':[0.1, 0.5, 0.9, 1,2,3],
          # 'class_weight':['balanced'],
            "kernel" : ['rbf']
          }
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def DTC(x_train,x_test, y_train, y_test):
  from sklearn.tree import DecisionTreeClassifier
  model = DecisionTreeClassifier()
  param = {'criterion':['gini','entropy'],
          'max_depth':[5,7,9,11],
          'class_weight':['balanced']}
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def GBC(x_train,x_test, y_train, y_test):
  from sklearn.ensemble import GradientBoostingClassifier
  model = GradientBoostingClassifier()
  param = {'learning_rate':[0.01,0.05,0.1,0.2],
            'random_state':[108],
            'n_estimators':[100,200]
          }
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

def XGB(x_train,x_test, y_train, y_test):
  from xgboost import XGBClassifier
  model = XGBClassifier(random_state=108)
  param = { 'objective':['reg:squarederror', 'binary:logistic'],
      #logistic regression for binary classification, output probability
            'learning_rate': np.linspace(uniform.ppf(0.01), uniform.ppf(0.09), 20), # or `eta` value
            'max_depth': [6,10, 20, 30], # avoid 0
            'min_child_weight': [2,3],
            'subsample': [0.8,0.9,1],
            # 'colsample_bytree': [0.8,0.9,1]
            }
  runtime = time.time()
  scores, model = get_results(x_train,x_test, y_train, y_test,model,param)
  g = time.time() - runtime
  scores.append(g)
  return scores, model

# @st.cache(suppress_st_warning=True)
def n_n(x_train,x_test, y_train, y_test):
  from tensorflow.keras.layers import Activation, Dense, Dropout, InputLayer, BatchNormalization
  from tensorflow.keras import optimizers, initializers, losses
  from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPool1D, Bidirectional, LSTM, Dropout, SpatialDropout1D
  from tensorflow.keras.models import Sequential
  from tensorflow.compat.v1.keras.layers import CuDNNLSTM
  from tensorflow.keras.initializers import Constant
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

  runtime = time.time()
  n_layers = 5
  neurons = 20
  act='relu'
  bat=True
  dout=True
  drop=0.2
  lr=0.001
  
  in_shape = x_train.toarray().shape[1]
  output = np.unique(y_train).shape[0]
  #st.write(x_train[0])
  
  clf = Sequential()
  clf.add(InputLayer(input_shape=(in_shape,)))
  for i in range(n_layers):
    clf.add(Dense(neurons))
    if bat:
      clf.add(BatchNormalization()) 
    clf.add(Activation(act))
    if dout:
      clf.add(Dropout(drop))
    # neurons=neurons-50
  clf.add(Dense(output, activation='softmax'))
  adam = optimizers.Adam(learning_rate=lr)
  clf.compile(optimizer=adam, loss = 'categorical_crossentropy', metrics= ['accuracy'])
  history = clf.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=5)
  train_pred = clf.predict(x_train)
  test_pred = clf.predict(x_test)
  scores = result_format(y_train,train_pred,y_test,test_pred)
  g = time.time() - runtime
  scores.append(g)
  return scores, clf

# @st.cache(suppress_st_warning=True)
def lstm(x_train,x_test, y_train, y_test):
  from tensorflow.keras.layers import Activation, Dense, Dropout, InputLayer, BatchNormalization
  from tensorflow.keras import optimizers, initializers, losses
  from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPool1D, Bidirectional, LSTM, Dropout, SpatialDropout1D
  from tensorflow.keras.models import Sequential
  from tensorflow.compat.v1.keras.layers import CuDNNLSTM
  from tensorflow.keras.initializers import Constant
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

  runtime = time.time()
  embed_dim = 128
  lstm_out = 150
  max_vocab = 10000
  inp_len = x_train.toarray().shape[1]
  out = np.unique(y_train).shape[0]

  model = Sequential()
  model.add(Embedding(max_vocab,embed_dim,input_length = inp_len))
  model.add(SpatialDropout1D(0.6))
  model.add(LSTM(lstm_out, dropout=0.5, recurrent_dropout=0.6))
  # model.add(Dropout(0.5))
  # model.add(LSTM(lstm_out-30, dropout=0.5, recurrent_dropout=0.5))
  model.add(Dense(out,activation='softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
  history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)
  train_pred = model.predict(x_train)
  test_pred = model.predict(x_test)
  scores = result_format(y_train,train_pred,y_test,test_pred)
  g = time.time() - runtime
  scores.append(g)

  return scores, model

def bilstm(x_train,x_test, y_train, y_test):
  from tensorflow.keras.layers import Activation, Dense, Dropout, InputLayer, BatchNormalization
  from tensorflow.keras import optimizers, initializers, losses
  from tensorflow.keras.layers import Embedding, Dense, GlobalMaxPool1D, Bidirectional, LSTM, Dropout, SpatialDropout1D
  from tensorflow.keras.models import Sequential
  from tensorflow.compat.v1.keras.layers import CuDNNLSTM
  from tensorflow.keras.initializers import Constant
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

  runtime = time.time()
  inp_len = x_train.toarray().shape[1]
  out = np.unique(y_train).shape[0]
  max_vocab = 10000
  embed_dim = 128
  biLSTM = Sequential()

  biLSTM.add(Embedding(max_vocab, embed_dim,
                      input_length= inp_len))
  biLSTM.add(Bidirectional(LSTM(units=32, recurrent_dropout = 0.5, dropout = 0.5, 
                              return_sequences = True)))
  biLSTM.add(Bidirectional(LSTM(units=16, recurrent_dropout = 0.5, dropout = 0.5)))
  biLSTM.add(Dense(out, activation='softmax'))

  biLSTM.compile(optimizer=optimizers.Adam(lr = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  history = biLSTM.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32)
  
  train_pred = biLSTM.predict(x_train)
  test_pred = biLSTM.predict(x_test)
  scores = result_format(y_train,train_pred,y_test,test_pred)
  g = time.time() - runtime
  scores.append(g)

  return scores, biLSTM

# @st.cache(suppress_st_warning=True)
def tt_split(data,split):
  from sklearn.model_selection import train_test_split
  train_text,test_text,train_label,test_label = train_test_split(data.iloc[:,0],data.iloc[:,1],test_size = split/100,
                                                                  stratify=data.iloc[:,1], random_state=42)
  #st.write("\nTrain-test split done:")
  return train_text,test_text,train_label,test_label

# @st.cache(suppress_st_warning=True)
def encode_labels(train_label,test_label):
  # 5. One hot encoding of labels
  from sklearn import preprocessing
  lb = preprocessing.LabelEncoder()
  #lb = preprocessing.LabelBinarizer()
  y_train = lb.fit_transform(train_label)
  y_test = lb.transform(test_label)
  return y_train,y_test, lb

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def vectorize(Vect, train_text_,test_text_):
  if Vect == 'tf-idf':
    Vector = TfidfVectorizer()
  elif Vect == 'CountVec':
    Vector = CountVectorizer()
  else:
    Vector = glove()
  x_train = Vector.fit_transform(train_text_)
  x_test = Vector.transform(test_text_)
  return Vector, x_train, x_test

def preprecess_chatbot(sentence,vector):
  stop_words = set(stopwords.words('english'))
  lemmer = nltk.stem.WordNetLemmatizer()
  sentence1 = nlp_preprocess(sentence,stop_words,lemmer)
  output = vector.transform([sentence1])
  return output

def prepare_trans_df(train_df,text,labels):
  file_add = 'https://raw.githubusercontent.com/mhtkmr1/Industrial_safety_chatbot/main/Google_translated_data.csv'
  google_translated = pd.read_csv(file_add)
  google_translated.drop_duplicates(subset=[text],inplace=True, ignore_index=True) # remove duplicate rows
  google_translated.set_index(text,inplace=True)

  train_df = train_df.set_index(text)
  clms = ['es', 'hi', 'it', 'Id', 'ja', 'he','ga', 'de', 'fr']
  dtf = pd.concat([train_df, google_translated.loc[train_df.index,clms]],axis=1)
  dtf.reset_index(inplace=True)

  dtf1 = dtf[[text,labels]].copy(deep=True)
  for lang in clms:
    new_df = pd.DataFrame()
    new_df[text] = dtf[lang]
    new_df[labels] = dtf[labels]
    dtf1 = pd.concat([dtf1, new_df],ignore_index=True)
  #dtf1.drop_duplicates(subset=[text],inplace=True, ignore_index=True) # remove duplicate rows
  # Function output contains some duplicates which can be removed later
  return dtf1

def syn_augmentor(text_df,text,labels):
  import nlpaug
  import nlpaug.augmenter.char as nac
  import nlpaug.augmenter.word as naw
  import nlpaug.augmenter.sentence as nas
  values = text_df[labels].value_counts().values
  levels = text_df[labels].value_counts().index
  augmented_sen = []
  level = []
  for i in range(1,len(levels)):
    data = text_df[text_df[labels] == levels[i]]
    for dt in data[text]:
      sent = dt
      for k in range(values[0]//values[i]):
        if len(sent) < 10:
          wrd_aug = naw.SynonymAug(aug_min=3)
          sent = wrd_aug.augment(sent)
        elif len(sent) > 10 and len(sent) < 25:
          wrd_aug = naw.SynonymAug(aug_min=5)
          sent = wrd_aug.augment(sent)
        else:
          wrd_aug = naw.SynonymAug(aug_min=8)
          sent = wrd_aug.augment(sent)

        augmented_sen.append(sent)
        level.append(levels[i])

  desc = pd.concat([text_df[text],pd.Series(augmented_sen)])
  acc_lvl = pd.concat([text_df[labels], pd.Series(level)])
  aug_df = pd.concat([desc,acc_lvl],axis=1)
  aug_df.reset_index(drop=True, inplace=True)
  aug_df.columns = [text,labels]
  return aug_df
# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
#def load_model():
  #final_model = joblib.load('final_model.pkl')
  #return final_model

#st.write("_" * 30)
# >>>>>>>>>>>>>>>>

def main():
  st.set_page_config(layout='wide')
  #st.write(st.session_state)
  hide_menu_style = """
                  <style>
                  #MainMenu {visibility:hidden;}
                  footer {visibility:hidden;}
                  </style>
                  """
  st.markdown(hide_menu_style, unsafe_allow_html = True)
  st.title('INDUSTRIAL SAFETY CHATBOT')
  #separate into 3 columns
  st.session_state.col_size = [1,2,3]
  col11, col12, col13 = st.columns(st.session_state.col_size)
  with col11:
    st.header("STEPS")
    st.write("**Step-1: Upload the data file:**")
  
  # Provide button in sidebar to clear all cache and restart the app
  #if st.sidebar.button('Press here to restart the app'):
  #  for key in st.session_state.keys():
  #    del st.session_state[key]

  with col12:
    st.header("INPUT")
    uploaded_file = st.file_uploader(
        label="Select and upload the csv file containing the data from your system:",
        type="csv",
        accept_multiple_files=False,
        key = 'file',
        help='''Upload a csv file that contains your chabot corpus.     
            required structure:     
            one column with text data or accident description
            and another column with Accident levels or categories
            first row = column headers
            ''')
    uploaded_file1 = st.button('Press to use an example data from repo',key='uploaded_file1')
  with col13:
    st.header("OUTPUT AND STATUS")
  
  if uploaded_file1 and not uploaded_file:
    #st.write('value of uploaded_file1 parameter is: ',uploaded_file1)
    file_url='https://raw.githubusercontent.com/mhtkmr1/Industrial_safety_chatbot/main/Data%20Set%20-%20industrial_safety_and_health_database_with_accidents_description.csv'
    st.session_state.df = pd.read_csv(file_url)
    with col13:
      st.write('**NOTE: Currently the example file is being used**')
  if uploaded_file:
    #if 'df' in st.session_state: del st.session_state[df] # Remove df if already exists, generally in case of new file uploaded
    st.session_state.df = pd.read_csv(uploaded_file)
    with col13:
      st.write('FILENAME: ', uploaded_file.name)
      st.write('FILETYPE: ', uploaded_file.type)
  if 'df' in st.session_state:
    with col13:
      st.success('File uploaded successfully')
      st.write('**Shape of original data: **',st.session_state.df.shape)
      #st.dataframe(st.session_state.df.head())
    col11_, col12_ = st.columns([1,5])
    with col12_:
      st.write('**Preview of the uploaded dataframe**')
      st.dataframe(st.session_state.df.head())
      st.write("_" * 30)
    
    if 'df' in st.session_state:
      col21, col22, col23, col24 = st.columns([1,1,1,3])
      columns = st.session_state.df.columns
      with col21:
        st.write("**Step-2: Select the columns with text and labels:**")
      with col22:
        text = st.radio('Select column with text data',(columns), key = 'text')
      with col23:
        labels = st.radio('Select column with labels',(columns), key = 'labels')
      with col24:

        st.write('Selected columns are:')
        st.write('**Text column: **',st.session_state['text']) 
        st.write('**Label column: **',st.session_state['labels'])
      st.write("_" * 30)
    
      if 'text' in st.session_state and 'labels' in st.session_state:
        col31, col32, col33 = st.columns(st.session_state.col_size)
        with col31:
          st.write("**Step-3: Perform Data cleaning:**")
        with col32:
          cleaning = st.button(label = ' Press here for Data cleaning and pre-processing', key = 'cleaning')
        if cleaning:
          for keys in ['y_train']: # remove all keys of importance to next step
            if keys in st.session_state:
              del st.session_state[keys]
          st.session_state.df1 = data_clean1(st.session_state.df[[st.session_state.text,st.session_state.labels]]).copy(deep=True) # first column should be the text column
        if  'df1' in st.session_state:
          with col33:
            st.success('Data cleaning is complete')
            st.write('**Shape of dataframe after cleaning: **', st.session_state.df1.shape)
            st.write('Preview of Dataframe after cleaning and removing duplicates from text column: ')
            st.dataframe(st.session_state.df1.head())
        st.write("_" * 30)
      
        if 'df1' in st.session_state:
          col41, col42, col43,col44 = st.columns([1,2,1.5,1.5])
          with col41:
            st.write("**Step-4: Train - Test split and encoding of labels**")
          with col42:
            test_split = st.slider('Select percentage of test data for test-train split?', min_value=5, max_value=50,value=20, step=5, key = 'test_split')
          #test_split = st.number_input('Write the percentage of test data for test-train split? (example: write "20" for 20%)')
          #if test_split is not None and test_split > 0:
          with col43:
            st.write("**Percentage of test split you entered: **", st.session_state.test_split,"%")
            st.write('  ')
            st.write('  ')
            st.write('**Label distribution before train-test split:**')
            distribution = st.session_state.df1.iloc[:,1].value_counts()
            st.dataframe(distribution)
            check_error = distribution.index[distribution <= 1].tolist()
            
          with col42:
            perform_split = st.button(label = 'Step - 3: Perform train-test split',key = 'perform_split')
          if perform_split:
            for keys in ['aug_train_text','y_train']: # remove all keys of importance to next step
              if keys in st.session_state:
                del st.session_state[keys]              
            if check_error:
              st.error('Some of the labels have only one value, thus train-test split cannot be performed. Please change the input paramaters accordingly')
            else:
              st.session_state.train_text,st.session_state.test_text,st.session_state.train_label,st.session_state.test_label =tt_split(st.session_state.df1,test_split)
              # One hot encoding of labels
              st.session_state.y_train, st.session_state.y_test, st.session_state.lb =encode_labels(st.session_state.train_label,st.session_state.test_label)
          if 'y_train' in st.session_state:
            with col44:
              st.success('Train - Test split is complete')
              st.write('**Shape of train data after stratified split: **',st.session_state.train_text.shape)
              st.write('**Shape of test data after split: **',st.session_state.test_text.shape)
              st.write('**Label distribution of test data after split:**')
              st.dataframe(st.session_state.test_label.value_counts())
          st.write("_" * 30)

          if 'y_train' in st.session_state:
            col51, col52, col53 = st.columns([1,2,3])
            with col51:
              st.write("**Step-5: Oversampling/augmentation of train data **")
            with col52:
              perform_oversampling = st.button(label = 'Step - 4: Perform Oversampling (only for train data)',key = 'perform_oversampling')
            if perform_oversampling:
              for keys in ['train_text1']: # remove all keys of importance to next step
                if keys in st.session_state:
                  del st.session_state[keys]
              # perform the Augmentation and avoid recalculation during program reruns
              if 'aug_train_text' not in st.session_state:
                with st.spinner('Augmentation in progress, please wait...'):
                  a = prepare_trans_df(pd.concat([st.session_state.train_text, st.session_state.train_label], axis=1),st.session_state.text,st.session_state.labels)
                  b = syn_augmentor(pd.concat([st.session_state.train_text, st.session_state.train_label], axis=1),st.session_state.text,st.session_state.labels)
                  c = pd.concat([a,b],ignore_index=True)
                  c = data_clean1(c)
                  # check augmentation sanity and remove data that doesn't make sense, example: sentences that have reduced a lot
                  # Remove sentences that have become shorter than 15
                  c['sent_len'] = [len(x.split()) for x in c[st.session_state.text].tolist()]
                  c = c[c['sent_len']>15]
                  c.drop(['sent_len'], axis=1, inplace = True)
                  c.reset_index(drop=True, inplace=True)
                  st.session_state.aug_train_text = c[st.session_state.text].copy(deep=True)
                  st.session_state.aug_train_labels = c[st.session_state.labels].copy(deep=True)
                  st.session_state.y_train = st.session_state.lb.transform(st.session_state.aug_train_labels)
            if 'aug_train_text' in st.session_state:
              with col53:
                st.success('Data augmentation is complete') 
                st.write('**Shape of augmented train data: **',st.session_state.aug_train_text.shape)
                st.write('**Label distribution before train-test split:**')
                st.dataframe(st.session_state.aug_train_labels.value_counts())
                #st.write('-------------- AUGMENTATION CODE NOT INCLUDED YET ---------------')
            st.write("_" * 30)

            if 'aug_train_text' in st.session_state:
              col61, col62, col63 = st.columns([1,2,3])
              with col61:
                st.write("**Step-6: NLP pre-processing of data **")
              with col62:
                perform_preprocessing = st.button(label = 'Perform NLP preprocessing', key = 'perform_preprocessing')
              if perform_preprocessing: # and 'train_text1' not in st.session_state:
                for keys in ['Vector']: # remove all keys of importance to next step
                  if keys in st.session_state:
                    del st.session_state[keys]
                with col62:
                  with st.spinner('NLP preprocessing in progress, please wait...'):
                    st.session_state.train_text1, st.session_state.test_text1 = preprocess_func(st.session_state.aug_train_text,st.session_state.test_text)
              if 'train_text1' in st.session_state:
                original = round(np.mean([len(x.split()) for x in st.session_state.aug_train_text.tolist()]),2)
                new = round(np.mean([len(x.split()) for x in st.session_state.train_text1.tolist()]),2)
                with col63:
                  st.success('NLP preprocessing is complete')
                  st.write('**Average text length before NLP preprocessing: **',original)
                  st.write('**Average text length after NLP preprocessing: **',new)
              st.write("_" * 30)

              if 'train_text1' in st.session_state:
                col71, col72, col73 = st.columns([1,2,3])
                with col71:
                  st.write("**Step-7: Vectorization of training data **")
                with col72:
                  Vect = st.radio('Select Vector type',('tf-idf','CountVec','CustomVect'),key = 'Vect')
                  perform_vectorization = st.button(label = 'Perform vectorization', key = 'perform_vectorization')
                if perform_vectorization:
                  for keys in ['current_model','saved_results']: # remove all keys of importance to next step
                    if keys in st.session_state:
                      del st.session_state[keys]
                  st.session_state.Vector, st.session_state.x_train, st.session_state.x_test =vectorize(Vect, st.session_state.train_text1,st.session_state.test_text1)
                if 'Vector' in st.session_state:
                  with col73:
                    st.success('Vectorization is complete')
                    st.write('**Vectorization method: **',Vect)
                st.write("_" * 30)

                if 'Vector' in st.session_state:
                  col81, col82, col83 = st.columns([1,2,3])
                  with col81:
                    st.write("**Step-8: Training and testing of models **")
                  # Create an empty dataframe to save all the results
                  if 'saved_results' not in st.session_state:
                    st.session_state.metrics = ['Train accuracy','Test accuracy','Weighted F1','F1 scores','Precision','Recall','Run time']
                    st.session_state.saved_results = pd.DataFrame(columns=st.session_state.metrics)
                    st.session_state.saved_results['saved_models'] = np.nan
                  # List of all the models that can be tested
                  st.session_state.model_func = {'Random forest': r_f, 'Logistic Regression':LogR,'Gaussian NB':GNB,'KNeighbors Classifier':KNC,'SVC':SVC,'Decision Tree Classifier':DTC ,'Neural Network': n_n, 'LSTM':lstm,'bi-LSTM':bilstm}
                  model_list = list(st.session_state.model_func.keys())
                  with col82:
                    modl = st.selectbox('Select the model',options = pd.Series(model_list),key = 'modl')
                    train_model = st.button(label = 'Train the model', key = 'train_model')
                  if train_model:
                    for keys in ['model_saved','is_model_saved']: # remove all keys of importance to next step
                      if keys in st.session_state:
                        del st.session_state[keys]
                    if train_model not in list(st.session_state.saved_results.keys()):
                      with col82:
                        with st.spinner('Model Training in progress, please wait...'):
                          results, temp_model = st.session_state.model_func.get(st.session_state.modl)(st.session_state.x_train, st.session_state.x_test, st.session_state.y_train, st.session_state.y_test)
                      st.session_state.saved_results.loc[st.session_state.modl,st.session_state.metrics], st.session_state.current_model = results, temp_model
                      st.session_state.saved_results.loc[st.session_state.modl,'saved_models'] = st.session_state.current_model
                  if 'current_model' in st.session_state:
                    with col83:
                      st.success('Performance metrices of the models are calculated: ')
                      st.dataframe(st.session_state.saved_results.loc[:,st.session_state.metrics])
                      st.write('**Current active model: **',st.session_state.modl)
                      #st.metric('Weighted F1 score of the current active model: ',st.session_state.saved_results.loc[st.session_state.current_model,'Weighted F1'])
                  st.write("_" * 30)

                  if 'current_model' in st.session_state:
                    col91, col92, col93 = st.columns([1,2,3])
                    with col91:
                      st.write("**Step-9: Pickling of the current active model **")
                    with col92:
                      model_to_be_saved = st.radio('Select one of the models to save and use later',st.session_state.saved_results.index,key='model_to_be_saved')
                      pickle_model = st.button(label = 'Pickle/Save the current active model', key = 'pickle_model')
                    if pickle_model:
                      for keys in ['run_chatbot']: # remove all keys of importance to next step
                        if keys in st.session_state:
                          del st.session_state[keys]
                      #joblib.dump(st.session_state.current_model,'final_model.pkl')
                      st.session_state.model_saved = st.session_state.saved_results.loc[model_to_be_saved,'saved_models']
                      st.session_state.final_model_name = model_to_be_saved
                      st.session_state.is_model_saved = 'yes'
                      #save_model(st.session_state.current_model)
                    if 'is_model_saved' in st.session_state:
                      with col93:
                        st.success('Model is successfully saved')
                        st.write('**Pickled model: **',st.session_state.final_model_name)
                    st.write("_" * 30)

                    if 'model_saved' in st.session_state:
                      col101, col102 = st.columns([1,5])
                      #with col101:
                      #  st.write("**Step-10: Load the current saved model and run the chatbot**")
                      with col101:
                        run_chatbot = st.button(label = 'Load model and Run the chatbot',key = 'run_chatbot')
                      if run_chatbot:
                          st.session_state.final_model = st.session_state.model_saved #load_model()

                      if 'run_chatbot' in st.session_state:
                        with col102:
                          description = st.text_area('Please enter the Accident description here', value='default_value',
                                                      height = 200, key= 'description')
                        if description is not 'default_value':
                          with col102:
                            st.write('current description value is: ',description)
                          desc = preprecess_chatbot(st.session_state.description,st.session_state.Vector)
                          result = st.session_state.final_model.predict(desc)
                          with col102:
                            #st.write('result value is:', result)
                            #try:
                            #  st.write('result probabilities are: ',st.session_state.final_model.predict_prob(desc))
                            #except:
                            #  pass
                            st.write('**Predicted Accident level is: **',st.session_state.lb.inverse_transform(result)[0])

  return

main()
