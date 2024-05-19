import pandas as pd
import numpy as np
import warnings
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
from xgboost import XGBRegressor

# Suprimir avisos
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def get_data():
    # Obter dados de treino e teste
    train_data_path = r'C:\\TCC\\Output\\ML\\train.csv'
    test_data_path = r'C:\\TCC\\Output\\ML\\test.csv'
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    return train, test

def get_combined_data():
    train, test = get_data()
    target = train.TD
    train.drop(['TD'], axis=1, inplace=True)
    combined = pd.concat([train, test], ignore_index=True)
    combined.drop(['TD'], axis=1, inplace=True)
    return combined, target

def get_cols_with_no_nans(df, col_type):
    if col_type == 'num':
        predictors = df.select_dtypes(exclude=['object'])
    elif col_type == 'no_num':
        predictors = df.select_dtypes(include=['object'])
    elif col_type == 'all':
        predictors = df
    else:
        print('Error: choose a type (num, no_num, all)')
        return []
    
    cols_with_no_nans = [col for col in predictors.columns if not df[col].isnull().any()]
    return cols_with_no_nans

def one_hot_encode(df, col_names):
    for col in col_names:
        if df[col].dtype == np.dtype('object'):
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop([col], axis=1, inplace=True)
    return df

def split_combined(combined, target):
    train = combined.iloc[:12, :]
    test = combined.iloc[12:, :]
    return train, test, target

def make_submission(prediction, sub_name):
    test = pd.read_csv(r'C:\\TCC\\Output\\ML\\test.csv')
    my_submission = pd.DataFrame({'TD': test.TD, 'TD': prediction})
    my_submission.to_csv(f'{sub_name}.csv', index=False)
    print(f'A submission file has been made: {sub_name}.csv')

# Load train and test data into pandas DataFrames
train_data, test_data = get_data()

# Combine train and test data to process them together
combined, target = get_combined_data()

num_cols = get_cols_with_no_nans(combined, 'num')
cat_cols = get_cols_with_no_nans(combined, 'no_num')

print(f'Number of numerical columns with no nan values: {len(num_cols)}')
print(f'Number of non-numerical columns with no nan values: {len(cat_cols)}')

combined = combined[num_cols + cat_cols]
combined.hist(figsize=(12, 10))
plt.show()

train_data = combined[:1460]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize=(15, 15))
sb.heatmap(C_mat, vmax=.8, square=True)
plt.show()

print(f'There were {combined.shape[1]} columns before encoding categorical features')
combined = one_hot_encode(combined, cat_cols)
print(f'There are {combined.shape[1]} columns after encoding categorical features')

train, test, target = split_combined(combined, target)

NN_model = Sequential()

# The Input Layer:
NN_model.add(Dense(128, kernel_initializer='normal', input_dim=train.shape[1], activation='relu'))

# The Hidden Layers:
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer:
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network:
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.keras'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# Training
NN_model.fit(train, target, epochs=500, batch_size=32, validation_split=0.2, callbacks=callbacks_list)

# Load weights file of the best model:
weights_file = 'Weights-298--0.00132.keras'  # choose the best checkpoint
NN_model.load_weights(weights_file)  # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

predictions = NN_model.predict(test)
make_submission(predictions[:, 0], r'C:\\TCC\\Output\\ML\\submission(NN).csv')

train_X, val_X, train_y, val_y = train_test_split(train, target, test_size=0.25, random_state=14)

model = RandomForestRegressor()
model.fit(train_X, train_y)

# Get the mean absolute error on the validation data
predicted_prices = model.predict(val_X)
MAE = mean_absolute_error(val_y, predicted_prices)
print(f'Random forest validation MAE = {MAE}')

predicted_prices = model.predict(test)
make_submission(predicted_prices, r'C:\\TCC\\Output\\ML\\Submission(RF).csv')

XGBModel = XGBRegressor()
XGBModel.fit(train_X, train_y, verbose=False)

# Get the mean absolute error on the validation data:
XGBpredictions = XGBModel.predict(val_X)
MAE = mean_absolute_error(val_y, XGBpredictions)
print(f'XGBoost validation MAE = {MAE}')

XGBpredictions = XGBModel.predict(test)
make_submission(XGBpredictions, r'C:\\TCC\\Output\\ML\\Submission(XGB).csv')
