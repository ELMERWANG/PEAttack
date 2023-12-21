import os
import pandas as pd
from pandas import DataFrame
from pandas import concat
import pickle
from tensorflow.keras.models import model_from_json

def read_csv_dataset(name):
    filename = "./dataset/swat/%s" % name
    try:
        dataset = pd.read_csv(filename, header=0, index_col=0)
        print('The dataset has been loaded!')
        return dataset
    except FileNotFoundError:
        print('File name is not found, please try again! Try adding the file suffix csv :)')

def read_xlsx_dataset(name):
    filename = "./dataset/swat/%s" % name
    try:
        dataset = pd.read_excel(filename, header=0, index_col=0)
        print('The dataset has been loaded!')
        return dataset
    except FileNotFoundError:
        print('File name is not found, please try again! Try adding the file suffix xlsx :)')

def save_model(model, model_name, overwrite=False):

    model_json = model.to_json()
    model_name_main = "./saved/saved_model/%s.json" % model_name
    model_name_h5 = "./saved/saved_model/%s.h5" % model_name


    if os.path.exists(model_name_main) and overwrite == False:
        print('The file with same name exists! The model is not saved!')

    elif os.path.exists(model_name_main) and overwrite == True:
        with open(model_name_main, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_name_h5)
        print("The model exists! Force save enabled, the model is saved to disk!")

    else:
        with open(model_name_main, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(model_name_h5)
        print("Model saved to disk")

def load_model(model_name):

    model_name_load = "./saved/saved_model/%s.json" % model_name
    model_weight_load = "./saved/saved_model/%s.h5" % model_name

    json_file = open(model_name_load, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weight_load)
    print("Loaded model from disk")

    return model


def save_variable(target, name, overwrite=False):
    
    filename = "./saved/saved_variable/%s" % name

    if os.path.exists(filename) and overwrite == False:
        print('The file with same name exists! The content are not saved to the file!')
    elif os.path.exists(filename) and overwrite == True:
        print('The file with same name exists! Overwrite enabled, the file has been overwritten and saved!')
        pickle.dump(target,open(filename,"wb"))
    else:
        pickle.dump(target,open(filename,"wb"))
        print("Variable saved!")

def load_variable(name):

    filename = "./saved/saved_variable/%s" % name
    loaded_target = pickle.load(open(filename,"rb"))
    print("Variable loaded!")
    return loaded_target


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	'''convert series to supervised learning, adopted from https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/'''
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
