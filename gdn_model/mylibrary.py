import os
import pickle


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