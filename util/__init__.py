import cPickle as pickle


def save_pickle(filename, data):
    with open(filename, 'w') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename) as f:
        return pickle.load(f)
