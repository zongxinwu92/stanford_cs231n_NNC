def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict

for x in range(1, 6):
    file = 'data_batch_' + str(x)
    unpickle(file)

unpickle('test_batch')
unpickle('batches.meta')