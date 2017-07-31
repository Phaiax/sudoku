import numpy as np
import sys
from six.moves import cPickle as pickle

files = ['training_data_'+a for a in sys.argv[-1].split(',')]
saves = []

for f in files:
    with open(f, 'rb') as f:
        saves.append(pickle.load(f))

num_examples = sum([s['dataset'].shape[0] for s in saves])
r = saves[0]['dataset'].shape[1]
c = saves[0]['dataset'].shape[2]

all_data = np.empty((num_examples,r,c), dtype=np.float32)
all_labels = np.empty((num_examples), dtype=np.float32)

i = 0
for s in saves:
    l = s['dataset'].shape[0]
    all_data[i:i+l,:,:] = s['dataset']
    all_labels[i:i+l] = s['labels']
    i += l

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

randomize(all_data, all_labels)
randomize(all_data, all_labels)

train_size = 200000
valid_size = 10000
test_size = 10000

train_dataset = all_data[0:train_size,:,:]
train_labels = all_labels[0:train_size]
valid_dataset = all_data[train_size:train_size+valid_size,:,:]
valid_labels = all_labels[train_size:train_size+valid_size]
test_dataset = all_data[train_size+valid_size:train_size+valid_size+test_size,:,:]
test_labels = all_labels[train_size+valid_size:train_size+valid_size+test_size]

save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
}

pickle_file = 'sudoku_ml_data'
print "\nsave to", pickle_file, "...",
sys.stdout.flush()
with open(pickle_file, 'wb') as f:
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
print "done."
sys.stdout.flush()

