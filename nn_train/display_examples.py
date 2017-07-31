from matplotlib import pyplot as plt
from six.moves import cPickle as pickle
import numpy as np

pickle_file = "training_data"
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    dl_dataset = save['dataset']
    dl_labels = save['labels']

w,h=7,5
for i, r in enumerate(np.random.randint(0, dl_dataset.shape[0], w*h)):
    plt.subplot(w,h,i+1)
    plt.imshow(dl_dataset[r,:,:], cmap='gray')

plt.show()