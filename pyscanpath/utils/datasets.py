# avoiding pysaliecny compatibility issues
try:
    import collections.abc
    collections.Sequence = collections.abc.Sequence
    collections.MutableMapping = collections.abc.MutableMapping
except ImportError:
    pass

import pysaliency
import numpy as np

def getDatasetScanpath(dataset, data_location, stimuli_index, subject_index):
    
    if dataset == 'MIT1003':
        dataset_stimuli, dataset_fixations = pysaliency.external_datasets.get_mit1003(location = data_location)
    elif dataset == 'CAT2000':
        dataset_stimuli, dataset_fixations = pysaliency.external_datasets.get_cat2000_train(location = data_location)
        
    img = dataset_stimuli.stimuli[ stimuli_index ]
    subject_fixations = dataset_fixations[dataset_fixations.subjects == subject_index]
    img_subject_fixations = subject_fixations[subject_fixations.n == stimuli_index]
    x = img_subject_fixations.x
    y = img_subject_fixations.y
    scanpath = np.array([y, x]).transpose()
    img_path = dataset_stimuli.filenames[stimuli_index]
    
    if scanpath.size == 0:
        raise Exception("The scanpath is empty. Please select a new subject or a new stimuli.")
    
    return img_path, img, scanpath