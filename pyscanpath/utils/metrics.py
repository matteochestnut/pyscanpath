import numpy as np
from scipy.spatial.distance import euclidean
from nltk.metrics.distance import edit_distance
import random
from tabulate import tabulate

def euclidean_distance(P, Q):
    scanpath_len = min(P.shape[0], Q.shape[0]) - 1
    
    return np.linalg.norm(
        np.sqrt( (P[0 : scanpath_len, 0] - Q[0 : scanpath_len, 0])**2 + (P[0 : scanpath_len, 1] - Q[0 : scanpath_len, 1])**2 ) )

def scaled_euclidean_distance(P, Q, height, width):
    scanpath_len = min(P.shape[0], Q.shape[0]) - 1
    
    P_scanpath = np.copy(P)
    Q_scanpath = np.copy(Q)

    # First, coordinates are rescaled as to an image with maximum dimension 1
    # This is because, clearly, smaller images would produce smaller distances
    max_dim = max(height, width)

    for fixation in P_scanpath:
        fixation[0] /= max_dim
        fixation[1] /= max_dim

    for fixation in Q_scanpath:
        fixation[0] /= max_dim
        fixation[1] /= max_dim
    
    return np.mean(
        np.sqrt( (P_scanpath[0 : scanpath_len, 0] - Q_scanpath[0 : scanpath_len, 0])**2 + (P_scanpath[0 : scanpath_len, 1] - Q_scanpath[0 : scanpath_len, 1])**2 ) )
    
def random_scanpath(scanpath_len, height, width):
		scanpath = np.zeros( (scanpath_len, 2) )
		for i in range(scanpath_len):
			scanpath[i, 0] = random.uniform(0, 1) * height
			scanpath[i, 1] = random.uniform(0, 1) * width
		return scanpath

def mannan_distance(P, Q, height, width):
    
    def D(P, Q, height, width):
        P_len = P.shape[0]
        Q_len = Q.shape[0]
        dist = np.zeros( (P_len, Q_len) )
        for i in range(P_len):
            for j in range(Q_len):
                dist[i,j] = euclidean(P[i], Q[j])
        
        d1i = np.min(dist, axis = 1)
        d2j = np.min(dist, axis = 0)
        
        result = Q_len * np.sum(np.power(d2j, 2)) + P_len * np.sum(np.power(d1i, 2))
        
        scale = 2 * P_len * Q_len * (height**2 + width**2)
        
        return result / scale
    
    d = D(P, Q, height, width)
    PR = random_scanpath(P.shape[0], height, width)
    QR = random_scanpath(Q.shape[0], height, width)
    dr = D(PR, QR, height, width)
    
    # return mannan distance similarity between 0 and 1 (0 completely different, 1 completely similar)
    return np.abs( 1 - ( np.sqrt(d) / np.sqrt(dr) ) ) #* 100

def scanpath_to_string(scanpath, height, width, grid_size):
    '''
    Convert scanpath to a string
    From Fixatons:
    https://github.com/dariozanca/FixaTons
    '''
    width_step = width // grid_size
    height_step = height // grid_size

    string = ''

    for i in range(np.shape(scanpath)[0]):
        fixation = scanpath[i].astype(np.int32)
        correspondent_square = (fixation[0] // width_step) + (fixation[1] // height_step) * grid_size
        string += chr(97 + int(correspondent_square))

    return string

def levenshtein_distance(P,Q, height, width, grid_size = 8):
	'''
	Levenshtein distance, or edit distance
	'''
	P_string = scanpath_to_string(P, height, width, grid_size)
	Q_string = scanpath_to_string(Q, height, width, grid_size)
    # normalizing the edit distance by the maximum length between the two scanpaths
	return edit_distance(P_string, Q_string) / max(P.shape[0], Q.shape[0])

def time_delay_embedding_distance(P, Q, k=3, distance_mode='Mean'):
    '''
	time delay embedding distance
	From Fixatons:
    https://github.com/dariozanca/FixaTons 
	'''
    # k must be shorter than both scanpath lenghts
    if len(P) < k or len(Q) < k:
        print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths
    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])
    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    # for each k-vector from Q, look for the k-vector from P
    # which has the minumum distance, and save the distance value
    distances = []
    for s_k_vec in Q_vectors:
        # find human k-vec of minimum distance
        norms = []
        for h_k_vec in P_vectors:
            d = euclidean_distance(s_k_vec, h_k_vec)
            norms.append(d)

        distances.append(min(norms) / k)

    # "distances" contains the value of the minimum distance
    # or each Q k-vec
    # according to the distance_mode, here is computed the similarity
    # between the two scanpaths.
    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False

def scaled_time_delay_embedding_distance(P, Q, height, width):
    '''
	scaled time delay embedding distance
	From Fixatons:
    https://github.com/dariozanca/FixaTons 
	'''
    # to preserve data, we work on copies of the lists
    P_scanpath = np.copy(P)
    Q_scanpath = np.copy(Q)

    # First, coordinates are rescaled as to an image with maximum dimension 1
    # This is because, clearly, smaller images would produce smaller distances
    max_dim = max(height, width)

    for fixation in P_scanpath:
        fixation[0] /= max_dim
        fixation[1] /= max_dim

    for fixation in Q_scanpath:
        fixation[0] /= max_dim
        fixation[1] /= max_dim

    # Then, scanpath similarity is computer for all possible k
    max_k = min(len(P_scanpath), len(Q_scanpath))
    similarities = []
    for k in np.arange(1, max_k + 1):
        s = time_delay_embedding_distance(
            P_scanpath,
            Q_scanpath,
            k=k,  # time-embedding vector dimension
            distance_mode='Mean')
        similarities.append(np.exp(-s))

    # Now that we have similarity measure for all possible k
    # we compute and return the mean

    return sum(similarities) / len(similarities)

def printMetrics(img, ground_truth, scanpath_list, methods_list):
    
    table = []
    
    headers = ['Model', 'Euclidean', 'Mannan', 'Levenshtein', 'TDE']
        
    for i in range(len(scanpath_list)):
        ed = scaled_euclidean_distance(ground_truth, scanpath_list[i], img.shape[0], img.shape[1])
        lev = levenshtein_distance(ground_truth, scanpath_list[i], img.shape[0], img.shape[1])
        tde = scaled_time_delay_embedding_distance(ground_truth, scanpath_list[i], img.shape[0], img.shape[1])
        md = mannan_distance(ground_truth, scanpath_list[i], img.shape[0], img.shape[1])
        scanpath_metrics = [methods_list[i], ed, md, lev, tde]
        table.append(scanpath_metrics)
    
    print(tabulate(table, headers))