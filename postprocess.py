import numpy as np
from scipy.ndimage import label

def largest_connected_component(mask):
    labeled, num = label(mask)
    if num == 0:
        return mask
    counts = [(labeled==i).sum() for i in range(1, num+1)]
    largest = counts.index(max(counts)) + 1
    return (labeled == largest).astype(np.uint8)

def refine_segmentation(seg):
    # seg: numpy array with values {0,1,2}
    liver = largest_connected_component(seg==1)
    tumor = seg==2
    tumor = tumor * liver  # restrict tumor inside liver
    out = np.zeros_like(seg)
    out[liver==1] = 1
    out[tumor==1] = 2
    return out