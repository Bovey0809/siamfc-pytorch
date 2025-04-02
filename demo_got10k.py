from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

def load_groundtruth(gt_file):
    """Load ground truth annotations from file."""
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    gt = []
    for line in lines:
        x, y, w, h = map(float, line.strip().split(','))
        gt.append([x, y, w, h])
    return np.array(gt)

if __name__ == '__main__':
    # Load the longer sequence
    seq_dir = os.path.expanduser('~/data/test/GOT-10k_Test_000180')
    img_files = sorted(glob.glob(os.path.join(seq_dir, '*.jpg')))
    anno = load_groundtruth(os.path.join(seq_dir, 'groundtruth.txt'))
    
    print(f'Found {len(img_files)} images and {len(anno)} annotations')
    print(f'First image: {img_files[0]}')
    print(f'First annotation: {anno[0]}')
    
    # Initialize tracker
    tracker = TrackerSiamFC(net_path='pretrained/siamfc_alexnet_e50.pth')
    
    # Run tracking
    tracker.track(img_files, anno[0], visualize=True)
    print("Tracking completed successfully!") 