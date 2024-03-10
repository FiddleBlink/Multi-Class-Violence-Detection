import numpy as np
import os
import cv2

clip_len = 16

# the dir of testing images
video_root = 'Features/Videos'   ## the path of test videos
feature_list = 'list/rgb_test.list'
# the ground truth txt

gt_txt = 'Features/annotations.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0

# Define a mapping from class names to numeric labels
class_mapping = {'B1': 1, 'B2': 2, 'G': 3, 'B4': 4, 'B5': 5, 'B6': 6}

for idx in range(len(lists)):
    name = lists[idx].strip('\n').split('/')[-1]
    if '__0.npy' not in name:
        continue
    name = name[:-7]
    vname = name+'.mp4'
    cap = cv2.VideoCapture(os.path.join(video_root,vname))
    lens = int(cap.get(7)) #Used to get the total number of frames in the video

    # the number of testing images in this sub-dir

    # Initialize gt_vec with the label for the "background" class
    gt_vec = np.full(lens, 0).astype(np.float32)
    if '_label_A' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                        for j in range(2,len(gt_content),2) if j==i+1]
                class_name = gt_content[0].split('_')[-1].split('-')[0]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        gt_vec[frag[0]:frag[1]]=class_mapping[class_name]
                break
    mod = (lens-1) % clip_len # minusing 1 is to align flow  rgb: minusing 1 when extracting features
    gt_vec = gt_vec[:-1]
    if mod:
        gt_vec = gt_vec[:-mod]
    gt.extend(gt_vec)
    if sum(gt_vec)/len(gt_vec):
        tlens += len(gt_vec)
        vlens += sum(gt_vec)

np.save('gtMulti.npy', gt)