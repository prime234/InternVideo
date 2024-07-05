import torch
import os
import numpy as np
import shutil
from collections import Counter
from utils import (retrieve_text,
                  _frame_from_video,
                  setup_internvideo2, frames2tensor)
def calculate_video_similarity(video_path, classifier):
    feat = classifier.intern_model.get_vid_feat(frames2tensor(frames, fnum=classifier.config.get('num_frames', 8),
                        target_size=(classifier.config.get('size_t', 224), classifier.config.get('size_t', 224)),
                        device='cuda'))

    text_feats = classifier.generate_text_features()

    label_probs = (100.0 * feat @ text_feats.view(-1, text_feats.size(0))).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(len(classifier.class_dirs_cn), dim=-1)
    top_labels = top_labels.squeeze().tolist()

    for i, class_cn in enumerate(classifier.class_dirs_cn):
        print(f"Similarity with class '{class_cn}': {top_probs[i]:.2f}")

# 用法示例
video_path = "path_to_video.mp4"
calculate_video_similarity(video_path, classifier)