import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import glob
import shutil
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from config import Config, eval_dict_leaf
from utils import _frame_from_video, setup_internvideo2, frames2tensor
# from cos_text_video import frames2tensor


class InternVideo2VideoClassifier:
    def __init__(self, base_path, class_dirs_cn, class_dirs_en, config_path, intern_video2_path, bert_large_uncased_path):
        self.base_path = base_path
        self.class_dirs_cn = class_dirs_cn
        self.class_dirs_en = class_dirs_en
        self.config, self.intern_model, self.tokenizer = self.setup_internvideo2(config_path, intern_video2_path,
                                                                                 bert_large_uncased_path)
        self.intern_model = self.intern_model.to('cuda')

        self.video_input_dir = os.path.join(base_path, 'videos', '20240430')
        self.feats_save_dir = os.path.join(base_path, 'feats_test')
        self.center_dir = os.path.join(base_path, '2_level_center')
        self.output_dir = os.path.join(base_path, '2_level_test_v2v_update')

        os.makedirs(self.feats_save_dir, exist_ok=True)
        os.makedirs(self.center_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    # def __init__(self, base_path, class_dirs_cn, class_dirs_en, config_path, intern_video2_path,
    #              bert_large_uncased_path):
    #     self.base_path = base_path
    #     self.class_dirs_cn = class_dirs_cn
    #     self.class_dirs_en = class_dirs_en
    #     self.config, self.intern_model, self.tokenizer = self.setup_internvideo2(config_path, intern_video2_path,
    #                                                                              bert_large_uncased_path)
    #     self.intern_model = self.intern_model.to('cuda')
    #
    #     self.center_dir = os.path.join(base_path, '2_level_center')
    #     self.output_dir = os.path.join(base_path, '2_level_test_v2v_update')
    #
    #     os.makedirs(self.center_dir, exist_ok=True)
    #     os.makedirs(self.output_dir, exist_ok=True)

    def setup_internvideo2(self, config_path, intern_video2_path, bert_large_uncased_path):
        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)
        config['pretrained_path'] = intern_video2_path
        config['model']['text_encoder']['pretrained'] = bert_large_uncased_path
        intern_model, tokenizer = setup_internvideo2(config)
        return config, intern_model, tokenizer

    def extract_video_features(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = [x for x in _frame_from_video(cap)]
        cap.release()
        if len(frames) == 0:
            print(f"No frames extracted for video: {video_path}")
            return None
        feats = self.intern_model.get_vid_feat(frames2tensor(frames, fnum=self.config.get('num_frames', 8),
                                                             target_size=(self.config.get('size_t', 224),
                                                                          self.config.get('size_t', 224)),
                                                             device='cuda'))
        return feats

    def generate_text_features(self):
        text_feats = []
        for i, d in enumerate(self.class_dirs_cn):
            class_center_dir = os.path.join(self.center_dir, d)
            os.makedirs(class_center_dir, exist_ok=True)
            files = os.listdir(class_center_dir)
            if len(files):
                video_paths = [os.path.join(class_center_dir, f) for f in files if f.endswith('.mp4')]
                video_feats = []
                for video_path in video_paths:
                    video_feat = self.extract_video_features(video_path)
                    video_feats.append(video_feat)
                video_feats = torch.stack(video_feats, dim=0)
                feat = torch.mean(video_feats, dim=0, keepdim=True)
                feat /= feat.norm(dim=-1, keepdim=True)
                text_feats.append(feat)
            else:
                text_feats.append(self.intern_model.get_txt_feat(self.class_dirs_en[i]))
        return torch.cat(text_feats)

    def extract_video_features_and_save(self, video_input_dir, feats_save_dir):
        video_files = os.listdir(video_input_dir)
        for video_dir in video_files:
            video_dir_path = os.path.join(video_input_dir, video_dir)
            video_folders = [x for x in os.listdir(video_dir_path) if
                             os.path.isdir(os.path.join(video_dir_path, x))]

            for video_folder in video_folders:
                video_folder_path = os.path.join(video_dir_path, video_folder)
                video_paths = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if
                               f.endswith('.mp4')]

                video_feat_list = []
                video_paths_list = []

                for video_path in video_paths:
                    video_basename = os.path.basename(video_path)
                    feat_save_path = os.path.join(feats_save_dir, f"{os.path.splitext(video_basename)[0]}.pth")

                    if os.path.exists(feat_save_path):
                        print(f"Features already extracted for video: {video_basename}")
                        continue

                    cap = cv2.VideoCapture(video_path)
                    frames = [x for x in _frame_from_video(cap)]
                    cap.release()

                    if len(frames) == 0:
                        print(f"No frames extracted for video: {video_path}")
                        continue

                    feats = self.intern_model.get_vid_feat(
                        frames2tensor(frames, fnum=self.config.get('num_frames', 8),
                                      target_size=(self.config.get('size_t', 224),
                                                   self.config.get('size_t', 224)),
                                      device='cuda'))
                    video_feat_list.append(feats)
                    video_paths_list.append(video_path)

                if len(video_feat_list) > 0:
                    output_path = os.path.join(feats_save_dir, f"{video_folder}.pth")
                    torch.save({'feats': video_feat_list, 'video_paths': video_paths_list}, output_path)

    # def classify_videos(self, text_feats):
    #     video_paths_list = []
    #     video_files = glob.glob(os.path.join(self.video_input_dir, '*', '*.mp4'), recursive=True)
    #     print(f"Found {len(video_files)} video files in {self.video_input_dir}")
    #
    #     for video_path in tqdm(video_files, desc="Processing videos"):
    #         video_basename = os.path.basename(video_path)
    #         feat_save_path = os.path.join(self.feats_save_dir, f"{os.path.splitext(video_basename)[0]}.pth")
    #
    #         if os.path.exists(feat_save_path):
    #             print(f"Loading features for video: {video_basename}")
    #             data = torch.load(feat_save_path)
    #             video_feat = torch.mean(torch.stack(data['feats']), dim=0)
    #         else:
    #             video_feat = self.extract_video_features(video_path)
    #             if video_feat is None:
    #                 continue
    #
    #         video_paths_list.append({'feat': video_feat, 'path': video_path})
    #
    #     label_probs = (100.0 * torch.stack([v['feat'] for v in video_paths_list]) @ text_feats.view(-1, text_feats.size(
    #         0))).softmax(dim=-1)
    #     top_probs, top_labels = label_probs.cpu().topk(1, dim=-1)
    #
    #     top_labels = top_labels.squeeze()
    #
    #     if top_labels.dim() == 0:
    #         top_labels = [top_labels.item()]
    #     else:
    #         top_labels = top_labels.tolist()
    #
    #     for i, cdc in enumerate(self.class_dirs_cn):
    #         class_output_dir = os.path.join(self.output_dir, cdc)
    #         os.makedirs(class_output_dir, exist_ok=True)
    #         index = [j for j, x in enumerate(top_labels) if x == i]
    #
    #         if len(index) == 0:
    #             continue
    #
    #         np.random.shuffle(index)
    #         selects = [video_paths_list[j]['path'] for j in index[:200]]
    #         # selects = [video_paths_list[j]['path'] for j in index[:100]]
    #         for s in selects:
    #             shutil.copy(s, os.path.join(class_output_dir, os.path.basename(s)))
    #
    #     print(Counter(top_labels))
    def classify_videos(self, text_feats):
        video_paths_list = []
        video_files = os.listdir(self.video_input_dir)

        for video_dir in video_files:
            video_folder_path = os.path.join(self.video_input_dir, video_dir)
            video_paths = [os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if
                           f.endswith('.mp4')]

            for video_path in video_paths:
                video_basename = os.path.basename(video_path)
                feat_save_path = os.path.join(self.feats_save_dir, f"{os.path.splitext(video_basename)[0]}.pth")

                if not os.path.exists(feat_save_path):
                    print(f"Features file not found for video: {video_basename}")
                    continue

                data = torch.load(feat_save_path)
                video_feats = torch.stack(data['feats'])
                video_paths_list.append({'feat': torch.mean(video_feats, dim=0), 'path': video_path})

        label_probs = (100.0 * torch.stack([v['feat'] for v in video_paths_list]) @ text_feats.view(-1, text_feats.size(0))).softmax(dim=-1)
        # label_probs = (100.0 * video_feats @ text_feats.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.cpu().topk(1, dim=-1)
        top_labels = top_labels.squeeze()

        if top_labels.dim() == 0:
            top_labels = [top_labels.item()]
        else:
            top_labels = top_labels.tolist()

        for i, cdc in enumerate(self.class_dirs_cn):
            class_output_dir = os.path.join(self.output_dir, cdc)
            os.makedirs(class_output_dir, exist_ok=True)
            index = [j for j, x in enumerate(top_labels) if x == i]

            if len(index) == 0:
                continue

            np.random.shuffle(index)
            selects = [video_paths_list[j]['path'] for j in index]
            for s in selects:
                shutil.copy(s, os.path.join(class_output_dir, os.path.basename(s)))

        print(Counter(top_labels))
        # Print probabilities for each video
        for i, video in enumerate(video_paths_list):
            label_probs_video = (100.0 * video['feat'] @ text_feats.T).softmax(dim=-1)
            print(f"Video {os.path.basename(video['path'])} probabilities for each class:")
            for j, prob in enumerate(label_probs_video):
                print(f"Class: {self.class_dirs_cn[j]}, Probability: {prob.item()}")


if __name__ == '__main__':
    base_path = "/data/30062036/projects/InternVideo2/data"
    class_dirs_cn = [
        '人类活动场景',
        '动物活动场景',
        '植物',
        '常见事物',
        '环境和自然现象',
        '超现实场景'
    ]

    class_dirs_en = [
        'Human Activity Scenes',
        'Animal Activity Scenes',
        'Plants',
        'Common Items',
        'Environment and Natural Phenomena',
        'Surreal Scenes'
    ]
    config_path = "demo/internvideo2_stage2_config.py"
    intern_video2_path = "/data/30062036/projects/InternVideo2/checkpoints/InternVideo2-stage2_1b-224p-f4.pt"
    bert_large_uncased_path = "/data/30062036/projects/InternVideo2/checkpoints/bert-large-uncased"

    classifier = InternVideo2VideoClassifier(base_path, class_dirs_cn, class_dirs_en, config_path, intern_video2_path,
                                             bert_large_uncased_path)

    video_input_dir = os.path.join(base_path, 'videos', '20240430')
    feats_save_dir = os.path.join(base_path, 'feats_test')
    classifier.extract_video_features_and_save(video_input_dir, feats_save_dir)

    text_feats = classifier.generate_text_features()
    classifier.classify_videos(text_feats)


def calculate_similarity_for_video(video_path, text_feats, intern_model):
    cap = cv2.VideoCapture(video_path)
    frames = [_frame_from_video(cap) for _ in cap]
    cap.release()

    if len(frames) == 0:
        print(f"No frames extracted for video: {video_path}")
        return

    video_feats = intern_model.get_vid_feat(frames2tensor(frames, fnum=intern_model.config.get('num_frames', 8),
                                                        target_size=(intern_model.config.get('size_t', 224),
                                                                     intern_model.config.get('size_t', 224)),
                                                        device='cuda'))
    video_avg_feat = torch.mean(video_feats, dim=0)

    similarity = torch.nn.functional.cosine_similarity(video_avg_feat, text_feats, dim=0)
    print(f"相似度结果 for {video_path}: {similarity}")


video_path = 'path/to/video.mp4'
calculate_similarity_for_video(video_path, text_feats, classifier.intern_model)




