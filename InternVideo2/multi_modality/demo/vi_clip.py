import os
import glob
import shutil
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
from simple_tokenizer import SimpleTokenizer as _Tokenizer
from viclip import ViCLIP
import cv2


class ViCLIP_VideoClassifier:
    def __init__(self, base_path, class_dirs_cn, class_dirs_en, model_name='viclip'):
        self.base_path = base_path
        self.class_dirs_cn = class_dirs_cn
        self.class_dirs_en = class_dirs_en
        self.clip, self.tokenizer = self.get_clip(model_name)
        self.clip = self.clip.to('cuda')

        self.video_input_dir = os.path.join(base_path, 'video')
        self.feats_save_dir = os.path.join(base_path, 'feats_viclip')
        self.output_dir = os.path.join(base_path, 'classified_videos_viclip')

        os.makedirs(self.feats_save_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_clip(self, name='viclip'):
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            return vclip, tokenizer
        else:
            raise Exception(f'The target clip model {name} is not found.')

    def get_text_feat_dict(self, texts):
        text_feat_d = {}
        for t in texts:
            feat = self.clip.get_text_features(t, self.tokenizer)
            text_feat_d[t] = feat
        return text_feat_d

    def get_vid_feat(self, frames):
        return self.clip.get_vid_features(frames)

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = [frame for frame in self._frame_from_video(cap)]
        cap.release()
        return frames

    def normalize(self, data):
        v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        return (data / 255.0 - v_mean) / v_std

    def frames2tensor(self, vid_list, fnum=8, target_size=(224, 224)):
        assert (len(vid_list) >= fnum)
        step = len(vid_list) // fnum
        vid_list = vid_list[::step][:fnum]
        vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
        vid_tube = [np.expand_dims(self.normalize(x), axis=(0, 1)) for x in vid_list]
        vid_tube = np.concatenate(vid_tube, axis=1)
        vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
        vid_tube = torch.from_numpy(vid_tube).to('cuda', non_blocking=True).float()
        return vid_tube

    def extract_video_features(self, video_path):
        frames = self.extract_frames(video_path)
        if len(frames) == 0:
            print(f"No frames extracted for video: {video_path}")
            return None
        feats = self.get_vid_feat(self.frames2tensor(frames))
        return feats

    def classify_videos(self):
        video_feats_list = []
        video_paths_list = []

        video_files = glob.glob(os.path.join(self.video_input_dir, '*.mp4'), recursive=True)
        print(f"Found {len(video_files)} video files in {self.video_input_dir}")

        for video_path in tqdm(video_files, desc="Processing videos"):
            video_feat = self.extract_video_features(video_path)
            if video_feat is not None:
                video_feats_list.append(video_feat)
                video_paths_list.append(video_path)

        if not video_feats_list:
            print("No features extracted from videos.")
            return

        video_feats = torch.stack(video_feats_list)
        feats_save_path = os.path.join(self.feats_save_dir, 'video_features_viclip.pth')
        torch.save({'feats': video_feats, 'video_paths': video_paths_list}, feats_save_path)

        text_feat_d = self.get_text_feat_dict(self.class_dirs_en)
        text_feats = torch.cat([text_feat_d[t] for t in self.class_dirs_en], 0)

        label_probs = (100.0 * video_feats @ text_feats.T).softmax(dim=-1)
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
            selects = np.array(video_paths_list)[index[:100]]
            for s in selects:
                shutil.copy(s, os.path.join(class_output_dir, os.path.basename(s)))

        print(Counter(top_labels))


if __name__ == '__main__':
    base_path = "/data/30062036/projects/data/"
    class_dirs_cn = [
        '打扫', '洗漱', '穿戴', '美妆', '烹饪', '饮食', '亲子', '宠物', '休息', '购物',
        '社交', '游览', '棋牌', '护理', '摄影', '户外', '日常锻炼', '竞技比赛', '极限运动',
        '唱歌', '跳舞', '演奏', '戏剧', '魔术', '绘画', '雕刻', '文职工作', '维修制作',
        '建筑施工', '工业生产', '农业种植', '驾驶控制', '军事活动', '传统习俗', '仪式', '庆祝'
    ]

    class_dirs_en = [
        'Human Activity Scenes Cleaning', 'Human Activity Scenes Grooming', 'Human Activity Scenes Dressing',
        'Human Activity Scenes Makeup Application', 'Human Activity Scenes Cooking', 'Human Activity Scenes Eating',
        'Human Activity Scenes Parent-Child Activities', 'Human Activity Scenes Pet Care',
        'Human Activity Scenes Resting',
        'Human Activity Scenes Shopping', 'Human Activity Scenes Socializing', 'Human Activity Scenes Sightseeing',
        'Human Activity Scenes Card and Board Games', 'Human Activity Scenes Caretaking',
        'Human Activity Scenes Photography',
        'Human Activity Scenes Outdoor Activities', 'Human Activity Scenes Routine Exercise',
        'Human Activity Scenes Competitive Sports',
        'Human Activity Scenes Extreme Sports', 'Human Activity Scenes Singing', 'Human Activity Scenes Dancing',
        'Human Activity Scenes Playing Instruments', 'Human Activity Scenes Drama', 'Human Activity Scenes Magic',
        'Human Activity Scenes Painting', 'Human Activity Scenes Sculpting', 'Human Activity Scenes Clerical Work',
        'Human Activity Scenes Repair and Crafting', 'Human Activity Scenes Construction',
        'Human Activity Scenes Industrial Production',
        'Human Activity Scenes Agricultural Cultivation', 'Human Activity Scenes Vehicle Operation',
        'Human Activity Scenes Military Activities',
        'Human Activity Scenes Traditional Customs', 'Human Activity Scenes Ceremonies',
        'Human Activity Scenes Celebrations'
    ]

    classifier = ViCLIP_VideoClassifier(base_path, class_dirs_cn, class_dirs_en)
    classifier.classify_videos()