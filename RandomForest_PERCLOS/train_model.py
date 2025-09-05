import cv2
import dlib
import pandas as pd
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import glob
import os
import sys
from tqdm import tqdm
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def calculate_ear(eye):

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def process_video(video_path, detector, predictor):

# 处理单个视频，提取眨眼相关特征 (跳过前20秒和后5秒)

    EAR_THRESHOLD = 0.25
    EAR_CONSEC_FRAMES = 3

    blink_counter = 0
    total_frames_processed = 0
    closed_eye_frames = 0
    consecutive_frames_counter = 0

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps == 0:
        fps = 30

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(20 * fps)
    end_frame = int(total_video_frames - (5 * fps))

    if start_frame >= end_frame:
        print(f"警告: 视频 {os.path.basename(video_path)} 时长不足25秒，将处理整个视频。")
        start_frame = 0
        end_frame = total_video_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pbar = tqdm(total=(end_frame - start_frame), desc=f"处理: {os.path.basename(video_path)}", unit="帧", leave=False)

    while cap.isOpened():
        current_frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame_pos >= end_frame:
            break

        ret, frame = cap.read()
        if not ret: break

        total_frames_processed += 1
        pbar.update(1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) > 0:
            shape = predictor(gray, rects[0])
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (calculate_ear(leftEye) + calculate_ear(rightEye)) / 2.0

            if ear < EAR_THRESHOLD:
                consecutive_frames_counter += 1
                closed_eye_frames += 1
            else:
                if consecutive_frames_counter >= EAR_CONSEC_FRAMES:
                    blink_counter += 1
                consecutive_frames_counter = 0

    pbar.close()
    cap.release()

    if total_frames_processed == 0:
        return {'video_id': os.path.basename(video_path).split('.')[0], 'blink_freq': 0, 'perclos': 0}

    duration_sec = total_frames_processed / fps
    blink_freq = (blink_counter / duration_sec) * 60 if duration_sec > 0 else 0
    perclos = (closed_eye_frames / total_frames_processed) * 100 if total_frames_processed > 0 else 0

    return {'video_id': os.path.basename(video_path).split('.')[0], 'blink_freq': blink_freq, 'perclos': perclos}


def main():

    print("--- 任务开始：疲劳检测模型训练 ---")

    print("\n[进度] 正在检查模型文件并设置路径...")
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print(f"错误: 关键模型文件 {predictor_path} 不存在！")
        sys.exit()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    base_folder = '/Users/bjy/Desktop/Detection/Experiment'
    video_folder = os.path.join(base_folder, '0_raw_videos')
    label_folder = os.path.join(base_folder, '1_label_data')

    print(f"  > 视频路径: {video_folder}")
    print(f"  > 标签路径: {label_folder}")

    print("\n[进度] 正在加载并预处理标签数据...")


    pre_test_files = glob.glob(os.path.join(label_folder, '*pretest*.xlsx'))
    if not pre_test_files:
        print(f"\n错误：在文件夹 '{label_folder}' 中找不到任何包含 'pretest' 的xlsx文件。")
        sys.exit()

    post_test_files = glob.glob(os.path.join(label_folder, '*posttest*.xlsx'))
    if not post_test_files:
        print(f"\n错误：在文件夹 '{label_folder}' 中找不到任何包含 'posttest' 的xlsx文件。")
        sys.exit()

    pre_test_file = pre_test_files[0]
    post_test_file = post_test_files[0]


    pre_df = pd.read_excel(pre_test_file)
    post_df = pd.read_excel(post_test_file)


    pre_df['video_id'] = pre_df['实验编号'].astype(str) + '_pretest'
    post_df['video_id'] = post_df['实验编号'].astype(str) + '_posttest'

    labels_df = pd.concat([pre_df, post_df])[['video_id', '疲劳等级']]
    print(f"  > 标签数据加载完毕，共 {len(labels_df)} 条记录。")

    video_paths = glob.glob(os.path.join(video_folder, '*.mp4')) + glob.glob(os.path.join(video_folder, '*.mov'))
    if not video_paths:
        print(f"\n错误: 在文件夹 '{video_folder}' 中找不到任何 .mp4 或 .mov 视频文件。")
        sys.exit()

    all_features = []
    print(f"\n[进度] 开始从 {len(video_paths)} 个视频中提取特征...")
    for video_path in sorted(video_paths):
        features = process_video(video_path, detector, predictor)
        if features:
            all_features.append(features)
    print("  > 所有视频特征提取完毕！")
    features_df = pd.DataFrame(all_features)

    print("\n[进度] 正在合并特征和标签...")
    final_df = pd.merge(features_df, labels_df, on='video_id', how='inner')
    final_df.to_csv('fatigue_features.csv', index=False)
    print(f"  > 合并完成！共找到 {len(final_df)} 个匹配数据，已保存至 fatigue_features.csv")

    print("\n[进度] 开始训练分类模型...")
    if len(final_df) < 5:
        print("  > 错误：匹配到的数据不足5条，无法训练模型。请检查视频和标签文件名是否能对应。")
        sys.exit()

    X = final_df[['blink_freq', 'perclos']]
    y_raw = final_df['疲劳等级']

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("  > 正在分割训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y if np.min(np.bincount(y)) >= 2 else None)

    print("  > 正在训练随机森林模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("  > 模型训练完毕！")

    joblib.dump(model, 'fatigue_model.joblib')
    print("  > 模型已保存至 fatigue_model.joblib")

    print("\n[进度] 开始评估模型并生成可视化结果...")
    y_pred = model.predict(X_test)
    print(f"\n  > 模型在测试集上的准确率: {accuracy_score(y_test, y_pred):.2%}")
    print("\n  > 分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("\n  > 混淆矩阵图已保存至 confusion_matrix.png")

    importances = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(
        'importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    print("  > 特征重要性图已保存至 feature_importance.png")

    print("\n--- 任务全部完成！ ---")


if __name__ == '__main__':
    main()