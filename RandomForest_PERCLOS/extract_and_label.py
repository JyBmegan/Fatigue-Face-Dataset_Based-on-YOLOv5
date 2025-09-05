from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
from pathlib import Path

VIDEO_SOURCE_DIR = "0_raw_videos"
LABEL_SOURCE_DIR = "1_label_data"
OUTPUT_DATASET_DIR = "2_yolo_dataset"
PRETEST_XLSX_FILENAME = "pretest.xlsx"
POSTTEST_XLSX_FILENAME = "posttest.xlsx"
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 1 - TRAIN_SPLIT_RATIO
SKIP_START_SECONDS = 20
SKIP_END_SECONDS = 5
FRAME_EXTRACT_INTERVAL = 15
CLASS_MAPPING = {"清醒": 0, "中度疲劳": 1, "重度疲劳": 2}
CLASS_NAMES = list(CLASS_MAPPING.keys())


# 分层抽样
def create_yolo_dataset():
    print("--- 开始创建YOLO数据集 (采用分层抽样策略) ---")

    try:
        pre_df = pd.read_excel(os.path.join(LABEL_SOURCE_DIR, PRETEST_XLSX_FILENAME))
        post_df = pd.read_excel(os.path.join(LABEL_SOURCE_DIR, POSTTEST_XLSX_FILENAME))
        print("成功从 .xlsx 文件中加载量表数据。")
    except Exception as e:
        print(f"读取 Excel 文件时出错: {e}")
        return

    pre_df['type'] = 'pre'
    post_df['type'] = 'post'
    all_labels_df = pd.concat([pre_df, post_df])
    all_labels_df.set_index(['实验编号', 'type'], inplace=True)

    Path(os.path.join(OUTPUT_DATASET_DIR, "images", "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DATASET_DIR, "images", "val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DATASET_DIR, "labels", "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(OUTPUT_DATASET_DIR, "labels", "val")).mkdir(parents=True, exist_ok=True)
    print("输出目录结构创建完成。")

    video_files = [f for f in os.listdir(VIDEO_SOURCE_DIR) if f.lower().endswith(('.mp4', '.mov'))]
    if not video_files:
        print(f"错误：在文件夹 '{VIDEO_SOURCE_DIR}' 中没有找到视频文件。")
        return

    subject_ids_all = sorted(list(set([int(f.split('_')[0]) for f in video_files])))

    stratify_labels = []
    valid_subject_ids = []
    for subject_id in subject_ids_all:
        try:
            # 以 posttest 结果为准进行分层
            fatigue_level = all_labels_df.loc[(subject_id, 'post'), '疲劳等级']
            stratify_labels.append(fatigue_level)
            valid_subject_ids.append(subject_id)
        except KeyError:
            print(f"[警告] 被试 {subject_id} 缺少 posttest 数据，将无法参与分层抽样。")

    if len(valid_subject_ids) < 2:
        print("错误：有效被试数量不足，无法进行数据集划分。")
        return

    train_subjects_list, val_subjects_list = train_test_split(
        valid_subject_ids,
        test_size=VAL_SPLIT_RATIO,
        stratify=stratify_labels,
        random_state=42 #确保每次运行代码，划分结果都一样
    )

    train_subjects = set(train_subjects_list)
    val_subjects = set(val_subjects_list)

    print(f"总共 {len(valid_subject_ids)} 位有效被试参与分层抽样。")
    print(f"训练集被试: {len(train_subjects)} 个")
    print(f"验证集被试: {len(val_subjects)} 个")

    for video_filename in video_files:
        print(f"\n--- 正在处理视频: {video_filename} ---")

        try:
            filename_stem = Path(video_filename).stem
            parts = filename_stem.split('_')
            subject_id = int(parts[0])
            raw_test_type = parts[1].lower()

            if raw_test_type == 'pretest':
                lookup_type = 'pre'
            elif raw_test_type == 'posttest':
                lookup_type = 'post'
            else:
                raise ValueError("Test type not 'pretest' or 'posttest'")
        except (ValueError, IndexError):
            print(f"  [警告] 文件名格式不正确，跳过: {video_filename}")
            continue

        dataset_type = "train" if subject_id in train_subjects else "val"

        try:
            fatigue_level_text = all_labels_df.loc[(subject_id, lookup_type), '疲劳等级']
            class_id = CLASS_MAPPING.get(fatigue_level_text)
            if class_id is None:
                print(f"  [警告] 未知的疲劳等级 '{fatigue_level_text}'，跳过视频。")
                continue
        except KeyError:
            print(f"  [警告] 在量表中找不到编号 {subject_id} ({lookup_type}) 的记录，跳过视频。")
            continue

        video_path = os.path.join(VIDEO_SOURCE_DIR, video_filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  [错误] 无法打开视频文件: {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"  [警告] 视频FPS为0，无法处理，跳过: {video_filename}")
            cap.release()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(SKIP_START_SECONDS * fps)
        end_frame = int(total_frames - (SKIP_END_SECONDS * fps))

        if start_frame >= end_frame:
            print(f"  [警告] 视频时长过短，无法提取帧，跳过。")
            cap.release()
            continue

        print(f"  疲劳等级: {fatigue_level_text} (Class ID: {class_id}) -> 分配至 {dataset_type} 集")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame_idx = start_frame
        saved_frame_count = 0
        while current_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret: break

            if (current_frame_idx - start_frame) % FRAME_EXTRACT_INTERVAL == 0:
                base_output_filename = f"{subject_id}_{raw_test_type}_frame_{current_frame_idx}"
                image_path = os.path.join(OUTPUT_DATASET_DIR, "images", dataset_type, f"{base_output_filename}.jpg")
                cv2.imwrite(image_path, frame)

                yolo_label_content = f"{class_id} 0.5 0.5 1.0 1.0"
                label_path = os.path.join(OUTPUT_DATASET_DIR, "labels", dataset_type, f"{base_output_filename}.txt")
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(yolo_label_content)
                saved_frame_count += 1
            current_frame_idx += 1

        cap.release()
        print(f"  处理完成，共保存了 {saved_frame_count} 帧图像和标签。")

    print("\n--- 所有视频处理完毕 ---")
    print(f"数据集已生成在 '{OUTPUT_DATASET_DIR}' 文件夹中。")


if __name__ == '__main__':
    create_yolo_dataset()