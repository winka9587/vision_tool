import cv2
import os

def save_frames_from_video(video_path, output_folder, n_digits=5):
    # 检查并创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 生成带有前导零的文件名
        file_name = f"{str(frame_count).zfill(n_digits)}.png"
        file_path = os.path.join(output_folder, file_name)

        # 保存帧为PNG图像
        cv2.imwrite(file_path, frame)

        frame_count += 1

    # 释放视频对象
    cap.release()
    print(f"save frame {frame_count} to {file_path}")


if __name__ == "__main__":
    # 示例用法
    video_path = './data/IMG_4097.mp4'  # 输入视频路径
    output_folder = './data/output_frames'  # 输出图像文件夹
    n_digits = 5  # 文件名数字位数，例如00001.png
    save_frames_from_video(video_path, output_folder, n_digits)