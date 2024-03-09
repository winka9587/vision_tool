import os
import time
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs

"""
    pyrealsense2: https://github.com/cansik/pyrealsense2-macosx
"""


def capture_frames(n=5):
    """
    Captures frames from the RealSense camera and saves color, depth, and IMU data.

    Press 's' to start recording frames.
    Press 't' to stop recording frames.

    Frames are saved in the specified path with the following structure:
    - path/
        - color/
            - 0.png
            - 1.png
            ...
        - depth/
            - 0.png
            - 1.png
            ...
        - imu/
            - 0.txt
            - 1.txt
            ...

    Returns:
        None
    """

    width = 640
    height = 480
    fps = 30

    print(f"Found {len(rs.context().query_devices())} realsense devices!")

    # Configure depth, color, and IMU streams
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)  # 1280, 720

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imu", action="store_true", help="Enable IMU data collection")
    args = parser.parse_args()

    if args.imu:
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print("Depth Scale is: ", depth_scale)

    depth_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    color_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics

    print("color intrinsics is: ", depth_intrinsics)
    print("depth intrinsics is: ", color_intrinsics)

    align_to = rs.stream.color
    align = rs.align(align_to)  # align depth to color

    path = "F:/dataset/capture"

    input_key = -1
    record = False
    count = 0


    try:
        while True:
            input_key = cv2.waitKey(1)
            if input_key == ord('s'):
                if not record:
                    record = True
                    print("start record!")
                    second_path = str(time.time())
                    print("new dir is " + second_path)
                    os.mkdir(path + second_path)
                    os.mkdir(path + second_path + "/color")
                    os.mkdir(path + second_path + "/depth")
                    if args.imu:
                        os.mkdir(path + second_path + "/imu")
                    path = path + second_path + "/"
            elif input_key == ord('t'):
                print("stop record!")
                break

            # Wait for a coherent set of frames: depth, color, and IMU
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if args.imu:
                motion_frame = aligned_frames.first_or_default(rs.stream.accel)
                motion_data = motion_frame.as_motion_frame().get_motion_data()

            # without padding
            # if record:
            #     cv2.imwrite(path + "/color/" + str(count) + ".png", color_image)
            #     cv2.imwrite(path + "/depth/" + str(count) + ".png", depth_image)
            #     if args.imu:
            #         np.savetxt(path + "/imu/" + str(count) + ".txt", motion_data)
            #     print(count)
            #     count += 1
            # padding zeros
            if record:
                # 使用str.zfill()方法填充count
                padded_count = str(count).zfill(n)
                # 或者使用str.format()方法填充count
                # padded_count = "{:0{n}d}".format(count, n=n)

                cv2.imwrite(f"{path}/color/{padded_count}.png", color_image)
                cv2.imwrite(f"{path}/depth/{padded_count}.png", depth_image)
                if args.imu:
                    # 保存IMU数据时同样应用填充
                    np.savetxt(f"{path}/imu/{padded_count}.txt", motion_data)
                print(count)
                count += 1

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Show images
            cv2.imshow('depth', depth_colormap)
            cv2.imshow('color', color_image)
    finally:
        pipeline.stop()  # Stop streaming

capture_frames()

