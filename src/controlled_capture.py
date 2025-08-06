import time
import cv2
import numpy as np
import pyrealsense2 as rs
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Enable OpenEXR support in OpenCV

"""
Controlled Capture for Intel RealSense Cameras
This script captures images from Intel RealSense cameras with controlled settings.
Operation checked with D435i camera.
"""

# TODO: Convert to a capture class


class CamConfigs:
    def __init__(self, exposure=100, gain=64, white_balance=4600, fps=30, ignore_exposure_warnings=False):
        self.exposure = exposure
        self.gain = gain
        self.white_balance = white_balance
        self.fps = fps
        self.ignore_exposure_warnings = ignore_exposure_warnings

    def __str__(self):
        return f"Exposure: {self.exposure}, Gain: {self.gain}, White Balance: {self.white_balance}, FPS: {self.fps}, Ignore Warnings: {self.ignore_exposure_warnings}"


class PathConfigs:
    def __init__(self, save_path="capture"):
        self.save_path = save_path

    # return image/depth/ir_left/ir_right directories as attributes
    @property
    def image_path(self):
        return os.path.join(self.save_path, 'color')

    @property
    def depth_path(self):
        return os.path.join(self.save_path, 'depth')

    @property
    def ir_left_path(self):
        return os.path.join(self.save_path, 'ir_left')

    @property
    def ir_right_path(self):
        return os.path.join(self.save_path, 'ir_right')

    @property
    def config_path(self):
        os.makedirs(self.save_path, exist_ok=True)
        return os.path.join(self.save_path, 'camera_config.yaml')


def set_realsense_options(sensors, config: CamConfigs):
    # Turn off all automatic adjustments
    for sensor in sensors:
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            # Turn off all automatic adjustments
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.enable_auto_white_balance, 0)
            # sensor.set_option(rs.option.auto_exposure_priority, 0)

            # check if the given values are within the valid range
            print(sensor.get_option_range(rs.option.exposure).min, sensor.get_option_range(rs.option.exposure).max, sensor.get_option_range(rs.option.exposure).step)
            assert config.exposure >= sensor.get_option_range(rs.option.exposure).min and config.exposure <= sensor.get_option_range(rs.option.exposure).max, f"Exposure value {config.exposure} is out of range: {sensor.get_option_range(rs.option.exposure)}"
            assert config.gain >= sensor.get_option_range(rs.option.gain).min and config.gain <= sensor.get_option_range(rs.option.gain).max, f"Gain value {config.gain} is out of range: {sensor.get_option_range(rs.option.gain)}"
            assert config.white_balance >= sensor.get_option_range(rs.option.white_balance).min and config.white_balance <= sensor.get_option_range(rs.option.white_balance).max, f"White Balance value {config.white_balance} is out of range: {sensor.get_option_range(rs.option.white_balance)}"
            if config.ignore_exposure_warnings:
                if config.fps < config.exposure:
                    print(f"Warning: FPS {config.fps} is less than exposure {config.exposure} in microseconds. This may lead to unexpected behavior.")
            else:
                assert config.fps >= config.exposure, f"Warning: FPS {config.fps} must be greater than or equal to exposure {config.exposure} in microseconds"

            # Apply settings
            sensor.set_option(rs.option.exposure, config.exposure)         # Unit: microseconds
            sensor.set_option(rs.option.gain, config.gain)              # Usually 0~128
            sensor.set_option(rs.option.white_balance, config.white_balance)   # Color temperature (e.g.: 4600K)
            time.sleep(0.5)  # Wait a bit for settings to take effect

        if sensor.get_info(rs.camera_info.name) == 'Stereo Module':
            sensor.set_option(rs.option.enable_auto_exposure, 0)


def print_realsense_options(sensors):
    for sensor in sensors:
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            exposure = sensor.get_option(rs.option.exposure)
            gain = sensor.get_option(rs.option.gain)
            white_balance = sensor.get_option(rs.option.white_balance)
            auto_exposure = sensor.get_option(rs.option.enable_auto_exposure)
            auto_white_balance = sensor.get_option(rs.option.enable_auto_white_balance)

            print(f"Exposure: {exposure}")
            print(f"Gain: {gain}")
            print(f"White Balance: {white_balance}")
            print(f"Auto Exposure Enabled: {auto_exposure}")
            print(f"Auto White Balance Enabled: {auto_white_balance}")


def save_realsense_options(sensors, path):
    # Save current camera settings to a file
    with open(path, 'w') as f:
        for sensor in sensors:
            if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                exposure = sensor.get_option(rs.option.exposure)
                gain = sensor.get_option(rs.option.gain)
                white_balance = sensor.get_option(rs.option.white_balance)
                auto_exposure = sensor.get_option(rs.option.enable_auto_exposure)
                auto_white_balance = sensor.get_option(rs.option.enable_auto_white_balance)

                f.write(f"Exposure: {exposure}\n")
                f.write(f"Gain: {gain}\n")
                f.write(f"White Balance: {white_balance}\n")
                f.write(f"Auto Exposure Enabled: {auto_exposure}\n")
                f.write(f"Auto White Balance Enabled: {auto_white_balance}\n")


def sanitary_check_camconfigs(sensors, config: CamConfigs):
    # Sanitary check for camera settings
    for sensor in sensors:
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            exposure = sensor.get_option(rs.option.exposure)
            gain = sensor.get_option(rs.option.gain)
            white_balance = sensor.get_option(rs.option.white_balance)
            assert gain == config.gain, f"Gain mismatch: {gain} != {config.gain}"
            assert white_balance == config.white_balance, f"White Balance mismatch: {white_balance} != {config.white_balance}"
            assert sensor.get_option(rs.option.enable_auto_exposure) == 0, "Auto Exposure should be disabled"
            assert sensor.get_option(rs.option.enable_auto_white_balance) == 0, "Auto White Balance should be disabled"
            if not config.ignore_exposure_warnings:
                assert exposure == config.exposure, f"Exposure mismatch: {exposure} != {config.exposure}"


def controlled_capture(cam_config: CamConfigs, save=False, path_config: PathConfigs = None):
    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # # Enable RGB and depth streams
    # print(rs.stream)
    # for att in dir(rs.stream):
    #     if not att.startswith('_'):
    #         print(att, getattr(rs.stream, att))

    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # exposure should be smaller than 1/30s
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # Left IR
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # Right IR

    # Start the pipeline
    profile = pipeline.start(config)
    device = profile.get_device()
    sensors = device.query_sensors()
    print(sensors)

    print_realsense_options(sensors)
    set_realsense_options(sensors, cam_config)
    print("--->")
    print_realsense_options(sensors)
    save_realsense_options(sensors, path_config.config_path)

    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            time.sleep(0.5)  # Wait for 0.5 seconds (alternative condition for software trigger)
            # Get frames
            frames = pipeline.wait_for_frames()
            # Get RGB and depth images
            # color_frame = frames.get_color_frame()
            # depth_frame = frames.get_depth_frame()
            # Get and align frames
            aligned_frames = align.process(frames)
            # Get aligned RGB and Depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            ir_left_frame = aligned_frames.get_infrared_frame(1)
            ir_right_frame = aligned_frames.get_infrared_frame(2)

            if not color_frame or not depth_frame:
                continue

            # Convert frames to NumPy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_left_image = np.asanyarray(ir_left_frame.get_data())[..., None].repeat(3, axis=2)  # Convert IR image to 3 channels
            ir_right_image = np.asanyarray(ir_right_frame.get_data())[..., None].repeat(3, axis=2)  # Convert IR image to 3 channels

            # Display depth image in color
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Display RGB and depth images side by side
            combined_image = np.hstack((color_image, depth_colormap, ir_left_image, ir_right_image))
            cv2.imshow('RealSense Software Trigger Capture', combined_image)

            # Save captured images
            if save:
                timestamp = int(time.time())
                os.makedirs(path_config.image_path, exist_ok=True)
                os.makedirs(path_config.depth_path, exist_ok=True)
                os.makedirs(path_config.ir_left_path, exist_ok=True)
                os.makedirs(path_config.ir_right_path, exist_ok=True)
                cv2.imwrite(os.path.join(path_config.image_path, f'color_{timestamp}.png'), color_image)
                cv2.imwrite(os.path.join(path_config.depth_path, f'depth_{timestamp}.exr'), depth_image.astype(np.float32))
                cv2.imwrite(os.path.join(path_config.ir_left_path, f'ir_left_{timestamp}.exr'), ir_left_image.astype(np.float32))
                cv2.imwrite(os.path.join(path_config.ir_right_path, f'ir_right_{timestamp}.exr'), ir_right_image.astype(np.float32))
                # Save color and depth images with timestamp
                # cv2.imwrite(f'color_{timestamp}.png', color_image)
                # cv2.imwrite(f'depth_{timestamp}.png', depth_colormap)
                print(f"Captured images at timestamp {timestamp}")
            sanitary_check_camconfigs(sensors, cam_config)

            # Exit when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()


def get_formated_timestamp():
    import datetime
    # in JST timezone with format: YYYYMMDD_HHMMSS
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("capture_%Y%m%d_%H%M%S")
    return now


if __name__ == "__main__":
    import argparse
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='RealSense Camera Control Example')
    parser.add_argument('--exposure', type=int, default=100, help='Exposure time in microseconds (default: 19)')
    parser.add_argument('--gain', type=int, default=64, help='Gain value (default: 64)')
    parser.add_argument('--white_balance', type=int, default=3500, help='White balance value (default: 4600)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--save', action='store_true', help='Save captured images to disk')
    parser.add_argument('--ignore_exposure_warnings', action='store_true', help='Ignore warnings to avoid exposure>=FPS situation; the exposure will be different than the one specified.')
    parser.add_argument('--output_root', type=str, default='capture', help='Root directory to save captured images (default: capture)')
    args = parser.parse_args()
    # Define camera settings
    cam_config = CamConfigs(exposure=args.exposure, gain=args.gain, white_balance=args.white_balance, fps=args.fps, ignore_exposure_warnings=args.ignore_exposure_warnings)
    path_config = PathConfigs(os.path.join(args.output_root, get_formated_timestamp()))

    print(f"Using camera configuration: {cam_config}")
    # Start capture
    controlled_capture(cam_config, save=args.save, path_config=path_config)
