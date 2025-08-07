import time
from webbrowser import get
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

    @property
    def camparams_path(self):
        os.makedirs(self.save_path, exist_ok=True)
        return os.path.join(self.save_path, 'camera_params.yaml')


def controlled_capture(path_config: PathConfigs = None):
    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # exposure should be smaller than 1/30s
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # Left IR
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # Right IR

    # Start the pipeline
    profile = pipeline.start(config)
    device = profile.get_device()
    sensors = device.query_sensors()
    print(sensors)

    try:
        while True:
            time.sleep(0.5)  # Wait for 0.5 seconds (alternative condition for software trigger)
            aligned_frames = pipeline.wait_for_frames()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            left_ir_frame = aligned_frames.get_infrared_frame(1)
            right_ir_frame = aligned_frames.get_infrared_frame(2)

            if not color_frame or not depth_frame:
                continue

            # Intrinsics & Extrinsics
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            color_to_depth_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
            depth_to_color_extrin = color_frame.profile.get_extrinsics_to(depth_frame.profile)
            print("color", color_intrin)

            if not left_ir_frame or not right_ir_frame:
                print("Infrared frames not available.")
                continue

            left_ir_intrin = left_ir_frame.profile.as_video_stream_profile().intrinsics
            right_ir_intrin = right_ir_frame.profile.as_video_stream_profile().intrinsics
            left_ir_to_depth_extrin = left_ir_frame.profile.get_extrinsics_to(depth_frame.profile)
            right_ir_to_depth_extrin = right_ir_frame.profile.get_extrinsics_to(depth_frame.profile)

            # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            camera_params = {
                'depth_intrinsics': get_all_attributes(depth_intrin),
                'color_intrinsics': get_all_attributes(color_intrin),
                'color_to_depth_extrinsics': extr2matrix(color_to_depth_extrin),
                'depth_to_color_extrinsics': extr2matrix(depth_to_color_extrin),
                'left_ir_intrinsics': get_all_attributes(left_ir_intrin),
                'right_ir_intrinsics': get_all_attributes(right_ir_intrin),
                'left_ir_to_depth_extrinsics': extr2matrix(left_ir_to_depth_extrin),
                'right_ir_to_depth_extrinsics': extr2matrix(right_ir_to_depth_extrin),
                'depth_scale': depth_scale
            }

            # Map depth to color
            depth_pixel = [240, 320]   # Random pixel
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_scale)
            color_point = rs.rs2_transform_point_to_point(depth_to_color_extrin, depth_point)
            color_pixel = rs.rs2_project_point_to_pixel(color_intrin, color_point)

            camparams_path = path_config.camparams_path
            import yaml
            with open(camparams_path, 'w') as f:
                yaml.dump(camera_params, f, default_flow_style=False)
            print(f"Camera parameters saved to {camparams_path}")
            exit()

    finally:
        # Stop the pipeline
        # pipeline.stop()
        cv2.destroyAllWindows()


def get_all_attributes(obj):
    res = {}
    for att in dir(obj):
        if not att.startswith('_'):
            try:
                np.asanyarray(getattr(obj, att))
            except:
                continue
            res[att] = getattr(obj, att)
    return res


def extr2matrix(extr):
    """
    Convert extrinsics to a 4x4 matrix.
    :param extr: Extrinsics object
    :return: 4x4 numpy array
    """
    extr_matrix = np.eye(4)
    extr_matrix[:3, :3] = np.asanyarray(extr.rotation).reshape(3, 3)
    extr_matrix[:3, 3] = np.asanyarray(extr.translation)
    res = {
        'rotation': extr.rotation,
        'translation': extr.translation,
        'matrix': extr_matrix.tolist()  # Convert to list for YAML compatibility
    }
    return res


def get_formated_timestamp():
    import datetime
    # in JST timezone with format: YYYYMMDD_HHMMSS
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("capture_%Y%m%d_%H%M%S")
    return now


if __name__ == "__main__":
    import argparse
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='RealSense Camera Control Example')
    parser.add_argument('--output_root', type=str, default='capture', help='Root directory to save captured images (default: capture)')
    args = parser.parse_args()
    # Define camera settings
    path_config = PathConfigs(os.path.join(args.output_root, get_formated_timestamp()))
    # Start capture
    controlled_capture(path_config=path_config)
