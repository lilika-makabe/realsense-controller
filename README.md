# RealSense Controller
A Python application for controlled image capture using Intel RealSense cameras with manual exposure, gain, and white balance settings. 
This script provides precise control over camera parameters and captures RGB, depth, and infrared images simultaneously.
<!-- 
## Features

- **Manual Camera Control**: Set exposure time, gain, and white balance manually
- **Multi-stream Capture**: Captures RGB, depth, left IR, and right IR images simultaneously
- **Real-time Preview**: Live display of all camera streams in a combined view
- **Flexible Output**: Save images in PNG or EXR formats with timestamps
- **Parameter Validation**: Automatic validation of camera parameter ranges
- **Command Line Interface**: Easy-to-use CLI with customizable parameters -->

## Hardware Requirements
- Intel RealSense camera (tested with D435i)

## Installation

```bash
pip install pyrealsense2 opencv-python "numpy<2.0"
```
## Usage

### Basic Usage

```bash
python src/controlled_capture.py
```

### Advanced Usage with Custom Parameters

```bash
python src/controlled_capture.py --exposure 150 --gain 80 --white_balance 4000 --save --output_root my_captures
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--exposure` | int | 100 | Exposure time in microseconds |
| `--gain` | int | 64 | Gain value (typically 0-128) |
| `--white_balance` | int | 3500 | White balance color temperature |
| `--fps` | int | 30 | Frames per second |
| `--save` | flag | False | Save captured images to disk |
| `--ignore_exposure_warnings` | flag | False | Ignore exposure/FPS warnings |
| `--output_root` | str | "capture" | Root directory for saved images |

## Output Structure

When saving is enabled, images are organized in the following structure:

```
capture_YYYYMMDD_HHMMSS/
├── color/           # RGB images (.png)
├── depth/           # Depth images (.exr)
├── ir_left/         # Left IR images (.exr)
├── ir_right/        # Right IR images (.exr)
└── camera_config.yaml  # Camera settings used
```

## Key Controls

- **Q**: Quit the application
- Images are automatically captured every 0.5 seconds when `--save` is enabled