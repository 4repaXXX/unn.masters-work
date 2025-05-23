import os
import cv2
from tqdm import tqdm
import argparse

def extract_frames(video_path, output_dir, frame_interval=30):
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0

    with tqdm(total=total_frames, desc=f"Processing {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save every N-th frame
            if frame_count % frame_interval == 0:
                output_path = os.path.join(
                    output_dir,
                    f"{video_name}_{saved_count:04d}.jpg"
                )
                cv2.imwrite(output_path, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()

def process_video_list(input_file, output_root):
    """
    Process a file containing video paths and labels
    File format: label(0/1) video_path
    Example: 1 YouTube-real/00170.mp4
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            # Split into label and path
            label, video_path = line.split(maxsplit=1)
            label = label.strip()
            video_path = video_path.strip()

            # Validate label
            if label not in ('0', '1'):
                print(f"Invalid label {label} for video {video_path}. Skipping...")
                continue

            # Create output directory based on label
            output_dir = os.path.join(output_root, label)

            # Extract frames
            extract_frames(video_path, output_dir)

        except ValueError:
            print(f"Invalid line format: {line}. Skipping...")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos based on label file.')
    parser.add_argument('input_file', help='Path to file containing video paths and labels')
    parser.add_argument('output_dir', help='Root directory for output frames')

    args = parser.parse_args()

    # Process the video list
    process_video_list(args.input_file, args.output_dir)
