import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=30):
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc=f"Обработка {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Сохраняем каждый N-ый кадр
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

def process_dataset(input_root, output_root):

    video_extensions = ('.mp4', '.avi', '.mov')
    
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, input_root)
                output_dir = os.path.join(output_root, relative_path)
                
                extract_frames(video_path, output_dir)

                
if __name__ == "__main__":

    BASE_DIR = "celeb"
    OUTPUT_DIR = "images"
    

    process_dataset(
        os.path.join(BASE_DIR, "Celeb-real"),
        os.path.join(OUTPUT_DIR, "real")
    )
    
    process_dataset(
        os.path.join(BASE_DIR, "Celeb-synthesis"),
        os.path.join(OUTPUT_DIR, "fake")
    )