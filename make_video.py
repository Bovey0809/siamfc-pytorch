import cv2
import os
import glob

def frames_to_video(frame_dir, output_path, fps=30):
    # Get all frames
    frames = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.jpg')))
    if not frames:
        print("No frames found in the directory!")
        return
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frames[0])
    height, width = first_frame.shape[:2]
    
    # Initialize video writer with H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        fps, 
        (width, height),
        isColor=True
    )
    
    if not out.isOpened():
        print("Failed to initialize video writer!")
        return
    
    # Write frames to video
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        out.write(frame)
        print(f"Processing {os.path.basename(frame_path)}")
    
    # Release video writer
    out.release()
    print(f"\nVideo saved to {output_path}")

if __name__ == '__main__':
    frames_to_video('visualization', 'tracking_result.h264', fps=30) 