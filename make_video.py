import os
import glob
import subprocess

def frames_to_video(frame_dir, output_path, fps=30):
    # Get all frames
    frames = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.jpg')))
    if not frames:
        print("No frames found in the directory!")
        return
    
    # Use ffmpeg to create video with MPEG4 codec
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', os.path.join(frame_dir, 'frame_*.jpg'),
        '-c:v', 'mpeg4',  # Use MPEG4 codec
        '-q:v', '1',      # Highest quality
        output_path
    ]
    
    print("Creating video...")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"\nVideo saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")

if __name__ == '__main__':
    frames_to_video('visualization', 'tracking_result.mp4', fps=30) 