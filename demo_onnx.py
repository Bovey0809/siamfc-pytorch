import cv2
import numpy as np
import onnxruntime
import torch
import os
import random
from siamfc.siamfc import ops

class ONNXTracker:
    def __init__(self, model_path='siamfc.onnx'):
        """Initialize tracker with ONNX model."""
        self.session = onnxruntime.InferenceSession(model_path)
        
        # Print model information
        print("\nModel Inputs:")
        for input in self.session.get_inputs():
            print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
        
        print("\nModel Outputs:")
        for output in self.session.get_outputs():
            print(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
        
        # Config parameters (from original SiamFC)
        self.scale_num = 3
        self.scale_step = 1.0375
        self.scale_lr = 0.59
        self.scale_penalty = 0.9745
        self.window_influence = 0.176
        self.response_sz = 17
        self.response_up = 16
        self.total_stride = 8
        
        # Calculate scale factors
        self.scale_factors = self.scale_step ** np.linspace(
            -(self.scale_num // 2),
            self.scale_num // 2, self.scale_num)
        
    def init(self, img, box):
        """Initialize tracker with first frame and bbox."""
        # Convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]
        
        # Create hanning window
        self.upscale_sz = self.response_up * self.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()
        
        # Calculate exemplar and search sizes
        context = 0.5 * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * 255 / 127
        
        # Get average color for padding
        self.avg_color = np.mean(img, axis=(0, 1))
        
        # Extract template features (self.kernel in original code)
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=127,
            border_value=self.avg_color)
        self.template = self._preprocess_image(z, 127)  # Store the original template
        
        # Create dummy search image with correct size (255x255)
        dummy_search = np.zeros((255, 255, 3), dtype=np.float32)
        dummy_search = self._preprocess_image(dummy_search, 255)
        
        # Run template through model to get features
        self.template_features = self.session.run(
            ['response'],
            {
                'template': self.template,
                'search': dummy_search
            }
        )[0]
        print(f"Template features shape: {self.template_features.shape}")
    
    def update(self, img):
        """Update tracker with new frame."""
        # Search images at different scales
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=255,
            border_value=self.avg_color) for f in self.scale_factors]
        
        # Run inference for each scale
        responses = []
        for scale_idx in range(len(x)):
            # Preprocess one search image at a time
            search = self._preprocess_image(x[scale_idx], 255)
            
            # Get response directly
            response = self.session.run(
                ['response'],
                {
                    'template': self.template,  # Use original template image
                    'search': search  # One scale at a time
                }
            )[0]
            responses.append(response[0, 0])  # Remove batch and channel dims
        responses = np.stack(responses, axis=0)
        
        # Upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.scale_num // 2] *= self.scale_penalty
        responses[self.scale_num // 2 + 1:] *= self.scale_penalty
        
        # Find scale with highest peak response
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))
        
        # Find peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.window_influence) * response + \
            self.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        
        # Calculate target center displacement
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.total_stride / self.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / 255
        self.center += disp_in_image
        
        # Update target size
        scale = (1 - self.scale_lr) * 1.0 + \
            self.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale
        
        # Convert back to 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])
        
        return box
    
    def _preprocess_image(self, img, target_size):
        """Preprocess image for the model."""
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Add batch dimension and convert to NCHW format
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
        return img.astype(np.float32)

def process_test_sequence(test_folder, output_folder, init_box):
    """Process a test sequence and save frames with tracking results."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    if not frame_files:
        raise Exception(f"No image files found in {test_folder}")
    
    # Initialize tracker with first frame
    first_frame = cv2.imread(os.path.join(test_folder, frame_files[0]))
    if first_frame is None:
        raise Exception(f"Failed to read first frame: {frame_files[0]}")
    
    tracker = ONNXTracker()
    tracker.init(first_frame, init_box)
    
    # Process each frame
    for frame_file in frame_files:
        frame_path = os.path.join(test_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Failed to read frame {frame_file}")
            continue
        
        # Update tracker
        box = tracker.update(frame)
        
        # Draw result
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save frame
        output_path = os.path.join(output_folder, frame_file)
        cv2.imwrite(output_path, frame)
        print(f"Saved frame: {output_path}")

def read_init_box(list_file):
    """Read initial bounding box from list.txt file."""
    with open(list_file, 'r') as f:
        line = f.readline().strip()
        # Assuming format: x,y,w,h
        x, y, w, h = map(float, line.split(','))
        return [x, y, w, h]

if __name__ == '__main__':
    # Base directories
    test_base = '/home/pytorch/data/test'
    output_base = os.getcwd()  # Use current working directory
    
    # Get random test folder
    test_folders = [f for f in os.listdir(test_base) if os.path.isdir(os.path.join(test_base, f))]
    if not test_folders:
        raise Exception("No test folders found in /home/pytorch/data/test")
    
    test_folder = random.choice(test_folders)
    test_path = os.path.join(test_base, test_folder)
    output_path = os.path.join(output_base, test_folder)
    
    # Read initial bbox
    list_file = os.path.join(test_path, 'groundtruth.txt')
    if not os.path.exists(list_file):
        raise Exception(f"list.txt not found in {test_path}")
    
    init_box = read_init_box(list_file)
    
    # Process sequence
    print(f"Processing sequence: {test_folder}")
    process_test_sequence(test_path, output_path, init_box)
    print("Done!") 