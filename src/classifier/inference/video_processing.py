from pathlib import Path
import numpy as np
from PIL import Image
from typing import Union, Generator
import cv2
import tempfile
from concurrent.futures import ThreadPoolExecutor


class VideoProcessor:
    def __init__(self, target_size: int, batch_workers: int, video_interval: int = 24):
        """
        Initialize VideoProcessor.
        
        Args:
            target_size: Target size for each frame (will be resized to target_size x target_size)
            batch_workers: Number of worker threads for parallel frame processing
            video_interval: Frame interval for processing (e.g., 24 means process every 24th frame)
        """
        self.target_size = target_size
        self.batch_workers = batch_workers
        self.video_interval = video_interval

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame using the same logic as ImageProcessor.
        
        Args:
            frame: Video frame as numpy array (BGR format from cv2)
            
        Returns:
            Processed frame as numpy array ready for model inference
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for consistency with ImageProcessor
        image = Image.fromarray(frame_rgb)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image proportionally and add black padding
        original_width, original_height = image.size

        # Calculate scale factor to fit within target_size while maintaining aspect ratio
        scale = min(self.target_size / original_width, self.target_size / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Resize image proportionally
        image = image.resize((new_width, new_height), Image.BILINEAR)

        # Create black canvas
        canvas = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))

        # Calculate position to paste (center the image)
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2

        # Paste resized image onto black canvas
        canvas.paste(image, (paste_x, paste_y))

        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(canvas).astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std

        # Convert HWC to CHW format
        image_array = np.transpose(image_array, (2, 0, 1))

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of extracted frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process frame at specified intervals
            if frame_count % self.video_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        
        print(f"[✓] Extracted {len(frames)} frames from {total_frames} total frames (interval: {self.video_interval})")
        
        return frames

    def process(
            self,
            video: Union[str, bytes],
    ) -> Generator[np.ndarray, None, None]:
        """
        Process video and yield batches of processed frames.
        
        Args:
            video: Video file path or video bytes
            
        Yields:
            Batches of processed frames as numpy arrays
        """
        # Handle bytes input by saving to temporary file
        if isinstance(video, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video)
                video_path = tmp_file.name
            temp_file = True
        else:
            video_path = video
            temp_file = False
        
        try:
            # Extract frames at specified intervals
            frames = self._extract_frames(video_path)
            
            if not frames:
                print("[!] No frames extracted from video")
                return
            
            # Process frames in parallel
            print(f"[→] Processing {len(frames)} frames...")
            
            with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
                futures = [
                    executor.submit(self._process_frame, frame)
                    for frame in frames
                ]
                
                processed_frames = [future.result() for future in futures]
                processed_frames = [frame for frame in processed_frames if frame is not None]
            
            print(f"[✓] Frame preprocessing complete: {len(processed_frames)}/{len(frames)}")
            
            # Stack all processed frames into a single batch
            if processed_frames:
                yield np.vstack(processed_frames)
        
        finally:
            # Clean up temporary file if created
            if temp_file:
                try:
                    Path(video_path).unlink()
                except Exception as e:
                    print(f"[!] Could not delete temporary file: {e}")

    def process_with_batches(
            self,
            video: Union[str, bytes],
            batch_size: int = 32,
    ) -> Generator[np.ndarray, None, None]:
        """
        Process video and yield frames in specified batch sizes.
        
        Args:
            video: Video file path or video bytes
            batch_size: Number of frames per batch
            
        Yields:
            Batches of processed frames as numpy arrays
        """
        # Handle bytes input by saving to temporary file
        if isinstance(video, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video)
                video_path = tmp_file.name
            temp_file = True
        else:
            video_path = video
            temp_file = False
        
        try:
            # Extract frames at specified intervals
            frames = self._extract_frames(video_path)
            
            if not frames:
                print("[!] No frames extracted from video")
                return
            
            print(f"[→] Processing {len(frames)} frames in batches of {batch_size}...")
            
            # Process frames in chunks
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                
                # Process batch in parallel
                with ThreadPoolExecutor(max_workers=self.batch_workers) as executor:
                    futures = [
                        executor.submit(self._process_frame, frame)
                        for frame in batch_frames
                    ]
                    
                    processed_frames = [future.result() for future in futures]
                    processed_frames = [frame for frame in processed_frames if frame is not None]
                
                if processed_frames:
                    batch_array = np.vstack(processed_frames)
                    print(f"[✓] Batch {i // batch_size + 1} processed: {len(processed_frames)} frames")
                    yield batch_array
        
        finally:
            # Clean up temporary file if created
            if temp_file:
                try:
                    Path(video_path).unlink()
                except Exception as e:
                    print(f"[!] Could not delete temporary file: {e}")

    def get_video_info(self, video: Union[str, bytes]) -> dict:
        """
        Get information about the video.
        
        Args:
            video: Video file path or video bytes
            
        Returns:
            Dictionary containing video information
        """
        # Handle bytes input by saving to temporary file
        if isinstance(video, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_file.write(video)
                video_path = tmp_file.name
            temp_file = True
        else:
            video_path = video
            temp_file = False
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            info = {
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration_seconds': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'frames_to_process': (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self.video_interval) + 1,
            }
            
            cap.release()
            
            return info
        
        finally:
            # Clean up temporary file if created
            if temp_file:
                try:
                    Path(video_path).unlink()
                except Exception as e:
                    print(f"[!] Could not delete temporary file: {e}")
