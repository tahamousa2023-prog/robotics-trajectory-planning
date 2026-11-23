#!/usr/bin/env python3
"""
Modular Object Detection and Size Estimation System
Currently implements: Color Segmentation
Easy to add: SAM, YOLO, Depth-based methods
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# ABSTRACT BASE CLASS - All detectors inherit from this
# ============================================================================

class ObjectDetector(ABC):
    """
    Abstract base class for object detection methods
    All detector implementations must inherit from this
    """
    
    @abstractmethod
    def detect(self, image):
        """
        Detect object in image and return mask
        
        Args:
            image: RGB image (numpy array, H×W×3)
            
        Returns:
            mask: Binary mask (numpy array, H×W, dtype=uint8)
                  255 = object, 0 = background
        """
        pass
    
    @abstractmethod
    def get_name(self):
        """Return detector name for logging"""
        pass


# ============================================================================
# DETECTOR IMPLEMENTATIONS
# ============================================================================

class ColorSegmentationDetector(ObjectDetector):
    """
    Color-based segmentation detector
    Fast, simple, works great in simulation
    """
    
    def __init__(self, color_range='auto', min_area=100):
        """
        Args:
            color_range: 'auto' or dict with 'lower' and 'upper' HSV bounds
            min_area: Minimum object area in pixels (filters noise)
        """
        self.color_range = color_range
        self.min_area = min_area
        
    def detect(self, image):
        """Detect object using color segmentation"""
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define color range
        if self.color_range == 'auto':
            # Auto-detect: assume object is different from background
            # Use saturation and value to find colorful/bright objects
            lower = np.array([0, 30, 30])      # Any hue, some saturation, some brightness
            upper = np.array([180, 255, 255])  # Full range
        else:
            lower = np.array(self.color_range['lower'])
            upper = np.array(self.color_range['upper'])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find contours and keep only largest
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # No object found, return empty mask
            return np.zeros_like(mask)
        
        # Filter by area and get largest
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        if len(valid_contours) == 0:
            return np.zeros_like(mask)
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Create clean mask with only largest object
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)
        
        return clean_mask
    
    def get_name(self):
        return "ColorSegmentation"


class SAMDetector(ObjectDetector):
    """
    Segment Anything Model (SAM) detector
    Uses center-point prompting to segment object in center of image
    """
    
    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """
        Initialize SAM model
        
        Args:
            model_type: 'vit_h' (huge), 'vit_l' (large), or 'vit_b' (base)
            checkpoint_path: Path to SAM checkpoint, or None to auto-download
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError:
            raise ImportError(
                "SAM not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Default checkpoint paths
        if checkpoint_path is None:
            checkpoint_paths = {
                'vit_h': 'checkpoints/sam_vit_h_4b8939.pth',
                'vit_l': 'checkpoints/sam_vit_l_0b3195.pth',
                'vit_b': 'checkpoints/sam_vit_b_01ec64.pth'
            }
            checkpoint_path = checkpoint_paths.get(model_type, checkpoint_paths['vit_h'])
        
        # Load SAM model
        print(f"Loading SAM model ({model_type})...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        
        # Move to GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        
        self.predictor = SamPredictor(sam)
        self.device = device
        print(f"✓ SAM loaded on {device}")
    
    def detect(self, image):
        """
        Detect object using SAM with center-point prompt
        
        Args:
            image: RGB image (H×W×3 numpy array)
            
        Returns:
            mask: Binary mask (H×W, 255=object, 0=background)
        """
        # Set image
        self.predictor.set_image(image)
        
        # Use center of image as prompt point
        h, w = image.shape[:2]
        center_point = np.array([[w // 2, h // 2]])
        center_label = np.array([1])  # 1 = foreground point
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True  # Get 3 masks with different granularities
        )
        
        # Select best mask (highest score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        # Convert to uint8 (0 or 255)
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        
        return mask_uint8
    
    def get_name(self):
        return "SAM"


class YOLODetector(ObjectDetector):
    """
    YOLO-based object detector
    TODO: Implement when YOLO is needed
    """
    
    def __init__(self, model_path=None):
        """Initialize YOLO model"""
        # TODO: Load YOLO model
        raise NotImplementedError("YOLO detector not yet implemented")
    
    def detect(self, image):
        """Detect object using YOLO"""
        # TODO: Implement YOLO detection
        raise NotImplementedError("YOLO detector not yet implemented")
    
    def get_name(self):
        return "YOLO"


# ============================================================================
# SIZE ESTIMATOR - Works with any detector
# ============================================================================

class ObjectSizeEstimator:
    """
    Estimates real-world object size from image mask
    Uses camera geometry (pinhole camera model)
    """
    
    def __init__(self, camera_height=0.7, image_width_px=1280, image_height_px=720, 
                 fov_horizontal_deg=60):
        """
        Args:
            camera_height: Height of camera above table (meters)
            image_width_px: Image width in pixels
            image_height_px: Image height in pixels
            fov_horizontal_deg: Camera horizontal field of view (degrees)
        """
        self.camera_height = camera_height
        self.image_width_px = image_width_px
        self.image_height_px = image_height_px
        self.fov_horizontal_deg = fov_horizontal_deg
        
        # Calculate meters per pixel at table level
        # tan(fov/2) = (width_real/2) / height
        fov_rad = np.deg2rad(fov_horizontal_deg)
        table_width_visible = 2 * camera_height * np.tan(fov_rad / 2)
        self.meters_per_pixel_x = table_width_visible / image_width_px
        
        # Assume square pixels (same for Y)
        self.meters_per_pixel_y = self.meters_per_pixel_x
        
    def estimate_size_from_mask(self, mask):
        """
        Estimate object size from binary mask
        
        Args:
            mask: Binary mask (H×W, 255=object, 0=background)
            
        Returns:
            dict with keys:
                'width_m': Object width in meters
                'height_m': Object height in meters  
                'center_x_m': Object center X in meters (from image center)
                'center_y_m': Object center Y in meters (from image center)
                'bbox_px': Bounding box in pixels (x, y, w, h)
        """
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get bounding box in pixels
        x, y, w_px, h_px = cv2.boundingRect(contours[0])
        
        # Convert to meters
        width_m = w_px * self.meters_per_pixel_x
        height_m = h_px * self.meters_per_pixel_y
        
        # Calculate center position relative to image center
        center_x_px = x + w_px / 2
        center_y_px = y + h_px / 2
        
        # Convert to meters from image center
        center_x_m = (center_x_px - self.image_width_px / 2) * self.meters_per_pixel_x
        center_y_m = (center_y_px - self.image_height_px / 2) * self.meters_per_pixel_y
        
        return {
            'width_m': width_m,
            'height_m': height_m,
            'center_x_m': center_x_m,
            'center_y_m': center_y_m,
            'bbox_px': (x, y, w_px, h_px),
            'area_m2': width_m * height_m
        }


# ============================================================================
# DETECTION PIPELINE - Combines detector + size estimator
# ============================================================================

class ObjectDetectionPipeline:
    """
    Complete detection pipeline
    Detector (get mask) → Size Estimator (get real-world size)
    """
    
    def __init__(self, detector, size_estimator):
        """
        Args:
            detector: ObjectDetector instance
            size_estimator: ObjectSizeEstimator instance
        """
        self.detector = detector
        self.size_estimator = size_estimator
        
    def detect_and_measure(self, image, visualize=False):
        """
        Detect object and estimate its size
        
        Args:
            image: RGB image (numpy array)
            visualize: If True, return visualization image
            
        Returns:
            results: dict with detection results
            vis_image: visualization (if visualize=True)
        """
        
        # Step 1: Detect object (get mask)
        mask = self.detector.detect(image)
        
        # Step 2: Estimate size from mask
        size_info = self.size_estimator.estimate_size_from_mask(mask)
        
        # Combine results
        results = {
            'detector': self.detector.get_name(),
            'detected': size_info is not None,
            'mask': mask,
            'size': size_info
        }
        
        # Optional visualization
        if visualize and size_info is not None:
            vis_image = self._create_visualization(image, mask, size_info)
            return results, vis_image
        
        return results
    
    def _create_visualization(self, image, mask, size_info):
        """Create visualization with bbox and measurements"""
        vis = image.copy()
        
        # Overlay mask (semi-transparent green)
        mask_overlay = np.zeros_like(vis)
        mask_overlay[mask > 0] = [0, 255, 0]  # Green
        vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        
        # Draw bounding box
        x, y, w, h = size_info['bbox_px']
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Add text with measurements
        text = f"Size: {size_info['width_m']*100:.1f}cm × {size_info['height_m']*100:.1f}cm"
        cv2.putText(vis, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
        
        return vis


# ============================================================================
# FACTORY FUNCTION - Easy way to create pipelines
# ============================================================================

def create_detection_pipeline(method='color', camera_height=0.7, 
                              image_width=1280, image_height=720,
                              fov_deg=60, **detector_kwargs):
    """
    Factory function to create detection pipeline
    
    Args:
        method: 'color', 'sam', or 'yolo'
        camera_height: Camera height above table (meters)
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_deg: Horizontal field of view (degrees)
        **detector_kwargs: Additional args for detector
        
    Returns:
        ObjectDetectionPipeline instance
    """
    
    # Create detector
    if method == 'color':
        detector = ColorSegmentationDetector(**detector_kwargs)
    elif method == 'sam':
        # SAM-specific parameters
        model_type = detector_kwargs.get('model_type', 'vit_h')
        checkpoint_path = detector_kwargs.get('checkpoint_path', None)
        detector = SAMDetector(model_type=model_type, checkpoint_path=checkpoint_path)
    elif method == 'yolo':
        detector = YOLODetector(**detector_kwargs)
    else:
        raise ValueError(f"Unknown detection method: {method}")
    
    # Create size estimator
    size_estimator = ObjectSizeEstimator(
        camera_height=camera_height,
        image_width_px=image_width,
        image_height_px=image_height,
        fov_horizontal_deg=fov_deg
    )
    
    # Combine into pipeline
    pipeline = ObjectDetectionPipeline(detector, size_estimator)
    
    return pipeline


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use the modular detection system
    """
    
    # Create pipeline with color segmentation
    pipeline = create_detection_pipeline(
        method='color',
        camera_height=0.7,      # 70cm above table
        image_width=1280,
        image_height=720,
        fov_deg=60,
        min_area=500            # Filter small noise
    )
    
    # Load test image (replace with actual camera image)
    # image = cv2.imread('test_image.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # For demo, create fake image
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:, :] = [100, 100, 100]  # Gray background
    # Draw fake object (red square)
    image[300:420, 500:620] = [255, 0, 0]
    
    # Run detection
    results = pipeline.detect_and_measure(image, visualize=True)
    
    # Access results
    if isinstance(results, tuple):
        results, vis_image = results
    
    if results['detected']:
        print(f"\n✓ Object detected using {results['detector']}")
        print(f"  Width:  {results['size']['width_m']*100:.2f} cm")
        print(f"  Height: {results['size']['height_m']*100:.2f} cm")
        print(f"  Center: ({results['size']['center_x_m']*100:.1f}, {results['size']['center_y_m']*100:.1f}) cm from image center")
        print(f"  Area:   {results['size']['area_m2']*10000:.2f} cm²")
    else:
        print("\n✗ No object detected")
    
    print("\n" + "="*70)
    print("MODULAR DESIGN - Easy to swap methods:")
    print("="*70)
    print("Current:  pipeline = create_detection_pipeline(method='color')")
    print("Future:   pipeline = create_detection_pipeline(method='sam')")
    print("          pipeline = create_detection_pipeline(method='yolo')")
    print("Same API, different detector! ✓")