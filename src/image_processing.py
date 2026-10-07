#!/usr/bin/env python3
"""
Image Processing for Dust Removal App

Advanced image processing functions matching Spotless-Film's capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import cv2
from typing import Optional, Tuple, List
import threading
import time
from dataclasses import dataclass


# Import model architecture (copy from notebook)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
            )

        # 1 channel for greyscale
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.middle = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        m = self.middle(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(m), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


# TorchScript LaMa model (the same file lama-cleaner / IOPaint / simple-lama-inpainting use):
# https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt
LAMA_WEIGHTS_NAME = "big-lama.pt"


class LamaInpainter:
    """LaMa deep learning inpainting, loaded directly from the TorchScript weights file"""
    def __init__(self, weights_path: Optional[str] = None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.available = False

        if weights_path:
            try:
                self.model = torch.jit.load(weights_path, map_location=self.device).eval()
                if self.device.type != "cpu":
                    # The first two GPU runs are slow (JIT specialization, kernel tuning)
                    for _ in range(2):
                        self._run_model(np.zeros((640, 640, 3), np.uint8), np.zeros((640, 640), np.uint8), 640)
                self.available = True
                print(f"✅ LaMa inpainting model loaded from {weights_path}")
            except Exception as e:
                print(f"Failed to load LaMa: {e}")

    def inpaint(self, image: Image.Image, mask: Image.Image,
                progress_callback: Optional[callable] = None,
                tile_size: int = 512, margin: int = 64) -> Image.Image:
        """Inpaint masked pixels at full resolution.

        Only tiles that contain mask pixels are run through the model, each with a
        margin of surrounding context; pixels outside the mask are left untouched.
        """
        if not self.available:
            raise RuntimeError(f"LaMa model is not loaded ({LAMA_WEIGHTS_NAME} not found in weights/)")

        image_np = np.array(image.convert('RGB'))
        mask_np = np.array(mask.convert('L'))
        height, width = mask_np.shape
        result = image_np.copy()

        # Every crop has the same size (shifted inwards at the image borders): on GPU
        # each new input shape costs several seconds of kernel tuning
        window = tile_size + 2 * margin
        tiles = [(y, x) for y in range(0, height, tile_size) for x in range(0, width, tile_size)
                 if mask_np[y:y + tile_size, x:x + tile_size].any()]
        for i, (y, x) in enumerate(tiles):
            y0 = max(0, min(y - margin, height - window))
            x0 = max(0, min(x - margin, width - window))
            y1, x1 = min(height, y0 + window), min(width, x0 + window)
            out = self._run_model(image_np[y0:y1, x0:x1], mask_np[y0:y1, x0:x1], window)
            tile_mask = mask_np[y:y + tile_size, x:x + tile_size] > 0
            th, tw = tile_mask.shape
            result[y:y + th, x:x + tw][tile_mask] = out[y - y0:y - y0 + th, x - x0:x - x0 + tw][tile_mask]
            if progress_callback:
                progress_callback((i + 1) / len(tiles))

        print(f"✅ LaMa inpainting completed ({len(tiles)} tiles)")
        return Image.fromarray(result)

    def _run_model(self, image_np: np.ndarray, mask_np: np.ndarray, window: int) -> np.ndarray:
        """Run LaMa on one RGB crop, padded up to window x window (a multiple of 8)"""
        h, w = mask_np.shape
        pad_h, pad_w = window - h, window - w
        image_np = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='symmetric')
        mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode='symmetric')

        image_t = torch.from_numpy(image_np.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        mask_t = torch.from_numpy((mask_np[None, None] > 0).astype(np.float32))
        with torch.inference_mode():
            out = self.model(image_t.to(self.device), mask_t.to(self.device))
        out = out[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(out * 255, 0, 255).astype(np.uint8)[:h, :w]


class ImageProcessingService:
    """Service for handling image processing operations"""
    
    @staticmethod
    def load_model(weights_path: str, device: torch.device) -> UNet:
        """Load U-Net model from weights file (exact match to main.ipynb architecture)"""
        try:
            print(f"🔍 Loading model from: {weights_path}")
            print(f"🔍 Device: {device}")
            
            # Create model with exact same architecture as main.ipynb
            model = UNet()
            
            # Load weights (map to device)
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            
            # Move to device and set to eval mode
            model.to(device)
            model.eval()
            
            print(f"✅ Model loaded successfully from {weights_path}")
            print(f"✅ Model is on device: {next(model.parameters()).device}")
            
            # Test model with dummy input to verify it works
            with torch.no_grad():
                test_input = torch.randn(1, 1, 1024, 1024).to(device)
                test_output = model(test_input)
                print(f"✅ Model test successful - Output shape: {test_output.shape}")
                print(f"✅ Output range: {test_output.min():.6f} to {test_output.max():.6f}")
            
            return model
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    @staticmethod
    def predict_dust_mask(model: UNet, image_path_or_image, threshold: float = 0.5, 
                         window_size: int = 1024, stride: int = 512, 
                         device: torch.device = None, progress_callback: Optional[callable] = None) -> np.ndarray:
        """
        Fast path: scale the original image to 1024x1024 (squeezed), run once,
        then scale the probability map back to the original resolution.

        window_size/stride are ignored in this mode (kept for API compatibility).
        """
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Load image in grayscale
        if isinstance(image_path_or_image, str):
            image = Image.open(image_path_or_image).convert('L')
        else:
            image = image_path_or_image.convert('L') if image_path_or_image.mode != 'L' else image_path_or_image

        orig_w, orig_h = image.size
        print(f"🔍 Input image size: {orig_w}x{orig_h}")

        # Force-resize to 1024x1024 (squeezed if necessary)
        target = 1024
        image_1024 = image.resize((target, target), Image.Resampling.BILINEAR)

        img_np = np.array(image_1024, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

        if progress_callback:
            progress_callback(0.1)

        with torch.no_grad():
            pred = model(tensor)
            pred_np = pred.squeeze().detach().cpu().numpy().astype(np.float32)

        if progress_callback:
            progress_callback(0.7)

        # Resize prediction back to original dimensions (stretch back)
        up_pred = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        if progress_callback:
            progress_callback(1.0)

        print(f"🔍 Final prediction shape: {up_pred.shape}")
        print(f"🔍 Prediction range: {up_pred.min():.6f} to {up_pred.max():.6f}")
        return up_pred
    
    @staticmethod
    def create_binary_mask(prediction: np.ndarray, threshold: float, 
                          original_size: Tuple[int, int]) -> Image.Image:
        """Create binary mask from prediction (matches Swift ImageProcessingService)"""
        print(f"🎯 Creating binary mask with threshold {threshold:.3f}")
        print(f"🔍 Prediction shape: {prediction.shape}, Original size: {original_size}")
        
        # Handle different prediction shapes (from PyTorch model)
        if len(prediction.shape) == 4:
            # Shape is typically (1, 1, H, W) from PyTorch
            prediction = prediction.squeeze()
        elif len(prediction.shape) == 3:
            # Shape might be (1, H, W)
            prediction = prediction.squeeze()
        elif len(prediction.shape) == 2:
            # Already (H, W)
            pass
        else:
            print(f"❌ Unexpected prediction shape: {prediction.shape}")
            return None
        
        print(f"🔍 Final prediction shape: {prediction.shape}")
        
        # Apply threshold (matches Swift app logic exactly)
        binary_mask = (prediction > threshold).astype(np.uint8) * 255
        
        # DEBUG: Print non-black pixel count
        non_black_pixels = (binary_mask > 0).sum()
        total_pixels = binary_mask.size
        percentage = (non_black_pixels / total_pixels) * 100
        print(f"🎯 DUST DETECTION: {non_black_pixels:,} non-black pixels out of {total_pixels:,} ({percentage:.2f}%)")
        
        # Convert to PIL Image
        mask_image = Image.fromarray(binary_mask, mode='L')
        
        print(f"🔍 Created mask size: {mask_image.size}")
        
        # Resize to original image size if needed
        if mask_image.size != original_size:
            print(f"🔍 Resizing mask from {mask_image.size} to {original_size}")
            mask_image = mask_image.resize(original_size, Image.Resampling.NEAREST)
            
            # DEBUG: Re-check after resize
            final_mask_array = np.array(mask_image)
            final_non_black = (final_mask_array > 0).sum()
            final_percentage = (final_non_black / final_mask_array.size) * 100
            print(f"🎯 FINAL DUST MASK: {final_non_black:,} non-black pixels ({final_percentage:.2f}%)")
        
        print(f"✅ Binary mask created: {mask_image.size}")
        return mask_image
    
    @staticmethod
    def dilate_mask(mask: Image.Image, kernel_size: int = 5) -> Image.Image:
        """Dilate mask for better inpainting coverage (fixed radius by default)."""
        # Convert to numpy
        mask_np = np.array(mask.convert('L'))
        
        # Use a fixed elliptical kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply dilation
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        
        print(f"🔍 Dilated mask with {kernel_size}x{kernel_size} kernel")
        
        return Image.fromarray(dilated, mode='L')
    
    @staticmethod
    def blend_images(original: Image.Image, inpainted: Image.Image, 
                    mask: Image.Image) -> Image.Image:
        """Blend original and inpainted images using mask"""
        # Ensure all images are same size and mode
        original = original.convert('RGB')
        inpainted = inpainted.convert('RGB')
        mask = mask.convert('L')
        
        # Resize inpainted and mask to match original if needed
        if inpainted.size != original.size:
            inpainted = inpainted.resize(original.size, Image.Resampling.LANCZOS)
        if mask.size != original.size:
            mask = mask.resize(original.size, Image.NEAREST)
        
        # Convert to numpy arrays
        orig_np = np.array(original, dtype=np.float32)
        inpaint_np = np.array(inpainted, dtype=np.float32)
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        
        # Expand mask to 3 channels
        mask_3d = np.stack([mask_np] * 3, axis=2)
        
        # Blend: use inpainted where mask is white, original elsewhere
        blended = orig_np * (1 - mask_3d) + inpaint_np * mask_3d
        
        # Convert back to PIL
        result = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
        
        print(f"✅ Images blended successfully")
        return result


class BrushTools:
    """Tools for brush and eraser operations on masks"""
    
    @staticmethod
    def apply_circular_brush(mask: Image.Image, center: Tuple[float, float], 
                           radius: int, is_erasing: bool = True) -> Image.Image:
        """Apply circular brush stroke to mask"""
        # Convert to numpy for processing
        mask_np = np.array(mask.convert('L'))
        h, w = mask_np.shape
        
        # Convert center to integer coordinates
        cx, cy = int(center[0]), int(center[1])
        
        # Bounds check
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            return mask
        
        # Create circular brush
        y, x = np.ogrid[:h, :w]
        mask_circle = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
        
        # Apply brush
        if is_erasing:
            mask_np[mask_circle] = 0  # Erase (set to black)
        else:
            mask_np[mask_circle] = 255  # Add dust (set to white)
        
        return Image.fromarray(mask_np, mode='L')
    
    @staticmethod
    def interpolated_stroke(mask: Image.Image, start_point: Tuple[float, float],
                          end_point: Tuple[float, float], radius: int, 
                          is_erasing: bool = True) -> Image.Image:
        """Apply interpolated stroke between two points"""
        # Calculate distance and steps
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        distance = np.sqrt(dx * dx + dy * dy)
        
        if distance < 1.0:
            # Single point application
            return BrushTools.apply_circular_brush(mask, end_point, radius, is_erasing)
        
        # Calculate number of steps based on brush size
        spacing = max(1.0, radius * 0.25)
        steps = max(1, int(distance / spacing))
        
        # Apply brush at interpolated points
        current_mask = mask
        for i in range(steps + 1):
            t = i / steps if steps > 0 else 0
            interp_point = (
                start_point[0] + t * dx,
                start_point[1] + t * dy
            )
            current_mask = BrushTools.apply_circular_brush(
                current_mask, interp_point, radius, is_erasing
            )
        
        return current_mask


class ProcessingTask:
    """Async processing task wrapper"""
    
    def __init__(self, target_func, args=(), kwargs=None, callback=None, error_callback=None):
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs or {}
        self.callback = callback
        self.error_callback = error_callback
        self.thread = None
        self.result = None
        self.error = None
        self.completed = False
    
    def start(self):
        """Start the processing task"""
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        """Run the task in background thread"""
        try:
            print(f"🧵 ProcessingTask thread started")
            start_time = time.time()
            self.result = self.target_func(*self.args, **self.kwargs)
            end_time = time.time()
            
            print(f"🧵 ProcessingTask completed, result type: {type(self.result)}")
            self.completed = True
            
            if self.callback:
                print(f"🧵 Calling completion callback...")
                self.callback(self.result, end_time - start_time)
            else:
                print(f"🧵 No callback provided")
        except Exception as e:
            print(f"🧵 ProcessingTask error: {e}")
            self.error = e
            self.completed = True
            
            if self.error_callback:
                print(f"🧵 Calling error callback...")
                self.error_callback(e)
            else:
                print(f"🧵 No error callback provided")
    
    def is_running(self) -> bool:
        """Check if task is still running"""
        return self.thread is not None and self.thread.is_alive()
    
    def join(self, timeout=None):
        """Wait for task completion"""
        if self.thread:
            self.thread.join(timeout)
