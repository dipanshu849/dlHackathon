
# app.py
import os
import io
import base64
import torch
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from fastai.vision.all import *
# --- FIXED: Import MaskBlock instead of Mask ---
from fastai.vision.all import MaskBlock  # This is the correct import

from PIL import Image  # Import the standard PIL Image module
from fastai.data.transforms import Normalize  # Ensure Normalize is imported if used in batch_tfms
from flask_cors import CORS  # Import CORS

# --- Custom Components (Copy from your training script) ---
# You need to include the definitions of any custom classes
# used in your model or DataBlock here, as they are not saved
# in the .pth file and are needed to recreate the model structure.

class AdvancedVesselPreprocessing(ItemTransform):
    """Advanced preprocessing specifically for eye vessel segmentation."""
    def __init__(self, clip_limit=4.0, tile_grid_size=(10, 10), use_gabor=False):
        store_attr()

    def encodes(self, img: PILImage):
        cv2.ocl.setUseOpenCL(False)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(10, 10))
        l_clahe = clahe.apply(l)

        if self.use_gabor:
             kernel_size = 15
             sigma = 5
             theta = np.pi/4
             lambd = 10.0
             gamma = 0.5
             psi = 0
             gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
             l_enhanced = cv2.filter2D(l_clahe, cv2.CV_8UC1, gabor_kernel)
             l_enhanced = cv2.normalize(l_enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        else:
            l_enhanced = l_clahe

        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        img_gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
        processed = cv2.merge([img_gray, img_gray, img_gray])

        return PILImage.create(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

# Define the loss function class as it was used during training
class EnhancedSegmentationLoss:
    """Combined loss for better vessel segmentation with boundary emphasis."""
    def __init__(self, dice_weight=0.7, bce_weight=0.15, focal_weight=0.15, gamma=2.0):
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.bce = BCEWithLogitsLossFlat(axis=1)

    def __call__(self, pred, targ):
        if targ.ndim < pred.ndim:
            targ = targ.unsqueeze(1)

        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * targ).sum()
        union = pred_sigmoid.sum() + targ.sum()
        dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

        bce_loss = self.bce(pred, targ)

        pt = targ * pred_sigmoid + (1 - targ) * (1 - pred_sigmoid)
        pt = pt.clamp(min=1e-6)
        focal_loss = torch.mean(-((1 - pt) ** self.gamma) * torch.log(pt))

        return (self.dice_weight * dice_loss +
                self.bce_weight * bce_loss +
                self.focal_weight * focal_loss)

# Define the metric function if it's needed for any reason (e.g., in show_results, though not strictly needed for inference)
def dice_vessel(inp, targ, smooth=1e-6):
    """Dice coefficient metric for vessel segmentation."""
    inp = (inp.sigmoid() > 0.5).float()
    targ = targ.float()
    intersection = (inp * targ).sum()
    return (2. * intersection + smooth) / (inp.sum() + targ.sum() + smooth)

# --- End Custom Components ---


# Configuration
IMG_SIZE = (256, 256)
# Path to the model weights file (.pth)
# Ensure this path is correct relative to where you run app.py
MODEL_WEIGHTS_PATH = Path('./models/bestmodel.pth')

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Global variable for the fastai learner
learn = None

def load_model():
    """Loads the fastai model weights."""
    global learn
    print("Recreating DataBlock and Learner structure...")
    try:
        # Create dummy data and dataloaders to initialize the learner structure
        # This is a workaround to instantiate the learner without a full dataset
        dummy_dir = Path('./dummy_data')
        print(f"Attempting to create dummy directory: {dummy_dir.resolve()}")
        try:
            dummy_dir.mkdir(parents=True, exist_ok=True)
            print(f"Dummy directory created or already exists: {dummy_dir.resolve()}")
        except Exception as e:
            print(f"Error creating dummy directory {dummy_dir.resolve()}: {e}")
            raise # Re-raise the exception if directory creation fails

        dummy_img_path = dummy_dir / 'dummy_image.png'
        dummy_mask_path = dummy_dir / 'dummy_mask.png'

        print(f"Dummy image path: {dummy_img_path.resolve()}")
        print(f"Dummy mask path: {dummy_mask_path.resolve()}")

        # Create a dummy image and mask file if they don't exist
        if not dummy_img_path.exists():
            print(f"Creating dummy image: {dummy_img_path.resolve()}")
            Image.new('RGB', IMG_SIZE).save(dummy_img_path) # Use standard PIL Image
        else:
             print(f"Dummy image already exists: {dummy_img_path.resolve()}")

        if not dummy_mask_path.exists():
             print(f"Creating dummy mask: {dummy_mask_path.resolve()}")
             Image.new('L', IMG_SIZE, color=0).save(dummy_mask_path) # Create a dummy black mask
        else:
             print(f"Dummy mask already exists: {dummy_mask_path.resolve()}")


        # Define the DataBlock structure that was used during training
        # This is necessary to correctly build the model architecture
        # We don't need actual data files here, just the structure definition
        data_block = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=['background', 'vessel'])),
            get_items=lambda x: [dummy_img_path.resolve()], # Use the resolved absolute path
            splitter=RandomSplitter(valid_pct=0.0), # No validation set needed for inference structure
            get_y=lambda x: dummy_mask_path.resolve(), # Also provide absolute path for mask
            item_tfms=[
                AdvancedVesselPreprocessing(clip_limit=4.0, tile_grid_size=(10, 10)),
                Resize(IMG_SIZE, method='squish')
            ],
            batch_tfms=[
                # Include Normalize if it was used in the training batch_tfms
                Normalize.from_stats(*imagenet_stats),
            ]
        )


        # Create DataLoaders from the dummy data
        # Pass the dummy directory as the path argument
        dls = data_block.dataloaders(dummy_dir, bs=1, shuffle=False, drop_last=False)

        # Define the learner architecture exactly as it was during training
        learn = unet_learner(
            dls,
            resnet34, # Ensure this matches the trained architecture
            n_out=1,
            metrics=[dice_vessel], # Include metrics if defined, though not strictly needed for inference
            loss_func=EnhancedSegmentationLoss(), # Include loss func if defined
            wd=1e-2, # Include other learner arguments as used in training
            path=Path('.'),
            model_dir=Path('./models').relative_to(Path('.')),
            self_attention=True # Ensure this matches the trained setup
        )

        # Load the state dictionary (weights) from the .pth file
        if not MODEL_WEIGHTS_PATH.exists():
             raise FileNotFoundError(f"Model weights file not found at {MODEL_WEIGHTS_PATH.resolve()}") # Resolve path for clarity

        # --- Load state_dict directly into the model ---
        # Use map_location='cpu' if you are not using a GPU in the Flask environment
        learn.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
        learn.model.eval() # Set the model to evaluation mode
        # --- End Load ---


        print("Model loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error loading model weights: {e}")
        print("Please ensure the model weights file exists at the specified path.")
        learn = None # Ensure learn is None if loading fails
    except Exception as e:
        print(f"Error recreating learner or loading weights: {e}")
        print("Check if the DataBlock, Learner arguments, custom components, and model weights path are correct.")
        learn = None # Ensure learn is None if loading fails


# Load the model when the Flask app starts
with app.app_context():
    load_model()

@app.route('/')
def index():
    # Serve the index.html file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if learn is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read the image file
        img_bytes = file.read()
        img = PILImage.create(io.BytesIO(img_bytes)) # Use fastai's PILImage.create for initial loading

        # Apply the same preprocessing as training/inference
        # fastai's predict handles item_tfms and batch_tfms
        # learn.predict returns a tuple: (decoded_prediction, prediction_index, probabilities_tensor)
        # For segmentation, decoded_prediction is often a Mask object, probabilities_tensor is the raw output before sigmoid/softmax
        # We need the probabilities_tensor (the third element) to apply our threshold
        _, _, prob_tensor = learn.predict(img) # Get the probability tensor

        # Apply sigmoid to the output logits and then threshold
        # Assuming n_out=1 and BCEWithLogitsLossFlat was used, output is logits
        # Squeeze to remove batch dimension (1) and channel dimension (1) if present
        mask_tensor = torch.sigmoid(prob_tensor).squeeze().cpu() # Apply sigmoid, remove batch/channel dims, move to CPU

        # Apply threshold (using 0.2 as in your training script)
        mask_np = (mask_tensor.numpy() > 0.2).astype(np.uint8) * 255 # Convert to uint8 (0 or 255)


        # Encode the mask numpy array as a base64 PNG image
        # Ensure the mask_np is in the correct format (H, W) for imencode
        # This check is likely redundant after the squeeze() above, but kept for safety
        if mask_np.ndim == 3 and mask_np.shape[2] == 1:
            mask_np = mask_np.squeeze(axis=2) # Remove channel dimension if present
        elif mask_np.ndim == 3 and mask_np.shape[0] == 1: # Handle potential (1, H, W) shape
             mask_np = mask_np.squeeze(axis=0)


        is_success, buffer = cv2.imencode(".png", mask_np)
        if not is_success:
            return jsonify({'error': 'Could not encode mask image'}), 500

        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the base64 encoded mask image
        return jsonify({'mask_image_base64': mask_base64})

    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {e}'}), 500

if __name__ == '__main__':
    print(f"Starting Flask server on port 5001...")
    # Use debug=True for development, set to False for production
    app.run(debug=True, host='0.0.0.0', port=5001)
