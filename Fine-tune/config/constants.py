"""
Configuration constants for training and models.

All magic numbers from the codebase are centralized here for easy tuning.
"""

# ============================================================================
# Gradient & Training Stability
# ============================================================================
GRADIENT_CLIP_MAX_NORM = 0.5  # Prevents gradient explosion during training
GRADIENT_CLIPPING_MIN_VALUE = -10.0  # Clamp heatmap predictions to prevent numerical issues
GRADIENT_CLIPPING_MAX_VALUE = 10.0

# ============================================================================
# Model Architecture & Initialization
# ============================================================================
# Heatmap bias initialization for drone dataset
# Logit value that produces P(positive) ≈ 0.1 when passed through sigmoid
# This biases the model towards predicting background initially
HEATMAP_BIAS_INIT = -4.6

# CenterNet architecture default channels
DEFAULT_HEAD_CONV_CHANNELS = 256
DEFAULT_DETECTION_DOWNSAMPLE_RATIO = 4  # Stride-4 output from detection head

# ============================================================================
# Detection Loss Configuration
# ============================================================================
DEFAULT_HEATMAP_SIGMA = 0.8  # Gaussian kernel sigma for target generation
DEFAULT_HEATMAP_POS_WEIGHT = 7.0  # Positive class weight for imbalanced heatmap
DEFAULT_DET_NEG_TOPK_RATIO = 0.1  # Fraction of hardest negatives to include in loss

# Background Suppression: Penalizes high activations in background regions
# Helps reduce false positives by explicitly teaching model that background = low scores
DEFAULT_BG_SUPPRESSION_WEIGHT = 0.0  # Set to 0.05-0.2 to enable (0.0 = disabled)

# Label Smoothing: Replaces hard 0/1 targets with soft labels (e.g., 0.05/0.95)
# Prevents overconfident predictions and improves score calibration
DEFAULT_LABEL_SMOOTHING = 0.0  # Set to 0.05-0.1 to enable (0.0 = disabled)

# ============================================================================
# Focal Loss Configuration
# ============================================================================
# Focal loss parameters: α(1-p_t)^γ log(p_t)
# Higher α focuses more on positives, γ > 0 downweights easy examples
DEFAULT_FOCAL_ALPHA = 0.75  # Class balance: higher for sparse positives
DEFAULT_FOCAL_GAMMA = 2.0   # Hard example focus: higher = more focus on hard examples

# ============================================================================
# Adaptive Threshold (Evaluation)
# ============================================================================
ADAPTIVE_THRESHOLD_PERCENTILE = 98  # Percentile for adaptive thresholding
ADAPTIVE_THRESHOLD_MIN_SCORE = 0.5  # Lower bound for adaptive threshold
ADAPTIVE_THRESHOLD_MAX_SCORE = 0.5  # Upper bound for adaptive threshold

# ============================================================================
# Evaluation Metrics
# ============================================================================
DEFAULT_AP_DISTANCE_THRESHOLD = 8.0  # Distance threshold in pixels for AP matching
DEFAULT_EVAL_NMS_RADIUS = 2.0  # NMS suppression radius in pixels
DEFAULT_EVAL_SOFT_NMS_SIGMA = 0.5  # Soft-NMS decay parameter
DEFAULT_EVAL_NMS_KERNEL = 3  # Max-pooling NMS kernel size

# ============================================================================
# Thermal Image Preprocessing
# ============================================================================
CLAHE_DEFAULT_CLIP_LIMIT = 2.0  # CLAHE clip limit for contrast enhancement
CLAHE_DEFAULT_TILE_SIZE = 8  # CLAHE tile grid size

# ============================================================================
# Data Augmentation
# ============================================================================
DEFAULT_AUG_SCALE_MIN = 0.5  # Minimum scale factor for random resize
DEFAULT_AUG_SCALE_MAX = 2.0  # Maximum scale factor for random resize

# ============================================================================
# Training Defaults
# ============================================================================
DEFAULT_LR = 1e-5  # Initial learning rate
DEFAULT_WEIGHT_DECAY = 1e-4  # L2 regularization
DEFAULT_MAX_EPOCH = 100  # Maximum training epochs
DEFAULT_VAL_EPOCH = 2  # Validation frequency (every N epochs)
DEFAULT_VAL_START = 10  # Epoch to start validation

# Early stopping patience for detection AP
DEFAULT_DET_PATIENCE = 10  # Stop after N epochs with no AP improvement

# ============================================================================
# Distributed Training
# ============================================================================
DEFAULT_NUM_WORKERS = 4  # Data loading workers
DEFAULT_BATCH_SIZE = 4  # Batch size per GPU

# ============================================================================
# Model Checkpointing
# ============================================================================
MAX_MODELS_TO_KEEP = 1  # Keep only 1 best model checkpoint

# ============================================================================
# DDP Configuration
# ============================================================================
FIND_UNUSED_PARAMETERS = False  # Set False after properly freezing components before DDP wrap

# ============================================================================
# NaN/Inf Detection
# ============================================================================
# Automatically skip batches with NaN/Inf losses to improve training stability
SKIP_CORRUPTED_BATCHES = True

# ============================================================================
# Counting Losses (Legacy)
# ============================================================================
# Weights for multi-task learning in counting mode
DEFAULT_OT_LOSS_WEIGHT = 0.1
DEFAULT_TV_LOSS_WEIGHT = 0.01
DEFAULT_RD_LOSS_WEIGHT = 0.1

# OT Solver parameters
DEFAULT_OT_ITERATIONS = 100
DEFAULT_OT_REGULARIZATION = 10.0
DEFAULT_NORMALIZE_COORDINATES = 0
