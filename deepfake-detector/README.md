# Deepfake Detection Pipeline

A complete, end-to-end pipeline for detecting deepfake videos using deep learning. This project implements best practices for data splitting (to prevent leakage), training, and evaluation of deepfake detection models.

## Features

- **Data Leakage Prevention**: Splits dataset by source ID to ensure no overlap between train/test sets
- **State-of-the-art Models**: Uses Xception or EfficientNet-B0 architectures optimized for deepfake detection
- **Video-level Evaluation**: Aggregates frame predictions for robust video-level metrics
- **Compression Robustness Testing**: Evaluates model performance on compressed videos
- **Comprehensive Visualization**: Generates detailed plots and metrics reports
- **Production-ready Code**: Full error handling, logging, and reproducibility

## Project Structure

```
deepfake-detector/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   ├── raw_videos/            # Original videos split by train/val/test
│   │   ├── train/
│   │   │   ├── REAL/
│   │   │   └── FAKE/
│   │   ├── val/
│   │   │   ├── REAL/
│   │   │   └── FAKE/
│   │   └── test/
│   │       ├── REAL/
│   │       └── FAKE/
│   └── frames/                # Extracted and cropped frames
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── 00_download.py         # Dataset download helper
│   ├── 01_split_by_source.py  # Split dataset preventing leakage
│   ├── 02_extract_frames.py   # Extract frames from videos
│   ├── 03_crop_faces.py       # Detect and crop faces
│   ├── 04_train.py            # Train the model
│   ├── 05_eval.py             # Evaluate the model
│   ├── 06_compress_test.py    # Test compression robustness
│   ├── 07_report_plots.py     # Generate visualization plots
│   └── utils/
│       ├── dataset.py         # Dataset and dataloaders
│       ├── model.py           # Model architecture
│       └── metrics.py         # Evaluation metrics
└── outputs/
    ├── models/                # Saved model checkpoints
    ├── metrics/               # Evaluation reports
    └── plots/                 # Visualization plots
```

## Installation

### 1. Clone or create the project directory

```bash
cd deepfake-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install ffmpeg (required for video processing)

**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH

**Linux:**
```bash
sudo apt-get install ffmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

## Dataset Setup

This pipeline is designed for the **FaceForensics++** dataset, but can be adapted for other datasets.

### Option 1: Manual Download

1. Visit the [FaceForensics++ repository](https://github.com/ondyari/FaceForensics)
2. Request access and download the dataset
3. Place videos in a temporary directory with the structure:
   ```
   faceforensics/
   ├── original_sequences/
   │   └── youtube/
   │       └── c23/videos/
   └── manipulated_sequences/
       ├── Deepfakes/
       │   └── c23/videos/
       ├── FaceSwap/
       │   └── c23/videos/
       ├── Face2Face/
       │   └── c23/videos/
       └── NeuralTextures/
           └── c23/videos/
   ```

### Option 2: Using the download script

```bash
python src/00_download.py --method manual
```

This will show you instructions for downloading the dataset.

### Alternative Datasets

You can also use:
- **Celeb-DF**: https://github.com/yuezunli/celeb-deepfakeforensics
- **DFDC**: https://www.kaggle.com/c/deepfake-detection-challenge

## Usage

Follow these steps to run the complete pipeline:

### Step 1: Split Dataset by Source ID

**Critical for preventing data leakage!** This ensures the same person doesn't appear in both training and test sets.

```bash
python src/01_split_by_source.py \
    --ffpp-root /path/to/faceforensics \
    --out data/raw_videos \
    --train-ids 100 \
    --val-ids 50 \
    --test-ids 60 \
    --seed 42
```

**Arguments:**
- `--ffpp-root`: Path to FaceForensics++ root directory
- `--out`: Output directory for split videos
- `--train-ids`: Number of unique source IDs for training
- `--val-ids`: Number of unique source IDs for validation
- `--test-ids`: Number of unique source IDs for testing
- `--seed`: Random seed for reproducibility

This will create `data/raw_videos/split_manifest.csv` documenting the split.

### Step 2: Extract Frames

Extract frames from videos at 3 FPS (configurable):

```bash
python src/02_extract_frames.py \
    --input data/raw_videos \
    --output data/frames \
    --fps 3 \
    --max-frames 16
```

**Arguments:**
- `--input`: Input directory with videos
- `--output`: Output directory for frames
- `--fps`: Frames per second to extract (default: 3)
- `--max-frames`: Maximum frames per video (default: 16)

### Step 3: Crop Faces

Detect and crop faces using MTCNN:

```bash
python src/03_crop_faces.py \
    --input data/frames \
    --output data/frames \
    --size 224 \
    --device cuda
```

**Arguments:**
- `--input`: Input directory with frames
- `--output`: Output directory (use same as input to overwrite)
- `--size`: Target face size (default: 224)
- `--device`: Device to use (cpu/cuda/mps)

**Note:** Frames without detectable faces will be removed.

### Step 4: Train Model

Train the deepfake detection model:

```bash
python src/04_train.py \
    --config config.yaml \
    --data data/frames \
    --output outputs \
    --epochs 10 \
    --batch-size 32 \
    --device cuda
```

**Arguments:**
- `--config`: Path to config file
- `--data`: Path to frames directory
- `--output`: Output directory for models and logs
- `--epochs`: Number of training epochs (optional, overrides config)
- `--batch-size`: Batch size (optional, overrides config)
- `--device`: Device to use (optional, auto-detects if not specified)

**Output:**
- `outputs/models/best.pt`: Best model checkpoint (highest val F1)
- `outputs/models/final.pt`: Final model checkpoint
- `outputs/metrics/training_history.json`: Training logs

### Step 5: Evaluate Model

Evaluate the trained model on test set:

```bash
python src/05_eval.py \
    --checkpoint outputs/models/best.pt \
    --data data/frames \
    --output outputs/metrics \
    --split test
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--data`: Path to frames directory
- `--output`: Output directory for metrics
- `--split`: Dataset split to evaluate (train/val/test)

**Output:**
- `outputs/metrics/test_report.csv`: Detailed metrics and per-video predictions

### Step 6: Test Compression Robustness (Optional)

Test how well the model handles compressed videos:

```bash
python src/06_compress_test.py \
    --input data/raw_videos/test \
    --checkpoint outputs/models/best.pt \
    --output outputs/compression_test \
    --crf 18 28 35
```

**Arguments:**
- `--input`: Directory with test videos
- `--checkpoint`: Model checkpoint
- `--output`: Output directory
- `--crf`: CRF values to test (lower = better quality)

**Output:**
- `outputs/compression_test/compression_results.json`: Metrics for each CRF level

### Step 7: Generate Plots

Create visualization plots:

```bash
python src/07_report_plots.py \
    --metrics-dir outputs/metrics \
    --output-dir outputs/plots
```

**Output:**
- `outputs/plots/prf_bar.png`: Precision/Recall/F1 comparison
- `outputs/plots/confusion_matrix.png`: Test set confusion matrix
- `outputs/plots/training_curves.png`: Loss and F1 over epochs
- `outputs/plots/f1_vs_crf.png`: Performance vs compression (if step 6 was run)

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
data:
  raw_videos_dir: "data/raw_videos"
  frames_dir: "data/frames"

preprocessing:
  fps: 3                      # Frames per second to extract
  max_frames_per_video: 16    # Max frames per video
  face_size: 224              # Face crop size

model:
  name: "xception"            # Model architecture
  pretrained: true            # Use ImageNet pretrained weights
  num_classes: 1              # Binary classification

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.01
  early_stopping_patience: 3

evaluation:
  aggregation: "mean"         # Frame aggregation method (mean/median)
  threshold: 0.5              # Classification threshold
```

## Expected Results

With proper training on FaceForensics++, you should achieve:

- **Validation F1**: 0.85-0.95
- **Test F1**: 0.80-0.93
- **Test Accuracy**: 0.82-0.94

Performance may vary based on:
- Number of training samples
- Model architecture
- Data augmentation
- Manipulation methods in test set

## Troubleshooting

### Issue: "No videos found"
**Solution:** Check that video files are in the correct directory structure with REAL/FAKE subdirectories.

### Issue: "No faces detected"
**Solution:**
- Ensure videos contain clear, frontal faces
- Try adjusting MTCNN thresholds in `03_crop_faces.py`
- Some frames may legitimately have no faces - they will be skipped

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size in config.yaml or command line
- Use gradient accumulation (modify training script)
- Use a smaller model (e.g., efficientnet_b0 instead of xception)

### Issue: "Training loss not decreasing"
**Solution:**
- Check that data is properly labeled (REAL/FAKE folders)
- Verify frames are correctly cropped
- Try lower learning rate
- Ensure sufficient training data

### Issue: "Model overfitting"
**Solution:**
- Add more data augmentation
- Increase weight decay
- Use dropout (modify model)
- Stop training earlier

### Issue: "ffmpeg not found"
**Solution:** Install ffmpeg and add to PATH (see Installation section)

## Advanced Usage

### Custom Dataset

To use a custom dataset:

1. Organize videos in the structure:
   ```
   my_dataset/
   ├── train/
   │   ├── REAL/
   │   └── FAKE/
   ├── val/
   │   ├── REAL/
   │   └── FAKE/
   └── test/
       ├── REAL/
       └── FAKE/
   ```

2. Skip step 1 (splitting) and start from step 2 (frame extraction)

3. Ensure video filenames allow extracting source IDs for proper aggregation

### Transfer Learning

To use a pretrained model on a new dataset:

```bash
python src/04_train.py \
    --config config.yaml \
    --data data/new_dataset \
    --checkpoint outputs/models/best.pt \
    --epochs 5 \
    --learning-rate 0.00001
```

### Ensemble Models

Train multiple models and ensemble predictions:

1. Train with different architectures (xception, efficientnet, resnet)
2. Modify evaluation script to load multiple checkpoints
3. Average predictions across models

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{deepfake-detection-pipeline,
  title={Deepfake Detection Pipeline},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/deepfake-detector}
}
```

## References

- **FaceForensics++**: Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images", ICCV 2019
- **Xception**: Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions", CVPR 2017
- **MTCNN**: Zhang et al., "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks", 2016

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- FaceForensics++ dataset creators
- PyTorch and timm library maintainers
- facenet-pytorch for MTCNN implementation

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
