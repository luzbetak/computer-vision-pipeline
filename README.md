# Computer Vision Pipeline (Python)

This repository rewrites the original Java-based computer vision pipeline in Python using NumPy and Pillow.
The pipeline executes the following stages on a sample image:

1. Percentile-based histogram stretching
2. Reflection padding
3. Median filtering
4. Gaussian blur
5. Sobel magnitude and direction calculation
6. Otsu binarization
7. Basic statistics for key outputs

Sample input and generated outputs live in `src/image/`.

## Getting started

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python main.py
   ```

The default run reads `src/image/A_Lenna.png` and writes intermediate artifacts along with the final binarized image to `src/image/`.

## Customizing the pipeline

The pipeline configuration lives in `processingpipeline/pipeline.py`. Update the `PipelineConfig` values or instantiate `ProcessingPipeline` directly for different inputs, padding sizes, Gaussian parameters, or percentile cutoffs.
