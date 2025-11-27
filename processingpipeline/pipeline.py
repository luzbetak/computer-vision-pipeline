from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from processingpipeline.gaussian import convolve, gaussian_kernel
from processingpipeline.io_utils import read_grayscale_image, scale_down, write_grayscale_image
from processingpipeline.median_filter import apply_median_filter
from processingpipeline.otsu import binarize, otsu_threshold
from processingpipeline.percentile import apply_stretch, histogram, set_cutoff, stretch_map
from processingpipeline.reflection import apply_reflection_padding
from processingpipeline.sobel import sobel_filters
from processingpipeline import statistics


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    padding_x: int = 7
    padding_y: int = 7
    kernel_size: int = 15
    sigma: float = 1.6
    percentile: float = 10.0


class ProcessingPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self) -> None:
        print("Processing Pipeline (Python)")
        input_image = read_grayscale_image(self.config.input_path)

        # Percentile stretch
        hist = histogram(input_image)
        _, first, last = set_cutoff(hist, input_image.size, self.config.percentile)
        stretched_lookup = stretch_map(first, last)
        stretched_image = apply_stretch(input_image, first, last, stretched_lookup)
        write_grayscale_image("src/image/Percentile.png", stretched_image)

        # Reflection padding
        reflected = apply_reflection_padding(stretched_image, self.config.padding_x, self.config.padding_y)
        write_grayscale_image("src/image/reflection.png", reflected)

        # Median filter
        median = apply_median_filter(reflected, kernel_size=3)
        write_grayscale_image("src/image/Median.png", median)

        # Gaussian blur
        kernel = gaussian_kernel(self.config.sigma, self.config.kernel_size)
        gaussian = convolve(median, kernel)
        write_grayscale_image("src/image/Gaussian.png", gaussian)

        # Sobel edge detection
        magnitude, direction = sobel_filters(gaussian)
        magnitude_trimmed = scale_down(magnitude, self.config.padding_x, self.config.padding_y)
        direction_trimmed = scale_down(direction, self.config.padding_x, self.config.padding_y)
        write_grayscale_image("src/image/SobelMagnitute.png", magnitude_trimmed)
        write_grayscale_image("src/image/SobelDirection.png", direction_trimmed)

        # Otsu binarization
        otsu_t = otsu_threshold(magnitude_trimmed)
        binary = binarize(magnitude_trimmed, otsu_t)
        write_grayscale_image(self.config.output_path, binary)

        # Statistics
        mag_mean = statistics.mean(magnitude_trimmed)
        dir_mean = statistics.mean(direction_trimmed)
        final_mean = statistics.mean(binary)

        print(f"\n1.) src/image/SobelMagnitute.png\n\tSobel Magnitude - Image Output Mean = {mag_mean:.3f}\n")
        print(f"\n2.) src/image/SobelDirection.png\n\tSobel Direction - Image Output Mean = {dir_mean:.3f}\n")
        print(
            f"\n3.) {self.config.output_path}\n\tSobel with Otsu Threshold = {otsu_t}"
            f" - Image Output Mean = {final_mean:.3f}\n"
        )


def run_default_pipeline() -> None:
    config = PipelineConfig(
        input_path=Path("src/image/A_Lenna.png"),
        output_path=Path("src/image/A_Final.png"),
    )
    pipeline = ProcessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    run_default_pipeline()
