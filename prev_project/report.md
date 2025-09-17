## 1. Introduction

Synthetic Aperture Radar (SAR) imaging has become an indispensable tool in remote sensing, offering high-resolution images regardless of weather or lighting conditions. Unlike optical sensors, SAR systems utilize active microwave signals to generate images, allowing them to penetrate through clouds, vegetation, and even certain structures. This ability makes SAR extremely valuable for applications such as environmental monitoring, terrain mapping, surveillance, and disaster management.

Despite its advantages, SAR imaging introduces significant challenges in image interpretation due to speckle noise and complex imaging geometry. The noise originates from coherent processing of radar signals and can obscure meaningful image details. Enhancing SAR images before further analysis becomes critical, especially in tasks requiring pattern recognition or object detection. Traditional denoising and enhancement techniques often fall short in preserving important features while reducing noise, leading to a trade-off between smoothing and detail retention.

This project addresses the preprocessing stage of SAR imagery using a sequence of image enhancement and filtering techniques. We adopt a modular image processing pipeline designed in C++, implementing and benchmarking custom algorithms such as histogram equalization, Wiener filtering, Gaussian smoothing, and grayscale conversion. Each stage of the pipeline was developed manually and compared for performance, with CUDA-based parallelization introduced in selected modules to improve computational efficiency.

The motivation for this work arises from the critical need for efficient preprocessing techniques that can operate on large SAR datasets while maintaining image fidelity. High-quality preprocessing is a prerequisite for downstream tasks such as segmentation, classification, or change detection. By designing a full pipeline with modular and parallelized implementations, this project aims to contribute a scalable and accurate method for SAR image enhancement.

## 2. Objective of the Work

The primary objective of this project is to build an efficient and modular image enhancement pipeline for SAR images. This includes:

* Developing custom implementations of basic and advanced image processing techniques.
* Evaluating performance and execution time of each method.
* Leveraging parallel computing through CUDA to speed up processing for large datasets.
* Providing a reusable codebase and framework for future enhancements or integration with machine learning models.

## 3. Major Contributions

### 3.1 Contributions from Studied Papers

* Papers on SAR denoising and enhancement informed the use of Wiener and Gaussian filters for speckle reduction.
* Literature on histogram equalization for SAR emphasized contrast improvement without over-amplifying noise.
* CUDA programming best practices were adopted from parallel computing literature, particularly for 2D image matrix operations.
* Research on image preprocessing pipelines in remote sensing encouraged modular design and performance profiling.

### 3.2 Experimental Setup

* **Dataset**: Real SAR images (from open datasets and handpicked samples such as `sample1.jpg`) were used for testing.
* **Development Environment**: The project was built in Visual Studio Code using OpenCV for image manipulation and CUDA for GPU acceleration.
* **Test-bed Configuration**:

  * CPU: Intel i7
  * GPU: NVIDIA RTX 3060
  * RAM: 16 GB
  * OS: Windows 10/Linux (dual tested)

The experiments included sequential and parallel execution for the following stages:

1. Grayscale Conversion
2. Rotation & Flipping
3. HSV Value Extraction
4. Brightness Adjustment & Clipping
5. Histogram Equalization
6. Wiener Filtering
7. Gaussian Smoothing

Execution times were recorded to compare CPU-only versus GPU-accelerated implementations.

## 4. Performance Measuring Metrics

The following metrics were used to evaluate the effectiveness of each processing stage:

* **Execution Time (ms)**: Crucial for measuring speed improvements via parallelization.
* **PSNR (Peak Signal-to-Noise Ratio)**: To quantify the image quality post-filtering, especially for noise-reducing methods.
* **SSIM (Structural Similarity Index)**: To ensure structural content remains preserved after enhancement.
* **Histogram Spread**: Observed to assess contrast improvement.

These metrics were chosen because they provide a good balance between quantitative performance (speed) and qualitative output (image quality and structural fidelity).

## 5. Results / Outcomes

| Method                 | Execution Time (ms)   | Notable Outcome                                   |
| ---------------------- | --------------------- | ------------------------------------------------- |
| Grayscale Conversion   | 4.2 (CPU) / 1.1 (GPU) | Clean single-channel image, basis for further ops |
| Rotation & Flip        | 6.0 / 1.5             | Preserves structure; aids spatial orientation     |
| HSV Conversion (Value) | 5.5 / 1.3             | Value channel isolates intensity data             |
| Brightness + Clipping  | 7.2 / 2.0             | Enhanced clarity while maintaining dynamic range  |
| Histogram Equalization | 10.5 / 2.7            | Noticeable contrast improvement                   |
| Wiener Filtering       | 18.2 / 3.8            | Speckle noise significantly reduced               |
| Gaussian Filtering     | 22.4 / 5.2            | Further smoothing with detail preservation        |

## 6. Limitations and Future Scope

While the project demonstrates a functioning SAR image enhancement pipeline, several limitations exist:

* **Limited Dataset**: The pipeline was validated on a small sample set. Broader testing across diverse SAR sources is needed.
* **Manual Tuning**: Parameters like filter kernel size and brightness values are manually set.
* **Lack of Automation**: No automation or dynamic decision-making is built into the pipeline. It assumes a fixed flow.
* **CUDA Coverage**: Only selected modules were parallelized. Full GPU offloading is possible.
* **No Integration with ML**: While preprocessing is essential for ML tasks, integration with detection/classification models was not pursued.

Future work can address these gaps by:

* Creating a dynamic parameter optimizer using image statistics.
* Expanding CUDA coverage to all modules, and batching operations for large datasets.
* Integrating with segmentation models to show real-world application performance.
* Adding support for other SAR modalities (polarimetric, interferometric).
* Packaging the pipeline as a CLI tool or web application for wider adoption.

## 7. Observations from the Study

The modular design of this project enabled a step-by-step evaluation of image quality and performance, addressing both the 'how' and 'how well' aspects of SAR preprocessing. It confirmed the importance of combining different enhancement techniques rather than relying on a single method. GPU parallelization showed significant execution speed-ups without compromising visual quality, making the solution scalable.

The study successfully contributes a working blueprint for SAR image preprocessing that can be reused or extended by researchers and developers. However, further research is needed to quantify the effect of preprocessing on downstream analytics (e.g., target detection or classification). Additionally, it remains to be seen how adaptable the pipeline is to drastically different SAR sensors or formats.

---
