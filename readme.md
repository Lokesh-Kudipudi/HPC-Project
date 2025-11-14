# SAR Image Segmentation using a Graph-Based Algorithm

This project provides a C++ implementation of a graph-based segmentation algorithm tailored for Synthetic Aperture Radar (SAR) images. The approach first over-segments the image into superpixels and then iteratively merges them based on a dissimilarity metric, effectively grouping similar regions together.

## Algorithm Overview

The segmentation process consists of the following sequential steps:

1.  **Image Preprocessing**:

    - The input image is converted to a single-channel grayscale format.
    - A **Bilateral Filter** is applied to reduce speckle noise while preserving important edge information.
    - Pixel intensities are normalized to a floating-point range of [0, 1] for consistent calculations.

2.  **Superpixel Extraction**:

    - The image is divided into initial superpixels using an improved **region-growing** algorithm.
    - Seeds for region growing are placed on a grid to ensure even coverage across the image.
    - Regions that do not meet a minimum size threshold are merged into the nearest neighboring superpixel to avoid oversegmentation artifacts.

3.  **Graph Construction**:

    - A Region Adjacency Graph (RAG) is built where each **node** represents a superpixel.
    - An **edge** is created to connect adjacent superpixels.
    - The **weight** of each edge is calculated based on the difference in mean intensity and the sum of variances between the two superpixels. This weighting scheme penalizes merges between heterogeneous regions.

4.  **Graph-Based Segmentation**:

    - The edges of the graph are sorted by weight in ascending order.
    - A **Union-Find** data structure is used to efficiently manage the merging of superpixels into larger segments.
    - The algorithm iterates through the sorted edges, merging two segments if the edge weight is small compared to the internal difference within the segments. This merging decision is controlled by a dynamic threshold parameter `k`.

5.  **Result Generation**:
    - The final segments are colored based on their average intensity to produce the output segmentation map.
    - A morphological closing operation is applied as a post-processing step to clean up small holes and smooth segment boundaries.

## Dependencies

- **OpenCV (4.x or later)**: Required for image I/O, filtering, and core data structures.

## Testing Environment for metrics table

Macbook M4 Air - Sequential and OpenMP Version \
Tesla T4 - CUDA

## Dataset Source

[Synspective](https://synspective.com/gallery/)

## Build Instructions

You can compile the program using a C++ compiler (like g++) and `pkg-config` to link against the necessary OpenCV libraries.

### Sequencial Version ( Require's C++17 )

```bash
g++ main_seq.cpp -std=c++17 -o main_seq `pkg-config --cflags --libs opencv4`
```

### OpenMP Version on Windows

```bash
g++ -fopenmp main_omp.cpp -std=c++17 -o main_omp ⁠`pkg-config --cflags --libs opencv4`
```

### OpenMP Version on MacOS using Clang

```bash
g++ -std=c++17 -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include `pkg-config --cflags --libs opencv4` main_omp.cpp -o main_omp -L/opt/homebrew/opt/libomp/lib -lomp
```

### CUDA

```bash
!nvcc -o main_cuda main_cuda.cu `pkg-config --cflags --libs opencv4` -std=c++17 -diag-suppress 611
```

## Run Instructions

```bash
./main_seq inputs/sample1.jpg
```

```bash
./main_omp inputs/sample1.jpg
```

```bash
./main_cuda inputs/sample1.jpg
```

# To create a profiling Report on seq

1. Compile the code with profiling enabled

```bash
g++ -pg main_seq.cpp -std=c++17 -o main_seq `pkg-config --cflags --libs opencv4`
```

2. Run the Program

```bash
./main_seq inputs/sample1.jpg
```

3. Run the gprof command

```bash
gprof main_seq gmon.out > report_main_seq.txt
```

## Evalution Metrics

### Calinski–Harabasz Score (CH Score)

What it is: \
Measures how well clusters are separated and compact.

Better range: \
Higher is better. \
(High CH = tight clusters + good separation.)

### Intra-class Variance

What it is: \
Measures how spread out the data points are within each class/cluster.

Better range: \
Lower is better.
(Low variance = compact, consistent clusters.)

### Inter-class Separation (Distance Between Class Means)

What it is: \
Distance between the centers (means) of different classes/clusters.

Better range: \
Higher is better. \
(Larger distance = better-separated classes.)

### Entropy of the Segmented Image

What it is: \
Measures randomness/complexity in the segmented image.

Better range: \
Depends on context: \
Low entropy → simpler, smoother segmentation. \
High entropy → more detailed/complex segmentation.

### Evalution Sheet

[Google Sheet Link](https://docs.google.com/spreadsheets/d/1SwYvWu2-6ZTNSZguZ7LxR_cQ_cC1RAzSuLoQ0ACM_14/edit?usp=sharing)
