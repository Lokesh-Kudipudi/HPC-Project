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

**What it is:** \
The Calinski-Harabasz Score, also known as the Variance Ratio Criterion, measures the quality of clustering by evaluating how well-defined the clusters are. It compares the between-cluster dispersion (how far apart different clusters are) to the within-cluster dispersion (how compact each cluster is internally). The score is calculated as the ratio of the sum of between-cluster dispersion to within-cluster dispersion, multiplied by a factor that accounts for the number of clusters and data points.

**Better range:** \
Higher is better. A higher CH score indicates that clusters are well-separated from each other and internally compact, which is desirable for good segmentation quality. Typically, values above 100 indicate reasonable clustering, though this can vary depending on the dataset and number of clusters.

**Interpretation:** \
High CH score = tight, compact clusters with good separation between different segments. \
Low CH score = overlapping or poorly defined clusters.

### Intra-class Variance

**What it is:** \
Intra-class variance measures the average spread or dispersion of pixel intensities within each segmented region. It quantifies how homogeneous each cluster is by calculating the variance of pixel values from the cluster mean. Lower variance indicates that pixels within the same segment have similar characteristics, which is a hallmark of good segmentation where similar pixels are grouped together.

**Better range:** \
Lower is better. Low intra-class variance indicates that pixels within each segment are very similar to each other, suggesting that the segmentation algorithm has successfully grouped homogeneous regions. This is particularly important for SAR image segmentation where we want to identify uniform areas.

**Interpretation:** \
Low variance = compact, consistent, and homogeneous clusters. \
High variance = heterogeneous clusters with diverse pixel values.

### Inter-class Separation (Distance Between Class Means)

**What it is:** \
Inter-class separation measures the average distance between the centroids (mean intensity values) of different segmented regions. It provides insight into how distinct different segments are from each other. A larger distance indicates that the algorithm has successfully identified regions with different characteristics and avoided merging dissimilar areas. This metric is computed as the average Euclidean distance between all pairs of cluster centroids.

**Better range:** \
Higher is better. Larger inter-class separation means that different segments have distinct characteristics, making them easily distinguishable from one another. This indicates that the segmentation has successfully partitioned the image into meaningful, non-overlapping regions.

**Interpretation:** \
Large distance = well-separated, distinct classes with different properties. \
Small distance = similar or overlapping classes that may not be well distinguished.

### Entropy of the Segmented Image

**What it is:** \
Entropy measures the randomness, complexity, or information content in the segmented image. It is calculated based on the histogram of pixel intensity values in the output segmentation. Higher entropy indicates more variation and complexity in the segmentation, while lower entropy suggests a simpler, more uniform result with fewer distinct intensity levels. This metric helps evaluate the level of detail preserved or the smoothness achieved by the segmentation.

**Better range:** \
Context-dependent: \
**Low entropy** → simpler, smoother segmentation with fewer distinct regions (useful for coarse-grained segmentation or when reducing noise). \
**High entropy** → more detailed, complex segmentation with many distinct regions (useful for preserving fine details and textures in SAR images).

**Interpretation:** \
For SAR images, moderate entropy often indicates a good balance between noise reduction and detail preservation. Very low entropy might suggest over-smoothing, while very high entropy could indicate under-segmentation or insufficient noise reduction.

### Evalution Sheet

[Google Sheet Link](https://docs.google.com/spreadsheets/d/1SwYvWu2-6ZTNSZguZ7LxR_cQ_cC1RAzSuLoQ0ACM_14/edit?usp=sharing)
