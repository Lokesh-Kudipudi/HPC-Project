# Observations Based on report_main_seq.txt

## Time taken by major functions ON 1280x819

- extractSuperPixels - 90.153s

  - growSuperPixel - 0.517s
  - calculateSuperPixelProperties - 0.023s
  - assignToNearestSuperPixel - 89.441s

- buildSuperPixelGraph - 0.656s

- segmentGraph - 0.007s

- generateSegmentationResult - 0.031s

## HotSpots With reasons why they are parallelizable or not.

- assignToNearestSuperPixel -
  - **Is it Parallelizable?** Yes.
  - **Reason:** The parent loop that calls this function iterates over all unassigned pixels. Each pixel can independently find its nearest superpixel. However, a race condition occurs when multiple threads try to add pixels to the _same_ superpixel's list.
- buildSuperPixelGraph -
  - **Is it Parallelizable?** Yes.
  - **Reason:** The loops iterate over every pixel to find adjacent superpixels. Each pixel can be processed independently. The main challenge is managing concurrent writes to the shared `edges` vector.
- growSuperPixel -
  - **Is it Parallelizable?** No.
  - **Reason:** This function uses a Breadth-First Search (queue-based region growing), which is inherently sequential. The state of the queue and the `visited` map in each step depends directly on the previous step, making it unsuitable for simple loop parallelization.
- calculateSuperPixelProperties -
  - **Is it Parallelizable?** Yes
  - **Reason:** The function processes a single superpixel. However, if you have a list of superpixels to process, a loop calling this function for each one can be parallelized because the calculations for one superpixel are independent of all others.
- generateSegmentationResult -
  - **Is it Parallelizable?** Yes.
  - **Reason:** This function contains two highly parallelizable loops. The first loop (aggregating segment properties) can be parallelized using atomic operations. The second loop (writing pixel colors to the final image) is also parallelizable as each pixel is processed independently.
- segmentGraph -
  - **Is it Parallelizable?** No.
  - **Reason:** The core of this algorithm iterates through a list of edges that **must be sorted** by weight. The Union-Find logic is sequential by nature, as the decision to merge two segments depends on all the merges that have occurred before it. Parallelizing this would break the algorithm.

## The Choosen Platform with Justification

- openMP
  - OpenMP is ideal for the fine-granularity parallelism present in the code's loops.
  - It efficiently parallelizes tasks (like distributing image rows to threads) on shared-memory CPUs.
- CUDA
  - CUDA is chosen for its strength in handling the fine-grained, data-parallel nature of the pixel-level hotspots.
