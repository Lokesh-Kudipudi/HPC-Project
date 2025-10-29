#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Structure to represent a superpixel
struct Superpixel
{
    vector<Point> pixels;
    double meanIntensity;
    double variance;
    Point center;
    int id;

    Superpixel(int _id = -1) : id(_id), meanIntensity(0.0), variance(0.0) {}
};

// Structure to represent a graph edge
struct Edge
{
    int from, to;
    double weight;

    Edge(int f, int t, double w) : from(f), to(t), weight(w) {}

    bool operator<(const Edge &other) const
    {
        return weight < other.weight;
    }
};

class SARSegmentation
{
private:
    Mat originalImage;
    Mat image;
    int rows, cols;
    double intensityThreshold;
    int minSuperpixelSize;

    // Superpixel-related variables
    vector<vector<int>> superpixelMap;
    vector<Superpixel> superpixels;

    // Timing variables
    double time_growSuperpixel;
    double time_calculateSuperpixelProperties;
    double time_assignToNearestSuperpixel;

public:
    SARSegmentation(const Mat &img, double threshold = 0.015, int minSize = 10)
        : originalImage(img), intensityThreshold(threshold), minSuperpixelSize(minSize),
          time_growSuperpixel(0.0), time_calculateSuperpixelProperties(0.0), time_assignToNearestSuperpixel(0.0)
    {

        originalImage.copyTo(image);
        rows = image.rows;
        cols = image.cols;

        // Convert to grayscale if needed
        if (image.channels() == 3)
        {
            cvtColor(image, image, COLOR_BGR2GRAY);
        }

        // Apply bilateral filter to reduce noise while preserving edges
        Mat filtered;
        bilateralFilter(image, filtered, 9, 75, 75);
        image = filtered;

        // Normalize image to 0-1 range
        image.convertTo(image, CV_64F, 1.0 / 255.0);

        superpixelMap = vector<vector<int>>(rows, vector<int>(cols, -1));

        cout << "Image preprocessed. Size: " << cols << "x" << rows << endl;
    }

    // Extract superpixels using improved region growing
    void extractSuperpixels()
    {
        auto start_total = chrono::high_resolution_clock::now();

        cout << "Extracting superpixels..." << endl;

        int superpixelId = 0;
        vector<vector<bool>> visited(rows, vector<bool>(cols, false));

        // Use a grid-based initialization to ensure better coverage
        int stepSize = sqrt(minSuperpixelSize);

        for (int i = stepSize / 2; i < rows; i += stepSize)
        {
            for (int j = stepSize / 2; j < cols; j += stepSize)
            {
                if (!visited[i][j])
                {
                    Superpixel sp(superpixelId);

                    auto start_grow = chrono::high_resolution_clock::now();
                    growSuperpixel(i, j, sp, visited);
                    auto end_grow = chrono::high_resolution_clock::now();
                    time_growSuperpixel += chrono::duration_cast<chrono::duration<double>>(end_grow - start_grow).count();

                    if (sp.pixels.size() >= minSuperpixelSize)
                    {
                        auto start_calc = chrono::high_resolution_clock::now();
                        calculateSuperpixelProperties(sp);
                        auto end_calc = chrono::high_resolution_clock::now();
                        time_calculateSuperpixelProperties += chrono::duration_cast<chrono::duration<double>>(end_calc - start_calc).count();

                        superpixels.push_back(sp);
                        superpixelId++;
                    }
                    else
                    {
                        // Mark small regions as unassigned for later processing
                        for (const Point &p : sp.pixels)
                        {
                            superpixelMap[p.y][p.x] = -1;
                            visited[p.y][p.x] = false;
                        }
                    }
                }
            }
        }

        // Process remaining unvisited pixels
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!visited[i][j])
                {
                    // Assign to nearest superpixel
                    auto start_assign = chrono::high_resolution_clock::now();
                    assignToNearestSuperpixel(i, j);
                    auto end_assign = chrono::high_resolution_clock::now();
                    time_assignToNearestSuperpixel += chrono::duration_cast<chrono::duration<double>>(end_assign - start_assign).count();
                }
            }
        }

        cout << "Created " << superpixels.size() << " superpixels" << endl;

        auto end_total = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed_total = end_total - start_total;
        cout << "Time taken by extractSuperpixels: " << elapsed_total.count() << " seconds" << endl;
        cout << "  - Time in growSuperpixel: " << time_growSuperpixel << " seconds" << endl;
        cout << "  - Time in calculateSuperpixelProperties: " << time_calculateSuperpixelProperties << " seconds" << endl;
        cout << "  - Time in assignToNearestSuperpixel: " << time_assignToNearestSuperpixel << " seconds" << endl;
    }

    // Assign unprocessed pixel to nearest superpixel
    void assignToNearestSuperpixel(int x, int y)
    {
        if (superpixels.empty())
            return;

        double minDist = DBL_MAX;
        int nearestId = 0;
        double pixelIntensity = image.at<double>(x, y);

        for (int i = 0; i < superpixels.size(); i++)
        {
            double intensityDiff = abs(pixelIntensity - superpixels[i].meanIntensity);
            double spatialDist = sqrt(pow(x - superpixels[i].center.y, 2) +
                                      pow(y - superpixels[i].center.x, 2));
            double combinedDist = intensityDiff + 0.01 * spatialDist; // Weight spatial distance less

            if (combinedDist < minDist)
            {
                minDist = combinedDist;
                nearestId = i;
            }
        }

        superpixelMap[x][y] = nearestId;
        superpixels[nearestId].pixels.push_back(Point(y, x));
    }

    // Grow superpixel using region growing
    void growSuperpixel(int startX, int startY, Superpixel &sp, vector<vector<bool>> &visited)
    {
        queue<Point> toProcess;
        toProcess.push(Point(startY, startX));
        visited[startX][startY] = true;

        double seedIntensity = image.at<double>(startX, startY);

        while (!toProcess.empty())
        {
            Point current = toProcess.front();
            toProcess.pop();

            sp.pixels.push_back(current);
            superpixelMap[current.y][current.x] = sp.id;

            // Check 8-connected neighbors
            for (int dx = -1; dx <= 1; dx++)
            {
                for (int dy = -1; dy <= 1; dy++)
                {
                    if (dx == 0 && dy == 0)
                        continue;

                    int nx = current.y + dx;
                    int ny = current.x + dy;

                    if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && !visited[nx][ny])
                    {
                        double neighborIntensity = image.at<double>(nx, ny);
                        double diff = abs(seedIntensity - neighborIntensity);

                        if (diff < intensityThreshold)
                        {
                            visited[nx][ny] = true;
                            toProcess.push(Point(ny, nx));
                        }
                    }
                }
            }
        }
    }

    // Calculate superpixel properties
    void calculateSuperpixelProperties(Superpixel &sp)
    {
        double totalIntensity = 0.0;
        int sumX = 0, sumY = 0;

        for (const Point &p : sp.pixels)
        {
            totalIntensity += image.at<double>(p.y, p.x);
            sumX += p.x;
            sumY += p.y;
        }

        sp.meanIntensity = totalIntensity / sp.pixels.size();
        sp.center = Point(sumX / sp.pixels.size(), sumY / sp.pixels.size());

        // Calculate variance
        double sumSquaredDiff = 0.0;
        for (const Point &p : sp.pixels)
        {
            double diff = image.at<double>(p.y, p.x) - sp.meanIntensity;
            sumSquaredDiff += diff * diff;
        }
        sp.variance = sumSquaredDiff / sp.pixels.size();
    }

    // Build adjacency graph of superpixels
    vector<Edge> buildSuperpixelGraph()
    {
        auto start = chrono::high_resolution_clock::now();

        cout << "Building superpixel graph..." << endl;

        vector<Edge> edges;
        set<pair<int, int>> addedEdges;

        // Find adjacent superpixels
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int currentId = superpixelMap[i][j];
                if (currentId == -1)
                    continue;

                // Check 4-connected neighbors
                vector<pair<int, int>> neighbors = {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};

                for (auto [ni, nj] : neighbors)
                {
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols)
                    {
                        int neighborId = superpixelMap[ni][nj];

                        if (neighborId != -1 && currentId != neighborId)
                        {
                            pair<int, int> edgeKey = {min(currentId, neighborId), max(currentId, neighborId)};

                            if (addedEdges.find(edgeKey) == addedEdges.end())
                            {
                                // Enhanced edge weight calculation
                                double intensityDiff = abs(superpixels[currentId].meanIntensity -
                                                           superpixels[neighborId].meanIntensity);
                                double varianceSum = superpixels[currentId].variance + superpixels[neighborId].variance;
                                double weight = intensityDiff * (1.0 + varianceSum * 10.0);

                                edges.push_back(Edge(currentId, neighborId, weight));
                                addedEdges.insert(edgeKey);
                            }
                        }
                    }
                }
            }
        }

        cout << "Created graph with " << edges.size() << " edges" << endl;

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Time taken by buildSuperpixelGraph: " << elapsed.count() << " seconds" << endl;

        return edges;
    }

    // Union-Find data structure
    class UnionFind
    {
    public:
        vector<int> parent, rank, size;
        vector<double> threshold;

        UnionFind(int n, double k, const vector<Superpixel> &superpixels)
        {
            parent.resize(n);
            rank.resize(n, 0);
            size.resize(n);
            threshold.resize(n);

            for (int i = 0; i < n; i++)
            {
                parent[i] = i;
                size[i] = superpixels[i].pixels.size();
                threshold[i] = k / size[i];
            }
        }

        int find(int x)
        {
            if (parent[x] != x)
            {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        void unite(int x, int y, double weight)
        {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX != rootY)
            {
                if (rank[rootX] < rank[rootY])
                {
                    parent[rootX] = rootY;
                    size[rootY] += size[rootX];
                    threshold[rootY] = weight + threshold[rootY];
                }
                else if (rank[rootX] > rank[rootY])
                {
                    parent[rootY] = rootX;
                    size[rootX] += size[rootY];
                    threshold[rootX] = weight + threshold[rootX];
                }
                else
                {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                    size[rootX] += size[rootY];
                    threshold[rootX] = weight + threshold[rootX];
                }
            }
        }

        bool shouldMerge(int x, int y, double weight)
        {
            int rootX = find(x);
            int rootY = find(y);

            if (rootX == rootY)
                return false;

            return weight <= min(threshold[rootX], threshold[rootY]);
        }
    };

    // Perform graph-based segmentation
    vector<int> segmentGraph(const vector<Edge> &edges, double k = 150.0)
    {
        auto start = chrono::high_resolution_clock::now();

        cout << "Performing graph-based segmentation with k=" << k << endl;

        vector<Edge> sortedEdges = edges;
        sort(sortedEdges.begin(), sortedEdges.end());

        UnionFind uf(superpixels.size(), k, superpixels);
        int mergedCount = 0;

        for (const Edge &edge : sortedEdges)
        {
            if (uf.shouldMerge(edge.from, edge.to, edge.weight))
            {
                uf.unite(edge.from, edge.to, edge.weight);
                mergedCount++;
            }
        }

        // Map each superpixel to its final segment
        vector<int> segmentMap(superpixels.size());
        map<int, int> rootToSegment;
        int segmentId = 0;

        for (int i = 0; i < superpixels.size(); i++)
        {
            int root = uf.find(i);
            if (rootToSegment.find(root) == rootToSegment.end())
            {
                rootToSegment[root] = segmentId++;
            }
            segmentMap[i] = rootToSegment[root];
        }

        cout << "Created " << segmentId << " segments" << endl;

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Time taken by segmentGraph: " << elapsed.count() << " seconds" << endl;

        return segmentMap;
    }

    // Generate segmentation result with better color scheme
    Mat generateSegmentationResult(const vector<int> &segmentMap)
    {
        auto start = chrono::high_resolution_clock::now();
        cout << "Generating segmentation result..." << endl;

        int maxSegment = *max_element(segmentMap.begin(), segmentMap.end());

        // Create grayscale values for segments based on mean intensity
        vector<uchar> segmentColors(maxSegment + 1);
        vector<double> segmentIntensities(maxSegment + 1, 0.0);
        vector<int> segmentPixelCounts(maxSegment + 1, 0);

        // Calculate mean intensity for each segment
        for (int i = 0; i < superpixels.size(); i++)
        {
            int segment = segmentMap[i];
            segmentIntensities[segment] += superpixels[i].meanIntensity * superpixels[i].pixels.size();
            segmentPixelCounts[segment] += superpixels[i].pixels.size();
        }

        for (int i = 0; i <= maxSegment; i++)
        {
            if (segmentPixelCounts[i] > 0)
            {
                segmentIntensities[i] /= segmentPixelCounts[i];
                segmentColors[i] = (uchar)(segmentIntensities[i] * 255);
            }
        }

        // Create result image
        Mat result(rows, cols, CV_8UC1);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int superpixelId = superpixelMap[i][j];
                if (superpixelId >= 0 && superpixelId < segmentMap.size())
                {
                    int segment = segmentMap[superpixelId];
                    result.at<uchar>(i, j) = segmentColors[segment];
                }
                else
                {
                    result.at<uchar>(i, j) = 0;
                }
            }
        }

        // Apply morphological operations to clean up the result
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        morphologyEx(result, result, MORPH_CLOSE, kernel);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Time taken by generateSegmentationResult: " << elapsed.count() << " seconds" << endl;

        return result;
    }

    // Main segmentation function
    Mat segment()
    {
        cout << "Starting SAR image segmentation..." << endl;

        // Step 1: Extract superpixels
        extractSuperpixels();

        if (superpixels.empty())
        {
            cout << "No superpixels created!" << endl;
            return Mat::zeros(rows, cols, CV_8UC1);
        }

        // Step 2: Build superpixel graph
        vector<Edge> edges = buildSuperpixelGraph();

        if (edges.empty())
        {
            cout << "No edges in graph!" << endl;
            return Mat::zeros(rows, cols, CV_8UC1);
        }

        // Step 3: Try different k values to get good segmentation
        vector<double> kValues = {50.0, 100.0, 150.0, 200.0, 300.0};

        for (double k : kValues)
        {
            vector<int> segmentMap = segmentGraph(edges, k);
            int numSegments = *max_element(segmentMap.begin(), segmentMap.end()) + 1;

            // Look for reasonable number of segments (not too few, not too many)
            if (numSegments >= 5 && numSegments <= superpixels.size() / 2)
            {
                cout << "Using k=" << k << " with " << numSegments << " segments" << endl;
                return generateSegmentationResult(segmentMap);
            }
        }
        // Fallback
        vector<int> segmentMap = segmentGraph(edges, 150.0);
        return generateSegmentationResult(segmentMap);
    }
};

int main(int argc, char **argv)
{
    try
    {
        string path = argv[1];
        // Load SAR image
        cout << "Loading SAR image..." << endl;
        Mat image = imread(path, IMREAD_COLOR);

        if (image.empty())
        {
            cerr << "Error: Could not load image 'sar.png'" << endl;
            return -1;
        }

        cout << "Image loaded successfully. Size: " << image.cols << "x" << image.rows << endl;

        // Create SAR segmentation object
        SARSegmentation segmenter(image, 0.015, 25);

        // Perform segmentation
        Mat segmentationResult = segmenter.segment();

        fs::path p(path);
        string inputName = p.stem().string();

        string outputName = "sar_segmented_" + inputName + ".png";
        // Save results
        imwrite(outputName, segmentationResult);

        // Display results
        namedWindow("Original Image", WINDOW_AUTOSIZE);
        namedWindow("Segmentation Result", WINDOW_AUTOSIZE);
        imshow("Original Image", image);
        imshow("Segmentation Result", segmentationResult);

        cout << "Results saved as " << outputName << endl;
        cout << "Press any key to exit..." << endl;
        waitKey(0);

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }
}