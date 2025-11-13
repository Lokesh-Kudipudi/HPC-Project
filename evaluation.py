import cv2
import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from skimage.measure import shannon_entropy
import argparse
import torch


def evaluate_segmentation(original_path, segmented_path):
    # Load images
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    segmented = cv2.imread(segmented_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    original /= 255.0
    segmented_labels = segmented.astype(np.int32)

    # Flatten
    orig_flat = original.reshape(-1, 1)
    labels_flat = segmented_labels.reshape(-1)

    # Number of segments
    num_segments = len(np.unique(labels_flat))

    print("----- Segmentation Evaluation -----")
    print(f"Segments detected: {num_segments}")


    # 2. Calinski–Harabasz Score
    if num_segments > 1:
        ch = calinski_harabasz_score(orig_flat, labels_flat)
        print(f"Calinski–Harabasz Score: {ch:.4f}")
    else:
        print("Calinski–Harabasz Score: Not applicable")

    # 4. Intra-class variance
    intra_variances = []
    for label in np.unique(labels_flat):
        region_vals = orig_flat[labels_flat == label]
        intra_variances.append(np.var(region_vals))
    avg_intra_var = np.mean(intra_variances)
    print(f"Average Intra-class Variance: {avg_intra_var:.6f}")

    # 5. Inter-class separation (distance between class means)
    means = []
    for label in np.unique(labels_flat):
        means.append(np.mean(orig_flat[labels_flat == label]))
    means = np.array(means)

    if len(means) > 1:
        inter_sep = np.mean(np.abs(means[:, None] - means[None, :]))
        print(f"Mean Inter-class Separation: {inter_sep:.6f}")
    else:
        print("Mean Inter-class Separation: Not applicable")

    # 6. Entropy of the segmented image
    entropy_val = shannon_entropy(segmented)
    print(f"Segmentation Entropy: {entropy_val:.4f}")

    print("-----------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation without ground truth")
    parser.add_argument("--original", type=str, required=True, help="Path to original SAR image")
    parser.add_argument("--segmented", type=str, required=True, help="Path to segmented output")
    args = parser.parse_args()
    evaluate_segmentation(args.original, args.segmented)
