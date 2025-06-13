import numpy as np
import torch
from typing import Callable, List, Tuple, Set
from utils.logger import CustomLogger
logger = CustomLogger(__name__).logger


def count_common_elements(arr1, arr2):
    """
        Args

        Return:
    """
    common_elements = np.intersect1d(arr1, arr2)
    return len(common_elements)/len(arr1) * 100


def count_common_ranked_elements(arr1, arr2):
    """
        Args

        Return:
    """
    same_pos = 0
    common_elements = np.intersect1d(arr1, arr2)
    for element in common_elements:
        first = np.where(arr1 == element)[0][0]
        second = np.where(arr2 == element)[0][0]
        if first == second:
            same_pos += 1
    return same_pos/len(arr1) * 100


def top_k_coords(cam: np.ndarray,
                k: int) -> List[tuple]:
    """
        Args: cam: gradcam saliency map
              k: desired percentage of pixels

        Returns: List of top-k coordinates in saliency map
    """
    total_pixels = cam.shape[0]*cam.shape[1]
    num_pixels = int((k /100) * total_pixels)
    sorted_indices = np.argsort(cam.flatten())[::-1]
    top_k_indices = np.unravel_index(sorted_indices[:num_pixels], cam.shape) # evaluate if this is correct
    return list(zip(*top_k_indices))


def top_k_coords_over_threshold(cam: np.ndarray, k: float) -> List[Tuple[int, int]]:
    """
    Args:
        cam: Grad-CAM saliency map (2D array)
        k: The desired percentile (0 to 1.0), e.g., 0.5 for the 50th percentile.

    Returns:
        List of coordinates where the pixel value exceeds the kth percentile.
    """
    if not (0 <= k <= 1.0):
        raise ValueError("k must be between 0 and 1")

    percentile = k * 100
    # Calculate the percentile threshold value
    threshold_value = np.percentile(cam.flatten(), percentile)

    # Find the coordinates of pixels where the value exceeds the threshold
    top_k_indices = np.where(cam >= threshold_value)
    logger.info(f'Threshold Value: {threshold_value}')
    # Return the list of coordinates as tuples
    #return list(zip(top_k_indices[0], top_k_indices[1])), threshold_value
    return list(zip(top_k_indices[1], top_k_indices[0])), threshold_value


def agreement_coords(baseline_coords: List[tuple], 
                            rotated_coords: List[tuple]) -> float:
    """
        Args: baseline_coords: the list of top-k coordinates for the original explanation

              rotated_coords: the rotated version of the original explanation

        Returns:  The Percentage Agreement between the two explanations
    """
    sbc = set(baseline_coords)
    src = set(rotated_coords)
    common = len(sbc.intersection(src))
    return common / len(baseline_coords) * 100


def entropic_label_selector(softmax_scores: torch.Tensor) -> int:
    """
    Args:
    """
    highest_entropy = 0
    label_with_highest_entropy = None
    for label in range(softmax_scores.size(-1)):
        softmax_without_label = torch.cat((softmax_scores[:, :label], softmax_scores[:, label + 1:]), dim=-1)
        entropy = -torch.sum(softmax_without_label * torch.log2(softmax_without_label + 1e-20), dim=-1).item()
        if entropy > highest_entropy:
            highest_entropy = entropy
            label_with_highest_entropy = label
    return label_with_highest_entropy


def find_least_similar(weights_matrix: torch.Tensor, 
                       class_of_interest: int) -> int:
    """
    """
    reference_weights = weights_matrix[class_of_interest]
    similarities = torch.matmul(weights_matrix, reference_weights) / (torch.norm(weights_matrix, dim=1) * torch.norm(reference_weights))
    similarities[class_of_interest] = float('inf')
    least_similar_index = torch.argmin(similarities)
    sorted_values, sorted_indices = torch.sort(similarities, descending=False)
    #logger.info(sorted_values)
    idx=0
    for value in sorted_values:
        if value <0:
            pass
        else:
            lpi = idx
            break
        idx+=1
    lpi = sorted_indices[-2]
    lpv = sorted_values[-2]
    #lpi = (sorted_values > 0).nonzero(as_tuple=False).max()
    logger.info(f'Last Positive Index: {lpi}')
    logger.info(f'Last Positive Values: {lpv}')
    #exit()
    return lpi


def pearson_correlation(weights_matrix: torch.Tensor, 
                       class_of_interest: int) -> int:
    """

    Returns: the label with the highest 
            pearson product-moment correlation
    """
    wm = weights_matrix.cpu().detach().numpy()
    logger.info('*'*50)
    logger.info(wm.shape)
    logger.info('*'*50)
    cm = np.corrcoef(wm)
    i_scores = cm[class_of_interest]
    sorted_scores = np.sort(i_scores)
    sorted_indices = np.argsort(i_scores)
    index_closest_to_zero = max(range(len(sorted_scores)), key=lambda i: abs(sorted_scores[i]))
    nni = np.where(sorted_scores >= 0)[0]
    idx = nni[len(nni)//16]
    idx = index_closest_to_zero
    # 15, 5 - densenet
    # 230 -resnet
    return sorted_indices[nni[0]] # next closest; then try furthest


def fractional_metric_scores(weights_matrix: torch.Tensor,
                             class_of_interest: int,
                             p: float) -> int:
    """
    Return : Closest Weight based on Fractional Distance Metric
    """
    wm = weights_matrix.cpu().detach().numpy()
    vector_of_interest = wm[class_of_interest]
    scores = []
    for i, vector in enumerate(wm):
        distance = np.linalg.norm(vector_of_interest - vector, ord=p)
        scores.append((i, distance))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_scores] 
    contrast_class = sorted_indices[-3]
    # logger.info('*'*50)
    # logger.info(f'GT Class: {class_of_interest}')
    # logger.info(f'Contrast Class: {contrast_class}')
    # logger.info('*'*50)
    return contrast_class


def find_closest_to_zero(tuples):
    """
    tuples: index, score
    """
    closest_tuple = None
    closest_score = float('inf')  # Set to positive infinity initially

    for index, score in tuples:
        # Check if the absolute value of the current score is closer to zero
        if abs(score) < abs(closest_score):
            closest_score = score
            closest_tuple = (index, score)
    return closest_tuple


def find_closest_to_one(tuples):
    """
    tuples: index, score
    """
    closest_tuple = None
    closest_distance_to_one = float('inf')  # Set to positive infinity initially

    for index, score in tuples:
        # Check if the absolute value of the difference from 1 is smaller
        if abs(1 - abs(score)) < abs(1 - abs(closest_distance_to_one)):
            closest_distance_to_one = 1 - abs(score)
            closest_tuple = (index, score)
    return closest_tuple


def find_closest_to_minus_one(tuples):
    """
    tuples: index, score
    """
    closest_tuple = None
    closest_distance_to_minus_one = float('inf')  # Set to positive infinity initially

    for index, score in tuples:
        # Check if the absolute value of the difference from -1 is smaller
        if abs(-1 - abs(score)) < abs(-1 - abs(closest_distance_to_minus_one)):
            closest_distance_to_minus_one = -1 - abs(score)
            closest_tuple = (index, score)
    return closest_tuple



def find_most_collinear_vector(weights_matrix: torch.Tensor, 
                               class_of_interest: int,
                               p: float) -> int:
    """
    Find the weight vector in the given weight matrix that is most collinear
    with the weight vector of the specified class.

    Parameters:
    - weight_matrix: 2D NumPy array representing the weight matrix (shape: [features, classes])
    - class_of_interest: Index of the class for which to find the most collinear vector.

    Returns:
    - most_collinear_vector: The weight vector that is most collinear with the specified class.
    """

    wm = weights_matrix.cpu().detach().numpy()
    vector_of_interest = wm[class_of_interest]
    scores = []
    # Initialize variables to store the most collinear vector and its similarity
    for i, vector in enumerate(wm):
        similarity = np.dot(vector_of_interest, vector) / (np.linalg.norm(vector_of_interest, ord=p) * np.linalg.norm(vector, ord=p))
        scores.append((i, similarity))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_scores] 
    #logger.info(sorted_scores)
    positive_scores = [(index, score) for index, score in sorted_scores if score > 0]
    ci = 3
    #ci = -1
    #ci = -1
    # contrast_class = positive_scores[ci][0]
    # contrast_alpha = positive_scores[ci][1]
    # ci = 2
    contrast_alpha = sorted_scores[ci][1]
    contrast_class = sorted_scores[ci][0]
    # # logger.info(sorted_scores)
    # # exit()
    # #logger.info(contrast_class)
    #return find_closest_to_zero(positive_scores)
    return contrast_class, contrast_alpha


def get_positive_indices_and_apply_mask(grad_tensor: torch.Tensor, 
                                        activation_tensor: torch.Tensor)-> torch.Tensor:
    """
    Args:
        grad_tensor: tensor to generate mask
        activation_tensor: tensor to apply mask to

    Returns: positive activations with positive gradients
    """
    hirescam = grad_tensor * activation_tensor
    return torch.clamp(hirescam, min=0.0)


def scale_map(arr: np.ndarray) -> np.ndarray:
    """
    Args:

    Returns:
    """
    logger.info(arr.shape)
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    # Avoid division by zero if all elements are the same
    if min_val == max_val:
        return np.zeros_like(arr)
    
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr


def baseline_label(softmax_scores: np.ndarray) -> int:
    """
    Returns: the label for the score
    closest to the baseline label
    """
    scores = softmax_scores.squeeze().cpu().detach().numpy()
    base_conf = 1/1000
    idx = np.abs(scores - base_conf).argmin()
    logger.info(idx)
    return idx


def singular_projection(weight_matrix: np.ndarray,
                        class_of_interest: int,
                        variance_proportion:float=0.9,
                        percentile:float=50.0) -> int:
    """
    """
    U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)
    total_variance = np.sum(S ** 2)
    explained_variance = np.cumsum(S ** 2) / total_variance
    num_singular_values = np.argmax(explained_variance >= variance_proportion) + 1
    class_weight_vector = weight_matrix[class_of_interest]
    reduced_weight_vector = np.dot(class_weight_vector, V[:num_singular_values].T)
    similarities = []
    for label, weight_vector in enumerate(weight_matrix):
        if label != class_of_interest:
            reduced_other_vector = np.dot(weight_vector, V[:num_singular_values].T)
            similarity = np.dot(reduced_weight_vector, reduced_other_vector) \
                / (np.linalg.norm(reduced_weight_vector) \
                * np.linalg.norm(reduced_other_vector))
            similarities.append((label, similarity))
    contrast_label, _ = max(similarities, key=lambda x: x[1])
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    scores = [score for _, score in sorted_similarities if score > 0]
    percentile_score = np.percentile(scores, q=percentile) # low percentiile for rank deficient models
    closest_score = min(scores, key=lambda x: abs(x - percentile_score))
    contrast_class = None
    for class_, score in sorted_similarities:
        if score == closest_score:
            contrast_class = class_
            break
    return contrast_class


def find_common_tuples(list_of_tuples_lists) -> Set[tuple]:
    """
    Find tuples that are common to all lists of tuples.

    Args:
        list_of_tuples_lists (list of list of tuples): A list where each element is a list of tuples
                                                       representing the coordinates for N images.

    Returns:
        set: A set of tuples that are common across all lists.
    """
    if not list_of_tuples_lists:
        return set()  # Return empty set if the input list is empty

    # Convert each list of tuples to a set and perform set intersection
    common_tuples = set(list_of_tuples_lists[0])  # Initialize with the first list of tuples
    for tuples_list in list_of_tuples_lists[1:]:
        common_tuples &= set(tuples_list)  # Intersect with subsequent lists

    return common_tuples


def zero_out_common_coords(explanations, 
                           common_coords,
                           fill_values) -> List[np.ndarray]:
    """
    Set values at the common coordinates to zero for each explanation.

    Args:
        explanations (list of ndarray): List of 224x224 ndarrays representing the explanations.
        common_coords (set of tuples): Set of common coordinates where values should be set to zero.

    Returns:
        list of ndarray: List of modified explanations with values at common coordinates set to zero.
    """
    modified_explanations = []

    for i, exp in enumerate(explanations):
        # Make a copy of the explanation to avoid modifying the original
        modified_exp = np.copy(exp)

        # Set values at common coordinates to zero
        for coord in common_coords:
            x, y = coord
            modified_exp[y, x] = fill_values[i]
            #modified_exp[x, y] = -1

        modified_explanations.append(modified_exp)

    return modified_explanations


def remove_common_pixels(A, B):
    """
    Remove pixels common to both lists.

    A: The Saliency Map with CMFS
    B: Saliency Map with UDFS

    Returns A': With ONLY CMFs allowed
    """
    # Convert B to a set for O(1) lookups
    set_B = set(B)

    # Create a new list excluding elements that are in B
    result = [pixel for pixel in A if pixel not in set_B]

    return result


def get_semantically_similar_classes(class_idx: int,
                                     confusion_matrix: np.ndarray,
                                     top_k: int = 1,
                                     exclude_self: bool = True,
                                     normalize: bool = True) -> List[int]:
    """
    Returns the top-k most semantically similar (confused) classes
    for a given true class index based on the confusion matrix.

    Args:
        class_idx (int): True class index.
        confusion_matrix (np.ndarray): [C x C] confusion matrix.
        top_k (int): Number of confused classes to return.
        exclude_self (bool): Whether to exclude the diagonal entry.
        normalize (bool): Whether to normalize rows to proportions.

    Returns:
        List[int]: Indices of the top-k most confused classes.
    """
    conf = confusion_matrix.copy()

    if normalize:
        row_sums = conf.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        conf = conf / row_sums

    row = conf[class_idx].copy()
    if exclude_self:
        row[class_idx] = 0

    top_k_indices = np.argsort(row)[-top_k:][::-1]  # descending order
    return top_k_indices.tolist()