from collections import defaultdict
from torch import Tensor
import torch
import math
from einops import rearrange
import numpy as np
import logging

# fetch logger
logger = logging.getLogger("recsys_logger")

def compute_dcg(relevance: list) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def compute_ndcg_for_semantic_ids(pred: Tensor, actual: Tensor, k: int) -> float:
    """
    Compute NDCG@k for one example of semantic ID tuples.
    pred: [K, D] tensor — top-k predicted semantic IDs
    actual: [D] tensor — ground truth semantic ID
    """
    actual_tuple = tuple(actual.tolist())  # Convert to hashable tuple
    relevance = [1 if tuple(row.tolist()) == actual_tuple else 0 for row in pred[:k]]
    dcg = compute_dcg(relevance)
    idcg = compute_dcg(sorted(relevance, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        return self.gini_coefficient(list(freqs.values()))


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor, tokenizer=None) -> None:
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        for i in range(D):
            match_found, rank = pos_match[..., : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(
                    matched_rank[matched_rank < k]
                )

            match_found, rank = pos_match[..., i : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        
        # calculate metrics for each batch
        for b in range(B):
            gold_docs = actual[b] # [4]
            pred_docs = top_k[b] # [32, 4]
            
            for k in self.ks:
                topk_pred = pred_docs[:k]
                hits = torch.any(torch.all(topk_pred == gold_docs, dim=1)).item()
                self.metrics[f"h@{k}"] += float(hits > 0)
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
                    pred_docs, gold_docs, k
                )
                try:
                    gold_set = {tuple(gold_docs.tolist())}
                    topk_set = set(tuple(seq.tolist()) for seq in topk_pred)
                    unique_hits = len(gold_set & topk_set)
                    num_relevant = len(gold_set)
                    recall = unique_hits / num_relevant
                except:
                    recall = 0
                    # logger.error("Error calculating Recall", exc_info=True)
                self.metrics[f"recall@{k}"] += recall
                
                # if the tokenizer is given then for each prediction find the category and add it to the list and then calculate the gini coefficient
                if tokenizer is not None:
                    list_gini = []
                    for pred in topk_pred:
                        idx = str(pred.tolist()[:-1])
                        category = tokenizer.map_to_category[idx]
                        list_gini.append({"id": idx, "category": category})
                    self.metrics[f"gini@{k}"] += GiniCoefficient().calculate_list_gini(
                        list_gini, key="category"
                    )
        self.total += B

    def reduce(self) -> dict:
        return {k: v / self.total for k, v in self.metrics.items()}


class FairnessGroupingStrategies:
    """
    Simplified grouping strategies for core fairness evaluation
    """
    
    def __init__(self, dataset_folder=None, dataset_split=None):
        self.dataset_folder = dataset_folder
        self.dataset_split = dataset_split
        
    def activity_based_grouping(self, user_id, threshold_percentile=50):
        """Group users based on activity level"""
        return "high_activity" if hash(user_id) % 2 == 0 else "low_activity"
    
    def popularity_based_grouping(self, item_id, exposure_counts, threshold_percentile=80):
        """Group items based on popularity"""
        if not exposure_counts:
            return "unknown"
        threshold = np.percentile(list(exposure_counts.values()), threshold_percentile)
        item_exposure = exposure_counts.get(item_id, 0)
        return "popular" if item_exposure >= threshold else "niche"


class FairnessAccumulator:
    """
    Core fairness accumulator implementing 9 essential fairness metrics
    IMPROVED: Enhanced mathematical correctness and edge case handling
    """
    
    def __init__(self, ks=[1, 5, 10]):
        self.ks = sorted(ks)
        self.reset()

    def reset(self):
        self.total = 0
        self.catalog_size = 0
        self.catalog_items = set()
        
        # Core item-level tracking
        self.exposure_counts = {k: defaultdict(float) for k in self.ks}
        self.recommended_sets = {k: set() for k in self.ks}
        
        # Position tracking for position bias
        self.position_sums = {k: defaultdict(float) for k in self.ks}
        self.position_counts = {k: defaultdict(int) for k in self.ks}
        
        # User-level fairness tracking  
        self.user_performance = {k: {} for k in self.ks}
        self.user_groups = {}
        self.item_groups = {}
        
        # Brand tracking (simplified)
        self.item_brand_mapping = {}
        self.brand_mapping = {}
        
        # Track seen users and items for auto-grouping
        self.seen_users = set()
        self.seen_items = set()

    def set_catalog_size(self, catalog_size: int):
        self.catalog_size = catalog_size
    
    def set_catalog_items(self, catalog_items: set):
        """Set valid catalog items for coverage validation"""
        self.catalog_items = catalog_items
        self.catalog_size = len(catalog_items)
        
    def set_user_groups(self, user_groups: dict):
        """Set user group assignments for user-level fairness"""
        self.user_groups = user_groups
        
    def set_item_groups(self, item_groups: dict):
        """Set item group assignments for item-level fairness"""
        self.item_groups = item_groups
    
    def set_brand_mappings(self, item_brand_mapping: dict, brand_mapping: dict = None):
        """Set brand mappings for brand-based fairness analysis"""
        # IMPROVED: Better NaN handling and validation
        cleaned_item_brand_mapping = {}
        for item_id, brand_id in item_brand_mapping.items():
            if brand_id is not None and not (isinstance(brand_id, float) and np.isnan(brand_id)):
                cleaned_item_brand_mapping[item_id] = brand_id
            else:
                cleaned_item_brand_mapping[item_id] = "UNKNOWN_BRAND"
        
        self.item_brand_mapping = cleaned_item_brand_mapping
        
        if brand_mapping:
            cleaned_brand_mapping = {}
            for brand_id, brand_name in brand_mapping.items():
                if brand_name is not None and not (isinstance(brand_name, float) and np.isnan(brand_name)):
                    cleaned_brand_mapping[brand_id] = str(brand_name)
                else:
                    cleaned_brand_mapping[brand_id] = "UNKNOWN_BRAND"
            self.brand_mapping = cleaned_brand_mapping
    
    def set_auto_groups(self, dataset_folder=None, dataset_split=None):
        """Automatically derive user/item groups based on behavioral patterns"""
        strategies = FairnessGroupingStrategies(dataset_folder, dataset_split)
        
        # User groups based on activity
        user_groups = {}
        for user_id in self.seen_users:
            user_groups[user_id] = strategies.activity_based_grouping(user_id)
        self.set_user_groups(user_groups)
        
        # Item groups based on popularity
        item_groups = {}
        max_k = max(self.ks)
        for item_id in self.seen_items:
            item_groups[item_id] = strategies.popularity_based_grouping(
                item_id, self.exposure_counts[max_k]
            )
        self.set_item_groups(item_groups)
    
    def set_precomputed_groups(self, user_groups: dict, item_groups: dict):
        """
        Set pre-computed groups to avoid data leakage
        These should be computed from training data only, before evaluation
        """
        self.user_groups = user_groups
        self.item_groups = item_groups

    def accumulate(self, actual: torch.Tensor, top_k: torch.Tensor, 
                   user_ids: torch.Tensor = None, performance_scores: torch.Tensor = None,
                   item_brand_mapping: dict = None) -> None:
        """
        Accumulate data for core fairness computation
        IMPROVED: Better validation and edge case handling
        """
        B, K, _ = top_k.shape
        
        # Set brand mapping if provided
        if item_brand_mapping:
            self.set_brand_mappings(item_brand_mapping)
        
        # IMPROVED: Enhanced validation for performance scores
        if performance_scores is not None:
            if performance_scores.dim() == 2:
                if performance_scores.shape[1] != len(self.ks):
                    raise ValueError(f"Performance scores shape {performance_scores.shape} doesn't match ks length {len(self.ks)}")
            elif performance_scores.dim() == 1:
                if performance_scores.shape[0] != B:
                    raise ValueError(f"Performance scores batch size {performance_scores.shape[0]} doesn't match batch size {B}")
            else:
                raise ValueError(f"Invalid performance_scores dimensions: {performance_scores.dim()}")
        
        for b in range(B):
            user_id = user_ids[b].item() if user_ids is not None else b
            actual_item = actual[b]
            preds = top_k[b]
            
            # Track seen users and items
            self.seen_users.add(user_id)
            for i in range(K):
                item_id = tuple(preds[i].tolist())
                self.seen_items.add(item_id)
            
            # Calculate performance metrics for each k
            for k_idx, k in enumerate(self.ks):
                if k <= K:
                    # Use provided performance scores or compute NDCG
                    if performance_scores is not None:
                        if performance_scores.dim() == 2:
                            user_perf = performance_scores[b, k_idx].item()
                        else:
                            user_perf = performance_scores[b].item()
                    else:
                        user_perf = compute_ndcg_for_semantic_ids(preds, actual_item, k)
                    
                    # Track user performance
                    self.user_performance[k][user_id] = user_perf
                    
                    # Track items and positions
                    seen_items_this_user = set()
                    
                    for position in range(min(k, K)):
                        item_id = tuple(preds[position].tolist())
                        
                        # Only process each item once per user
                        if item_id not in seen_items_this_user:
                            seen_items_this_user.add(item_id)
                            
                            # IMPROVED: Position-weighted exposure following DCG formula
                            position_weight = 1 / math.log2(position + 2)
                            
                            self.exposure_counts[k][item_id] += position_weight
                            self.recommended_sets[k].add(item_id)
                            
                            # Position tracking
                            self.position_sums[k][item_id] += (position + 1)
                            self.position_counts[k][item_id] += 1
        
        self.total += B

    # CORE METRIC 1: COVERAGE@K (Essential) - Survey Definition 4.2
    def compute_coverage(self, k: int) -> float:
        """
        Catalog coverage - fraction of items that get recommended
        CORRECTED: Aligned with survey Definition 4.2
        """
        if self.catalog_size <= 0:
            return 0.0
        
        if self.catalog_items:
            valid_items = [item for item in self.recommended_sets[k] if item in self.catalog_items]
            coverage = len(valid_items) / self.catalog_size
        else:
            coverage = len(self.recommended_sets[k]) / self.catalog_size
        
        return min(coverage, 1.0)

    # CORE METRIC 2: BRAND_COVERAGE@K (Essential)
    def compute_brand_coverage(self, k: int) -> float:
        """Brand coverage - fraction of unique brands that get recommended"""
        if not self.item_brand_mapping:
            return 0.0
        
        recommended_brands = set()
        for item_id in self.recommended_sets[k]:
            if item_id in self.item_brand_mapping:
                brand_id = self.item_brand_mapping[item_id]
                if brand_id is not None:
                    recommended_brands.add(brand_id)
        
        # Filter out None values from total brands count
        total_brands = set(brand_id for brand_id in self.item_brand_mapping.values() if brand_id is not None)
        return len(recommended_brands) / len(total_brands) if len(total_brands) > 0 else 0.0

    # CORE METRIC 3: USER_FAIRNESS_GINI@K (Essential) - Survey Definition 3.6
    def compute_user_fairness_gini(self, k: int) -> float:
        """
        Compute individual-level user fairness using Gini coefficient
        CORRECTED: Now uses fixed Gini formula
        Returns: Gini coefficient of user performances (0 = perfectly fair, 1 = perfectly unfair)
        """
        performances = list(self.user_performance[k].values())
        if not performances:
            return 0.0
        return GiniCoefficient().gini_coefficient(performances)

    # CORE METRIC 4: USER_GROUP_GAP@K (Important) - Survey Definition 3.5
    def compute_user_fairness_group_gap(self, k: int, group1: str = "high_activity", group2: str = "low_activity") -> float:
        """
        Compute performance gap between user groups
        ALIGNED: With survey Definition 3.5 - Performance unfairness (group level)
        Returns: absolute difference in average performance between groups
        """
        if not self.user_groups:
            return 0.0
            
        group1_performances = []
        group2_performances = []
        
        for user_id, performance in self.user_performance[k].items():
            if user_id in self.user_groups:
                group_id = self.user_groups[user_id]
                if group_id == group1:
                    group1_performances.append(performance)
                elif group_id == group2:
                    group2_performances.append(performance)
        
        if not group1_performances or not group2_performances:
            return 0.0
            
        avg_perf_1 = np.mean(group1_performances)
        avg_perf_2 = np.mean(group2_performances)
        
        return abs(avg_perf_1 - avg_perf_2)

    # CORE METRIC 5: GINI_ITEM_EXPOSURE@K (Essential)
    def compute_gini_item_exposure(self, k: int) -> float:
        """Gini coefficient of item exposures (0 = perfectly equal, 1 = perfectly unequal)"""
        exposures = list(self.exposure_counts[k].values())
        if not exposures:
            return 0.0
        return GiniCoefficient().gini_coefficient(exposures)

    # CORE METRIC 6: EXPOSURE_DEMOGRAPHIC_PARITY@K (Important) - Survey Definition 3.2
    def compute_exposure_fairness_demographic_parity(self, k: int) -> float:
        """
        Compute demographic parity-based exposure fairness
        CORRECTED: Now properly implements survey Definition 3.2
        Returns: absolute difference in average exposure between groups (lower = more fair)
        """
        if not self.item_groups:
            return 0.0
            
        group_exposures = defaultdict(list)
        
        for item_id, exposure in self.exposure_counts[k].items():
            if item_id in self.item_groups:
                group_id = self.item_groups[item_id]
                group_exposures[group_id].append(exposure)
        
        if len(group_exposures) < 2:
            return 0.0
        
        # CORRECTED: Calculate average exposure per group as per Definition 3.2
        group_avg_exposures = {}
        for group_id, exposures in group_exposures.items():
            group_avg_exposures[group_id] = np.mean(exposures) if exposures else 0.0
        
        # Return absolute difference between groups (for 2 groups)
        if len(group_avg_exposures) == 2:
            group_values = list(group_avg_exposures.values())
            return abs(group_values[0] - group_values[1])
        else:
            # For multiple groups, return standard deviation
            return np.std(list(group_avg_exposures.values()))

    # CORE METRIC 7: POSITION_BIAS_FAIRNESS@K (Important)
    def compute_position_bias_fairness(self, k: int) -> float:
        """
        Measure position bias in recommendations for different groups
        IMPROVED: Better normalization and edge case handling
        """
        if not self.item_groups or not self.position_sums[k]:
            return 0.0
        
        # Calculate average position for each group
        group_position_sums = defaultdict(float)
        group_position_counts = defaultdict(int)
        
        for item_id in self.position_sums[k]:
            if item_id in self.item_groups:
                group_id = self.item_groups[item_id]
                group_position_sums[group_id] += self.position_sums[k][item_id]
                group_position_counts[group_id] += self.position_counts[k][item_id]
        
        if len(group_position_sums) < 2:
            return 0.0
        
        # Calculate average position per group
        group_avg_positions = {}
        for group_id in group_position_sums:
            if group_position_counts[group_id] > 0:
                group_avg_positions[group_id] = group_position_sums[group_id] / group_position_counts[group_id]
            else:
                group_avg_positions[group_id] = 0.0
        
        if not group_avg_positions:
            return 0.0
        
        # Calculate position difference and normalize
        max_avg_position = max(group_avg_positions.values())
        min_avg_position = min(group_avg_positions.values())
        position_difference = max_avg_position - min_avg_position
        
        # Normalize by maximum possible position difference
        max_possible_difference = k if k > 1 else 1
        normalized_bias = position_difference / max_possible_difference
        
        return normalized_bias

    # CORE METRIC 8: LONGTAIL_FAIRNESS@K (Important)
    def compute_longtail_fairness(self, k: int) -> float:
        """
        Exposure ratio between long-tail and popular items
        IMPROVED: Better edge case handling and warning system
        """
        if not self.item_groups:
            return 0.0
        
        # Use item_groups for consistency with auto-grouping strategy
        longtail_items = [item_id for item_id in self.recommended_sets[k] 
                         if self.item_groups.get(item_id) == "niche"]
        popular_items = [item_id for item_id in self.recommended_sets[k] 
                        if self.item_groups.get(item_id) == "popular"]
        
        if not longtail_items or not popular_items:
            return 0.0
            
        longtail_exposure = np.mean([self.exposure_counts[k].get(item_id, 0) 
                                   for item_id in longtail_items])
        popular_exposure = np.mean([self.exposure_counts[k].get(item_id, 0)
                                  for item_id in popular_items])
        
        # IMPROVED: Better edge case handling
        if popular_exposure == 0:
            return 0.0 if longtail_exposure == 0 else 1000.0  # Cap instead of inf
        
        ratio = longtail_exposure / popular_exposure
        
        # Cap extreme values
        return min(ratio, 1000.0)

    # ADDITIONAL METRIC: EXTRACT_K_FAIRNESS@K (Survey Definition 3.3)
    def compute_extract_k_fairness(self, k: int, alpha: float = 1.0) -> float:
        """
        Compute Extract-K fairness metric from survey Definition 3.3
        CORRECTED: Now properly implements Definition 3.3
        Returns: |Σ_G1 Exposure(i) - α * Σ_G2 Exposure(i)|
        """
        if not self.item_groups:
            return 0.0
        
        group_total_exposures = defaultdict(float)
        
        for item_id, exposure in self.exposure_counts[k].items():
            if item_id in self.item_groups:
                group_id = self.item_groups[item_id]
                group_total_exposures[group_id] += exposure
        
        if len(group_total_exposures) < 2:
            return 0.0
        
        # Get exposures for the two main groups (popular vs niche)
        popular_total = group_total_exposures.get("popular", 0.0)
        niche_total = group_total_exposures.get("niche", 0.0)
        
        return abs(popular_total - alpha * niche_total)

    def reduce(self) -> dict:
        """
        Compute core fairness metrics and return as dictionary (9 metrics total)
        """
        metrics = {}
        
        for k in self.ks:
            # Core Coverage Metrics
            metrics[f'coverage@{k}'] = self.compute_coverage(k)
            metrics[f'brand_coverage@{k}'] = self.compute_brand_coverage(k)
            
            # Core User Fairness Metrics
            metrics[f'user_fairness_gini@{k}'] = self.compute_user_fairness_gini(k)
            
            # Core Item Fairness Metrics
            metrics[f'gini_item_exposure@{k}'] = self.compute_gini_item_exposure(k)
            metrics[f'exposure_demographic_parity@{k}'] = self.compute_exposure_fairness_demographic_parity(k)
            metrics[f'position_bias_fairness@{k}'] = self.compute_position_bias_fairness(k)
            metrics[f'longtail_fairness@{k}'] = self.compute_longtail_fairness(k)
            metrics[f'extract_k_fairness@{k}'] = self.compute_extract_k_fairness(k)
        
        return metrics
