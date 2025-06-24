from collections import defaultdict
from torch import Tensor
import torch
import math
from einops import rearrange
import numpy as np
from typing import Dict, List, Optional
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


def compute_user_activity_groups(train_dataset, debug=False) -> Dict[str, str]:
    """
    Compute user activity groups based on training interaction patterns
    
    Based on Ekstrand et al. (2018) methodology for activity-based user grouping:
    "All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and Effectiveness"
    
    Args:
        train_dataset: Training dataset with user interaction sequences
    
    Returns:
        Dict mapping user_id (str) to group_label ("heavy", "medium", "light")
    """
    
    # Step 1.1: Extract user interaction statistics
    user_interaction_counts = {}
    sample_count = 0
    
    logger.info("Analyzing training data for user activity grouping...")
    
    for i, sample in enumerate(train_dataset):
        sample_count += 1
        
        # DEBUG: Log first few samples to understand structure
        if i < 3:
            logger.info(f"Sample {i} structure: {type(sample)}")
            if hasattr(sample, '_fields'):
                logger.info(f"  SeqBatch fields: {sample._fields}")
            logger.info(f"  user_ids: {sample.user_ids}, type: {type(sample.user_ids)}")
            logger.info(f"  ids shape: {sample.ids.shape if hasattr(sample.ids, 'shape') else 'N/A'}")
            logger.info(f"  seq_mask shape: {sample.seq_mask.shape if hasattr(sample.seq_mask, 'shape') else 'N/A'}")
        
        # Extract user ID from SeqBatch structure
        if hasattr(sample, 'user_ids'):
            user_id = sample.user_ids
        else:
            logger.warning(f"Could not extract user_ids from sample {i}, skipping...")
            continue
        
        # Convert to string for consistency
        user_id = str(user_id.item() if hasattr(user_id, 'item') else user_id)
        
        # Count interactions using seq_mask (counts valid, non-padded items)
        if hasattr(sample, 'seq_mask'):
            # seq_mask is True for valid items, False for padding (-1)
            interaction_count = sample.seq_mask.sum().item()
        elif hasattr(sample, 'ids'):
            # Fallback: count non-negative items (items >= 0, since -1 is padding)
            interaction_count = (sample.ids >= 0).sum().item()
        else:
            # Last resort: default to 1 interaction
            interaction_count = 1
            logger.warning(f"Using default interaction count for sample {i}")
        
        # Aggregate interactions per user
        if user_id not in user_interaction_counts:
            user_interaction_counts[user_id] = 0
        user_interaction_counts[user_id] += interaction_count
    
    # DEBUG: Log data extraction summary
    logger.info(f"Data extraction summary:")
    logger.info(f"  Total samples processed: {sample_count}")
    logger.info(f"  Unique users found: {len(user_interaction_counts)}")
    
    # Log some interaction count examples
    if len(user_interaction_counts) > 0:
        sample_users = list(user_interaction_counts.items())[:5]
        logger.info(f"  Sample user interaction counts: {sample_users}")
    
    # Step 1.2: Calculate activity percentiles (Ekstrand et al., 2018)
    if len(user_interaction_counts) == 0:
        logger.error("No user interactions found in training data!")
        return {}
    
    activity_scores = list(user_interaction_counts.values())
    
    # DEBUG: Log activity score distribution
    logger.info(f"Activity Score Distribution:")
    logger.info(f"  Total users: {len(activity_scores)}")
    logger.info(f"  Min activity: {min(activity_scores)}")
    logger.info(f"  Max activity: {max(activity_scores)}")
    logger.info(f"  Mean activity: {np.mean(activity_scores):.2f}")
    logger.info(f"  Median activity: {np.median(activity_scores):.2f}")
    logger.info(f"  Unique scores: {len(set(activity_scores))}")
    
    # Log distribution percentiles for debugging
    percentiles = [10, 25, 33, 50, 66, 75, 90]
    for p in percentiles:
        val = np.percentile(activity_scores, p)
        logger.info(f"  P{p}: {val:.2f}")
    
    # Count users at different activity levels
    zero_activity = sum(1 for s in activity_scores if s == 0)
    low_activity = sum(1 for s in activity_scores if 0 < s <= 5)
    logger.info(f"  Users with 0 activity: {zero_activity}")
    logger.info(f"  Users with 1-5 activity: {low_activity}")
    
    p33 = np.percentile(activity_scores, 33)
    p66 = np.percentile(activity_scores, 66)
    
    logger.info(f"  P33 threshold: {p33:.2f}")
    logger.info(f"  P66 threshold: {p66:.2f}")
    
    # Handle case where P33 = P66 (many users with same activity)
    if p33 == p66:
        logger.warning(f"P33 equals P66 ({p33:.2f}) - using alternative grouping strategy")
        
        # Strategy 1: Use unique value-based thresholds
        unique_scores = sorted(set(activity_scores))
        if len(unique_scores) >= 3:
            # Use 1/3 and 2/3 of unique values
            third_idx = len(unique_scores) // 3
            two_third_idx = 2 * len(unique_scores) // 3
            p33 = unique_scores[third_idx] - 0.5  # Slightly lower to ensure inclusion
            p66 = unique_scores[two_third_idx] - 0.5
            logger.info(f"  Unique-based thresholds: P33={p33:.2f}, P66={p66:.2f}")
        
        # Strategy 2: If still equal or too few unique values, use count-based grouping  
        if p33 == p66 or len(unique_scores) < 3:
            logger.warning("Using count-based grouping instead of percentile-based")
            
            # Sort users by activity, then split by count
            sorted_users = sorted(user_interaction_counts.items(), key=lambda x: x[1])
            n_users = len(sorted_users)
            
            # Create roughly equal groups
            light_end = n_users // 3
            medium_end = 2 * n_users // 3
            
            # Get threshold values
            if light_end < n_users and medium_end < n_users:
                p33 = sorted_users[light_end][1] - 0.5
                p66 = sorted_users[medium_end][1] - 0.5
                logger.info(f"  Count-based thresholds: P33={p33:.2f}, P66={p66:.2f}")
            else:
                # Final fallback: force artificial split
                mean_score = np.mean(activity_scores)
                p33 = mean_score - 0.5
                p66 = mean_score + 0.5
                logger.info(f"  Forced split around mean: P33={p33:.2f}, P66={p66:.2f}")
    
    # Step 1.3: Assign group labels based on activity thresholds
    user_activity_groups = {}
    group_stats = {"heavy": 0, "medium": 0, "light": 0}
    
    # First attempt: Use percentile thresholds
    for user_id, activity_score in user_interaction_counts.items():
        if activity_score >= p66:
            group = "heavy"
        elif activity_score >= p33:
            group = "medium"
        else:
            group = "light"
        
        user_activity_groups[user_id] = group
        group_stats[group] += 1
    
    # Check if we have a balanced distribution
    total_users = len(user_activity_groups)
    light_ratio = group_stats["light"] / total_users
    
    # If light group is empty or very small (< 10%), force balanced grouping
    if light_ratio < 0.10:
        logger.warning(f"Light group too small ({group_stats['light']} users, {light_ratio:.1%}). Using forced balanced grouping.")
        
        # Sort users by activity score for balanced splitting
        sorted_users = sorted(user_interaction_counts.items(), key=lambda x: x[1])
        
        # Create exactly 3 balanced groups
        n_users = len(sorted_users)
        light_end = n_users // 3
        medium_end = 2 * n_users // 3
        
        # Clear previous assignments
        user_activity_groups = {}
        group_stats = {"heavy": 0, "medium": 0, "light": 0}
        
        # Assign users to balanced groups
        for i, (user_id, activity_score) in enumerate(sorted_users):
            if i < light_end:
                group = "light"
            elif i < medium_end:
                group = "medium"
            else:
                group = "heavy"
            
            user_activity_groups[user_id] = group
            group_stats[group] += 1
        
        logger.info(f"  Forced balanced grouping applied")
        logger.info(f"  Light threshold: <= {sorted_users[light_end-1][1] if light_end > 0 else 0}")
        logger.info(f"  Medium threshold: <= {sorted_users[medium_end-1][1] if medium_end > 0 else 0}")
    
    # DEBUG: Track assignment details
    assignment_examples = {"heavy": [], "medium": [], "light": []}
    
    for user_id, group in list(user_activity_groups.items())[:15]:  # First 15 for examples
        activity_score = user_interaction_counts[user_id]
        if len(assignment_examples[group]) < 3:
            assignment_examples[group].append(f"user_{user_id}: {activity_score}")
    
    # DEBUG: Log assignment examples
    for group in ["heavy", "medium", "light"]:
        if assignment_examples[group]:
            logger.info(f"  {group.title()} group examples: {assignment_examples[group]}")
        else:
            logger.info(f"  {group.title()} group: No users assigned")
    
    # Step 1.4: Log group distribution for validation
    total_users = len(user_activity_groups)
    logger.info(f"User Activity Group Distribution (Forced Balanced - Ekstrand et al., 2018 methodology):")
    logger.info(f"  Heavy Users: {group_stats['heavy']} ({group_stats['heavy']/total_users:.1%})")
    logger.info(f"  Medium Users: {group_stats['medium']} ({group_stats['medium']/total_users:.1%})")
    logger.info(f"  Light Users: {group_stats['light']} ({group_stats['light']/total_users:.1%})")
    logger.info(f"  Balanced split ensures fair comparison across activity levels")
    
    return user_activity_groups


class SemanticCoverageAccumulator:
    """
    Coverage metrics accumulator using semantic IDs with tokenizer catalog
    Computes both item coverage and brand coverage using intersection with actual catalogs
    
    Item coverage: Uses semantic IDs directly from tokenizer.cached_ids
    Brand coverage: Uses brand mapping from modules/tokenizer/semids.py (map_to_category)
    """
    
    def __init__(self, tokenizer, ks: List[int] = [1, 5, 10]):
        self.tokenizer = tokenizer
        self.ks = sorted(ks)
        self.catalog_semantic_ids = self._get_catalog_semantic_ids()
        self.brand_catalog = self._get_brand_catalog()
        self.reset()
    
    def _get_catalog_semantic_ids(self):
        """Get unique semantic IDs from the tokenizer catalog"""
        if not hasattr(self.tokenizer, 'cached_ids') or self.tokenizer.cached_ids is None:
            raise ValueError("Tokenizer must have cached_ids to create semantic ID catalog")
        
        # Get semantic IDs from tokenizer (remove last column if it's padding)
        semantic_ids = self.tokenizer.cached_ids[:, :-1]
        
        # Convert to set of tuples for uniqueness
        unique_semantic_ids = set()
        for semantic_id_tensor in semantic_ids:
            semantic_id_tuple = tuple(semantic_id_tensor.tolist())
            unique_semantic_ids.add(semantic_id_tuple)
        
        return unique_semantic_ids
    
    def _get_brand_catalog(self):
        """Get unique brand IDs from the tokenizer's map_to_category
        
        Uses the brand mapping created in modules/tokenizer/semids.py during precompute_corpus_ids()
        """
        if not hasattr(self.tokenizer, 'map_to_category') or not self.tokenizer.map_to_category:
            logger.warning("Tokenizer has no map_to_category - brand coverage will be 0")
            return set()
        
        # Get all unique brand IDs from the mapping
        unique_brands = set(self.tokenizer.map_to_category.values())
        
        # Filter out invalid brand IDs (None, NaN, negative values)
        valid_brands = set()
        for brand_id in unique_brands:
            if (brand_id is not None and 
                not (isinstance(brand_id, float) and np.isnan(brand_id)) and 
                brand_id >= 0):
                valid_brands.add(brand_id)
        
        logger.info(f"Brand catalog created: {len(valid_brands)} unique brands from tokenizer mapping")
        logger.info(f"Filtered out {len(unique_brands) - len(valid_brands)} invalid brand IDs (NaN/None/negative)")
        return valid_brands
    
    def _get_brand_for_semantic_id(self, semantic_id_tuple):
        """Get brand ID for a semantic ID tuple, handling NaN values"""
        if not hasattr(self.tokenizer, 'map_to_category'):
            return None
        
        # Convert tuple to string format used in map_to_category
        semantic_id_str = str(list(semantic_id_tuple))
        brand_id = self.tokenizer.map_to_category.get(semantic_id_str)
        
        # Return None if brand_id is None, NaN, or negative
        if (brand_id is None or 
            (isinstance(brand_id, float) and np.isnan(brand_id)) or 
            brand_id < 0):
            return None
        
        return brand_id
    
    def reset(self):
        """Reset all collected data"""
        self.unique_recommended_semantic_ids = {k: set() for k in self.ks}
        self.unique_recommended_brands = {k: set() for k in self.ks}
        self.total_users_processed = 0
    
    def accumulate(self, actual: torch.Tensor, top_k: torch.Tensor, 
                   user_ids: torch.Tensor = None, **kwargs) -> None:
        """
        Accumulate coverage data from a batch using semantic IDs directly
        
        Args:
            actual: Ground truth semantic IDs [batch_size, sem_id_dim]  
            top_k: Top-k predicted semantic IDs [batch_size, k, sem_id_dim]
            user_ids: User IDs (not used for coverage but kept for compatibility)
            **kwargs: Additional arguments (ignored)
        """
        batch_size, max_k, sem_id_dim = top_k.shape
        self.total_users_processed += batch_size
        
        # Process each user in the batch
        for user_idx in range(batch_size):
            user_top_k_sem_ids = top_k[user_idx]  # [max_k, sem_id_dim]
            
            # Process each k value for coverage calculation
            for k in self.ks:
                if k <= max_k:
                    # Get top-k recommendations for this user
                    user_top_k = user_top_k_sem_ids[:k]  # [k, sem_id_dim]
                    
                    # Process each semantic ID
                    for semantic_id_tensor in user_top_k:
                        semantic_id_tuple = tuple(semantic_id_tensor[:-1].tolist())  # Remove last column to match catalog format
                        
                        # Add to semantic ID coverage
                        self.unique_recommended_semantic_ids[k].add(semantic_id_tuple)
                        
                        # Add to brand coverage
                        brand_id = self._get_brand_for_semantic_id(semantic_id_tuple)
                        if (brand_id is not None and 
                            not (isinstance(brand_id, float) and np.isnan(brand_id)) and 
                            brand_id in self.brand_catalog):
                            self.unique_recommended_brands[k].add(brand_id)
    
    def get_catalog_size(self) -> int:
        """Get total number of unique semantic IDs in tokenizer catalog"""
        return len(self.catalog_semantic_ids)
    
    def get_brand_catalog_size(self) -> int:
        """Get total number of unique brands in brand catalog"""
        return len(self.brand_catalog)
    
    def compute_coverage(self, k: int) -> float:
        """Compute semantic ID coverage for a specific k using intersection with catalog"""
        if k not in self.ks:
            return 0.0
        
        catalog_size = self.get_catalog_size()
        if catalog_size == 0:
            return 0.0
        
        # Coverage = intersection of recommended IDs with catalog / total catalog size
        predicted_ids = self.unique_recommended_semantic_ids[k]
        valid_predicted_ids = predicted_ids.intersection(self.catalog_semantic_ids)
        coverage = len(valid_predicted_ids) / catalog_size
        return coverage
    
    def compute_brand_coverage(self, k: int) -> float:
        """Compute brand coverage for a specific k"""
        if k not in self.ks:
            return 0.0
        
        brand_catalog_size = self.get_brand_catalog_size()
        if brand_catalog_size == 0:
            logger.warning("Brand catalog is empty - brand coverage will be 0")
            return 0.0
        
        # Brand coverage = unique recommended brands / total brands in catalog
        unique_brand_count = len(self.unique_recommended_brands[k])
        brand_coverage = unique_brand_count / brand_catalog_size
        
        logger.info(f"Brand coverage@{k}: {unique_brand_count}/{brand_catalog_size} = {brand_coverage:.4f}")
        return brand_coverage
    
    def reduce(self) -> Dict[str, float]:
        """Compute all coverage metrics"""
        results = {}
        for k in self.ks:
            # Semantic ID coverage using intersection with catalog
            coverage = self.compute_coverage(k)
            results[f'coverage@{k}'] = coverage
            
            # Brand coverage using tokenizer's brand mapping
            brand_coverage = self.compute_brand_coverage(k)
            results[f'brand_coverage@{k}'] = brand_coverage
        return results


class UserFairnessAccumulator:
    """
    User fairness accumulator implementing activity-based group fairness
    
    Implements statistical parity framework from Burke et al. (2018):
    "Multisided Fairness for Recommendation"
    
    Uses activity-based grouping from Ekstrand et al. (2018):
    "All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and Effectiveness"
    """
    
    def __init__(self, user_activity_groups: Dict[str, str], ks: List[int] = [1, 5, 10]):
        self.user_activity_groups = user_activity_groups
        self.ks = sorted(ks)
        self.groups = ["heavy", "medium", "light"]
        self.reset()
    
    def reset(self):
        """Reset accumulator for new evaluation"""
        # Per-group NDCG collections for statistical parity computation
        self.group_ndcg_scores = {
            group: {f"ndcg@{k}": [] for k in self.ks} 
            for group in self.groups
        }
        self.total_users_processed = 0
        self.users_without_group = 0
    
    def accumulate(self, actual: torch.Tensor, top_k: torch.Tensor, 
                   user_ids: torch.Tensor, tokenizer=None, **kwargs) -> None:
        """
        Accumulate per-user NDCG scores by activity group
        
        Implements per-user quality tracking for statistical parity
        as described in Ekstrand et al. (2018)
        
        Args:
            actual: Ground truth semantic IDs [batch_size, sem_id_dim]
            top_k: Top-k predicted semantic IDs [batch_size, max_k, sem_id_dim]
            user_ids: User IDs [batch_size]
            tokenizer: Tokenizer (not used but kept for compatibility)
            **kwargs: Additional arguments (ignored)
        """
        batch_size = actual.shape[0]
        self.total_users_processed += batch_size
        
        for user_idx in range(batch_size):
            user_id = str(user_ids[user_idx].item())  # Convert to string
            
            # Skip users not in activity group mapping
            if user_id not in self.user_activity_groups:
                self.users_without_group += 1
                continue
            
            user_group = self.user_activity_groups[user_id]
            user_actual = actual[user_idx]  # [sem_id_dim]
            user_top_k = top_k[user_idx]    # [max_k, sem_id_dim]
            
            # Compute NDCG@k for this user (using existing function)
            for k in self.ks:
                if k <= user_top_k.shape[0]:
                    user_ndcg_k = compute_ndcg_for_semantic_ids(
                        pred=user_top_k, 
                        actual=user_actual, 
                        k=k
                    )
                    self.group_ndcg_scores[user_group][f"ndcg@{k}"].append(user_ndcg_k)
    
    def reduce(self) -> Dict[str, float]:
        """
        Compute statistical parity metrics based on group NDCG distributions
        
        Implements fairness metrics from Burke et al. (2018) and 
        Ekstrand et al. (2018) frameworks
        
        Returns:
            Dict with fairness metrics following the pattern:
            - fairness/max_ndcg_gap@k: Maximum quality gap between groups
            - fairness/ndcg_variance@k: Variance in quality across groups  
            - fairness/relative_disparity@k: Relative disparity normalized by mean
            - group_performance/ndcg@k_{group}: Individual group performance
        """
        fairness_metrics = {}
        
        # Step 2.1: Compute group-level NDCG statistics
        group_mean_ndcg = {}
        for group in self.groups:
            group_mean_ndcg[group] = {}
            for k in self.ks:
                ndcg_scores = self.group_ndcg_scores[group][f"ndcg@{k}"]
                if len(ndcg_scores) > 0:
                    group_mean_ndcg[group][f"ndcg@{k}"] = np.mean(ndcg_scores)
                    # Log individual group performance
                    fairness_metrics[f"group_performance/ndcg@{k}_{group}"] = group_mean_ndcg[group][f"ndcg@{k}"]
                    fairness_metrics[f"group_size/{group}"] = len(ndcg_scores)
                else:
                    group_mean_ndcg[group][f"ndcg@{k}"] = 0.0
                    fairness_metrics[f"group_performance/ndcg@{k}_{group}"] = 0.0
                    fairness_metrics[f"group_size/{group}"] = 0
        
        # Step 2.2: Compute Statistical Parity Metrics (Burke et al., 2018)
        for k in self.ks:
            ndcg_values = [group_mean_ndcg[group][f"ndcg@{k}"] for group in self.groups]
            
            if any(val > 0 for val in ndcg_values):
                # Maximum quality gap (Ekstrand et al., 2018)
                max_gap = max(ndcg_values) - min(ndcg_values)
                fairness_metrics[f"fairness/max_ndcg_gap@{k}"] = max_gap
                
                # Quality variance across groups
                quality_variance = np.var(ndcg_values)
                fairness_metrics[f"fairness/ndcg_variance@{k}"] = quality_variance
                
                # Relative disparity (normalized by overall mean)
                overall_mean = np.mean([val for val in ndcg_values if val > 0])
                if overall_mean > 0:
                    relative_disparity = max_gap / overall_mean
                    fairness_metrics[f"fairness/relative_disparity@{k}"] = relative_disparity
                
                # Pairwise group gaps for detailed analysis
                heavy_vs_light = group_mean_ndcg["heavy"][f"ndcg@{k}"] - group_mean_ndcg["light"][f"ndcg@{k}"]
                fairness_metrics[f"fairness/heavy_vs_light_gap@{k}"] = heavy_vs_light
                
                heavy_vs_medium = group_mean_ndcg["heavy"][f"ndcg@{k}"] - group_mean_ndcg["medium"][f"ndcg@{k}"]
                fairness_metrics[f"fairness/heavy_vs_medium_gap@{k}"] = heavy_vs_medium
                
                medium_vs_light = group_mean_ndcg["medium"][f"ndcg@{k}"] - group_mean_ndcg["light"][f"ndcg@{k}"]
                fairness_metrics[f"fairness/medium_vs_light_gap@{k}"] = medium_vs_light
            else:
                # All groups have zero NDCG
                fairness_metrics[f"fairness/max_ndcg_gap@{k}"] = 0.0
                fairness_metrics[f"fairness/ndcg_variance@{k}"] = 0.0
                fairness_metrics[f"fairness/relative_disparity@{k}"] = 0.0
                fairness_metrics[f"fairness/heavy_vs_light_gap@{k}"] = 0.0
                fairness_metrics[f"fairness/heavy_vs_medium_gap@{k}"] = 0.0
                fairness_metrics[f"fairness/medium_vs_light_gap@{k}"] = 0.0
        
        # Step 2.3: Log coverage statistics
        if self.users_without_group > 0:
            coverage_rate = (self.total_users_processed - self.users_without_group) / self.total_users_processed
            fairness_metrics["fairness/user_group_coverage"] = coverage_rate
            logger.warning(f"User group coverage: {coverage_rate:.1%} ({self.users_without_group} users without group mapping)")
        else:
            fairness_metrics["fairness/user_group_coverage"] = 1.0
        
        return fairness_metrics