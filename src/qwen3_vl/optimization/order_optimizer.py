"""
Optimization Order Manager for Qwen3-VL Model
Optimizes the order of applying optimizations for maximum synergistic effect.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Callable, Optional
import itertools
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from enum import Enum
from dataclasses import dataclass

from src.qwen3_vl.optimization.unified_optimization_manager import OptimizationType, OptimizationManager
from src.qwen3_vl.optimization.interaction_handler import OptimizationInteractionHandler
from src.qwen3_vl.optimization.performance_validator import PerformanceMetrics, PerformanceValidator


class OptimizationOrderStrategy(Enum):
    """Strategies for determining optimization order"""
    INTERACTION_BASED = "interaction_based"
    PERFORMANCE_BASED = "performance_based"
    EMPIRICAL_SEARCH = "empirical_search"
    HYBRID = "hybrid"


@dataclass
class OrderEvaluationResult:
    """Result of evaluating an optimization order"""
    order: List[OptimizationType]
    performance_metrics: PerformanceMetrics
    synergy_score: float
    execution_time: float
    memory_usage: float


class OptimizationOrderOptimizer:
    """
    Optimizes the order of applying optimizations to maximize synergistic effects.
    Uses multiple strategies to determine the best order.
    """
    
    def __init__(self, 
                 optimization_manager: OptimizationManager,
                 interaction_handler: OptimizationInteractionHandler,
                 performance_validator: PerformanceValidator):
        self.optimization_manager = optimization_manager
        self.interaction_handler = interaction_handler
        self.performance_validator = performance_validator
        self.logger = logging.getLogger(__name__)
        
        # Cache for storing previously evaluated orders
        self.order_cache: Dict[Tuple[OptimizationType, ...], OrderEvaluationResult] = {}
    
    def optimize_order(
        self,
        strategy: OptimizationOrderStrategy = OptimizationOrderStrategy.HYBRID,
        active_optimizations: Optional[List[OptimizationType]] = None,
        model: Optional[nn.Module] = None,
        test_input: Optional[torch.Tensor] = None,
        max_evaluations: int = 20
    ) -> List[OptimizationType]:
        """
        Optimize the order of applying optimizations using the specified strategy.
        """
        if active_optimizations is None:
            active_optimizations = self.optimization_manager.get_active_optimizations()
        
        if len(active_optimizations) <= 1:
            return active_optimizations
        
        self.logger.info(f"Optimizing order for {len(active_optimizations)} optimizations using {strategy.value} strategy")
        
        if strategy == OptimizationOrderStrategy.INTERACTION_BASED:
            return self._optimize_order_interaction_based(active_optimizations)
        elif strategy == OptimizationOrderStrategy.PERFORMANCE_BASED:
            return self._optimize_order_performance_based(active_optimizations, model, test_input, max_evaluations)
        elif strategy == OptimizationOrderStrategy.EMPIRICAL_SEARCH:
            return self._optimize_order_empirical_search(active_optimizations, model, test_input, max_evaluations)
        elif strategy == OptimizationOrderStrategy.HYBRID:
            return self._optimize_order_hybrid(active_optimizations, model, test_input, max_evaluations)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _optimize_order_interaction_based(self, optimizations: List[OptimizationType]) -> List[OptimizationType]:
        """
        Optimize order based on interaction relationships between optimizations.
        Prioritizes optimizations with synergistic relationships.
        """
        self.logger.info("Optimizing order based on interaction relationships...")
        
        # Create a graph of synergistic relationships
        synergy_graph = {opt: [] for opt in optimizations}
        
        for i, opt1 in enumerate(optimizations):
            for opt2 in optimizations[i+1:]:
                interaction = self.interaction_handler.get_interaction(opt1, opt2)
                if interaction and interaction.interaction_type.value == 'synergistic':
                    # Add both directions since synergistic relationships are typically bidirectional
                    synergy_factor = interaction.synergy_factor
                    synergy_graph[opt1].append((opt2, synergy_factor))
                    synergy_graph[opt2].append((opt1, synergy_factor))
        
        # Use a greedy approach to order optimizations
        # Prioritize optimizations that have the most synergistic relationships
        ordered_optimizations = []
        remaining_optimizations = set(optimizations)
        
        # Start with the optimization that has the most synergistic connections
        while remaining_optimizations:
            best_opt = None
            best_score = -1
            
            for opt in remaining_optimizations:
                # Score based on number of synergistic connections and their strength
                score = sum(fac for _, fac in synergy_graph[opt])
                if score > best_score:
                    best_score = score
                    best_opt = opt
            
            if best_opt:
                ordered_optimizations.append(best_opt)
                remaining_optimizations.remove(best_opt)
            else:
                # If no best opt found, just take the first remaining one
                ordered_optimizations.append(remaining_optimizations.pop())
        
        self.logger.info(f"Interaction-based order determined: {[opt.value for opt in ordered_optimizations]}")
        return ordered_optimizations
    
    def _optimize_order_performance_based(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor,
        max_evaluations: int
    ) -> List[OptimizationType]:
        """
        Optimize order based on performance impact of different orders.
        """
        self.logger.info("Optimizing order based on performance impact...")
        
        # For large number of optimizations, we can't evaluate all permutations
        # So we use a greedy approach or sample a subset
        if len(optimizations) <= 4:  # Evaluate all permutations for small sets
            return self._evaluate_all_permutations(optimizations, model, test_input)
        else:
            # Use greedy approach for larger sets
            return self._greedy_performance_optimization(optimizations, model, test_input, max_evaluations)
    
    def _evaluate_all_permutations(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor
    ) -> List[OptimizationType]:
        """Evaluate all permutations of optimizations (for small sets only)."""
        best_order = optimizations
        best_performance = float('inf')  # Lower is better (faster execution)
        
        for perm in itertools.permutations(optimizations):
            perm_tuple = tuple(perm)
            if perm_tuple in self.order_cache:
                # Use cached result
                cached_result = self.order_cache[perm_tuple]
                performance = cached_result.performance_metrics.execution_time
            else:
                # Evaluate this permutation
                performance, metrics = self._evaluate_order_performance(list(perm), model, test_input)
                
                # Cache the result
                result = OrderEvaluationResult(
                    order=list(perm),
                    performance_metrics=metrics,
                    synergy_score=1.0,  # Placeholder
                    execution_time=metrics.execution_time,
                    memory_usage=metrics.peak_memory if metrics.peak_memory else 0
                )
                self.order_cache[perm_tuple] = result
            
            if performance < best_performance:
                best_performance = performance
                best_order = list(perm)
        
        self.logger.info(f"Best performance-based order: {[opt.value for opt in best_order]} (time: {best_performance:.4f}s)")
        return best_order
    
    def _greedy_performance_optimization(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor,
        max_evaluations: int
    ) -> List[OptimizationType]:
        """Use greedy approach to find good optimization order."""
        remaining_opts = optimizations[:]
        ordered_opts = []
        
        # Start with the optimization that performs best individually
        best_individual = self._find_best_individual_optimization(remaining_opts, model, test_input)
        if best_individual:
            ordered_opts.append(best_individual)
            remaining_opts.remove(best_individual)
        
        # Greedily add optimizations that provide the best incremental benefit
        evaluations = 0
        while remaining_opts and evaluations < max_evaluations:
            best_next_opt = None
            best_performance = float('inf')
            
            for opt in remaining_opts:
                candidate_order = ordered_opts + [opt]
                performance, _ = self._evaluate_order_performance(candidate_order, model, test_input)
                
                if performance < best_performance:
                    best_performance = performance
                    best_next_opt = opt
            
            if best_next_opt:
                ordered_opts.append(best_next_opt)
                remaining_opts.remove(best_next_opt)
                evaluations += 1
            else:
                # If no improvement found, just add the first remaining
                ordered_opts.append(remaining_opts.pop(0))
                evaluations += 1
        
        # Add any remaining optimizations
        ordered_opts.extend(remaining_opts)
        
        self.logger.info(f"Greedy performance-based order: {[opt.value for opt in ordered_opts]}")
        return ordered_opts
    
    def _find_best_individual_optimization(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor
    ) -> Optional[OptimizationType]:
        """Find the optimization that performs best when applied individually."""
        best_opt = None
        best_performance = float('inf')
        
        baseline_time = self._measure_baseline_performance(model, test_input)
        
        for opt in optimizations:
            # Temporarily enable only this optimization
            original_states = {o: self.optimization_manager.optimization_states[o] for o in self.optimization_manager.get_active_optimizations()}
            for o in self.optimization_manager.get_active_optimizations():
                self.optimization_manager.optimization_states[o] = (o == opt)
            
            try:
                opt_performance, _ = self._evaluate_order_performance([opt], model, test_input)
                # Calculate improvement over baseline
                improvement = baseline_time / opt_performance if opt_performance > 0 else float('inf')
                
                if improvement > (baseline_time / best_performance) if best_performance != float('inf') else True:
                    best_performance = opt_performance
                    best_opt = opt
            finally:
                # Restore original states
                for o, state in original_states.items():
                    self.optimization_manager.optimization_states[o] = state
        
        return best_opt
    
    def _evaluate_order_performance(
        self, 
        order: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor
    ) -> Tuple[float, PerformanceMetrics]:
        """Evaluate the performance of a specific optimization order."""
        # Temporarily set the optimization order
        original_states = {o: self.optimization_manager.optimization_states[o] for o in self.optimization_manager.get_active_optimizations()}
        
        # Enable only the optimizations in this order
        for opt in self.optimization_manager.get_active_optimizations():
            self.optimization_manager.optimization_states[opt] = opt in order
        
        try:
            # Benchmark the model with this specific order of optimizations
            metrics = self.performance_validator.benchmark_model_performance(
                model, test_input, num_runs=3, warmup_runs=1
            )
            return metrics.execution_time, metrics
        finally:
            # Restore original states
            for opt, state in original_states.items():
                self.optimization_manager.optimization_states[opt] = state
    
    def _measure_baseline_performance(self, model: nn.Module, test_input: torch.Tensor) -> float:
        """Measure baseline performance without optimizations."""
        # Temporarily disable all optimizations
        original_states = {o: self.optimization_manager.optimization_states[o] for o in self.optimization_manager.get_active_optimizations()}
        for opt in self.optimization_manager.get_active_optimizations():
            self.optimization_manager.optimization_states[opt] = False
        
        try:
            metrics = self.performance_validator.benchmark_model_performance(
                model, test_input, num_runs=3, warmup_runs=1
            )
            return metrics.execution_time
        finally:
            # Restore original states
            for opt, state in original_states.items():
                self.optimization_manager.optimization_states[opt] = state
    
    def _optimize_order_empirical_search(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor,
        max_evaluations: int
    ) -> List[OptimizationType]:
        """
        Use empirical search to find the best optimization order.
        Evaluates a sample of possible orders and selects the best one.
        """
        self.logger.info(f"Optimizing order using empirical search (max {max_evaluations} evaluations)...")
        
        # Generate random permutations to evaluate
        evaluated_orders = []
        best_order = optimizations[:]
        best_performance = float('inf')
        
        # Evaluate the default order first
        default_performance, _ = self._evaluate_order_performance(optimizations, model, test_input)
        if default_performance < best_performance:
            best_performance = default_performance
            best_order = optimizations[:]
        
        # Generate and evaluate random permutations
        evaluations = 1
        while evaluations < max_evaluations:
            # Create a random permutation
            random_order = optimizations[:]
            np.random.shuffle(random_order)
            
            # Skip if we've already evaluated this order
            if tuple(random_order) in self.order_cache:
                continue
            
            # Evaluate the random order
            performance, metrics = self._evaluate_order_performance(random_order, model, test_input)
            evaluated_orders.append((random_order, performance))
            
            # Update best if this is better
            if performance < best_performance:
                best_performance = performance
                best_order = random_order[:]
            
            evaluations += 1
        
        self.logger.info(f"Empirical search best order: {[opt.value for opt in best_order]} (time: {best_performance:.4f}s)")
        return best_order
    
    def _optimize_order_hybrid(
        self, 
        optimizations: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor,
        max_evaluations: int
    ) -> List[OptimizationType]:
        """
        Use a hybrid approach combining interaction-based and performance-based strategies.
        """
        self.logger.info("Optimizing order using hybrid approach...")
        
        # Start with interaction-based ordering as a good initial guess
        interaction_order = self._optimize_order_interaction_based(optimizations)
        
        # Fine-tune with performance-based evaluation
        if len(optimizations) <= 5:  # For small sets, evaluate all or most permutations
            return self._evaluate_all_permutations(optimizations, model, test_input)
        else:
            # For larger sets, use a local search around the interaction-based order
            return self._local_search_around_initial_order(interaction_order, model, test_input, max_evaluations)
    
    def _local_search_around_initial_order(
        self, 
        initial_order: List[OptimizationType], 
        model: nn.Module, 
        test_input: torch.Tensor,
        max_evaluations: int
    ) -> List[OptimizationType]:
        """Perform local search around an initial order."""
        current_order = initial_order[:]
        current_performance, _ = self._evaluate_order_performance(current_order, model, test_input)
        
        evaluations = 1
        improved = True
        
        while improved and evaluations < max_evaluations:
            improved = False
            
            # Try swapping adjacent elements
            for i in range(len(current_order) - 1):
                if evaluations >= max_evaluations:
                    break
                
                # Create neighbor by swapping adjacent elements
                neighbor_order = current_order[:]
                neighbor_order[i], neighbor_order[i+1] = neighbor_order[i+1], neighbor_order[i]
                
                neighbor_performance, _ = self._evaluate_order_performance(neighbor_order, model, test_input)
                evaluations += 1
                
                if neighbor_performance < current_performance:  # Better performance
                    current_order = neighbor_order
                    current_performance = neighbor_performance
                    improved = True
                    self.logger.debug(f"Improved to {current_performance:.4f}s with swap at position {i}")
            
            # Try swapping non-adjacent elements if no adjacent swap improved
            if not improved:
                for i in range(len(current_order)):
                    for j in range(i + 2, len(current_order)):
                        if evaluations >= max_evaluations:
                            break
                        
                        # Create neighbor by swapping non-adjacent elements
                        neighbor_order = current_order[:]
                        neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
                        
                        neighbor_performance, _ = self._evaluate_order_performance(neighbor_order, model, test_input)
                        evaluations += 1
                        
                        if neighbor_performance < current_performance:  # Better performance
                            current_order = neighbor_order
                            current_performance = neighbor_performance
                            improved = True
                            self.logger.debug(f"Improved to {current_performance:.4f}s with swap at positions {i}, {j}")
                            break
                    if improved:
                        break
        
        self.logger.info(f"Hybrid approach final order: {[opt.value for opt in current_order]} (time: {current_performance:.4f}s)")
        return current_order


class AdaptiveOptimizationOrderer:
    """
    Adapts optimization order based on runtime conditions and performance feedback.
    """
    
    def __init__(self, order_optimizer: OptimizationOrderOptimizer):
        self.order_optimizer = order_optimizer
        self.logger = logging.getLogger(__name__)
        self.performance_history: List[OrderEvaluationResult] = []
        self.context_weights: Dict[str, float] = {}
    
    def adapt_order_based_on_context(
        self,
        active_optimizations: List[OptimizationType],
        model: nn.Module,
        test_input: torch.Tensor,
        context: Dict[str, Any] = None
    ) -> List[OptimizationType]:
        """
        Adapt optimization order based on runtime context.
        """
        if context is None:
            context = {}
        
        # Determine the most appropriate strategy based on context
        strategy = self._select_strategy_based_on_context(context)
        
        # Optimize order using selected strategy
        optimized_order = self.order_optimizer.optimize_order(
            strategy=strategy,
            active_optimizations=active_optimizations,
            model=model,
            test_input=test_input
        )
        
        return optimized_order
    
    def _select_strategy_based_on_context(self, context: Dict[str, Any]) -> OptimizationOrderStrategy:
        """Select optimization strategy based on runtime context."""
        # Default to hybrid strategy
        strategy = OptimizationOrderStrategy.HYBRID
        
        # Adjust based on context
        if context.get('time_sensitive', False):
            # For time-sensitive applications, use interaction-based for speed
            strategy = OptimizationOrderStrategy.INTERACTION_BASED
        elif context.get('memory_constrained', False):
            # For memory-constrained environments, consider memory usage
            strategy = OptimizationOrderStrategy.HYBRID
        elif context.get('compute_abundant', False):
            # When compute is abundant, we can afford more empirical search
            strategy = OptimizationOrderStrategy.EMPIRICAL_SEARCH
        elif context.get('batch_size', 1) > 16:
            # For large batches, certain optimization orders might be better
            strategy = OptimizationOrderStrategy.HYBRID
        
        return strategy
    
    def learn_from_performance_feedback(
        self,
        order: List[OptimizationType],
        metrics: PerformanceMetrics,
        context: Dict[str, Any] = None
    ):
        """Learn from performance feedback to improve future order selection."""
        result = OrderEvaluationResult(
            order=order,
            performance_metrics=metrics,
            synergy_score=1.0,  # Would be calculated in a real implementation
            execution_time=metrics.execution_time,
            memory_usage=metrics.peak_memory if metrics.peak_memory else 0
        )
        
        self.performance_history.append(result)
        
        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        self.logger.info(f"Learned from performance feedback: order length {len(order)}, time {metrics.execution_time:.4f}s")


def create_optimization_order_optimizer(
    optimization_manager: OptimizationManager,
    interaction_handler: OptimizationInteractionHandler,
    performance_validator: PerformanceValidator
) -> OptimizationOrderOptimizer:
    """Create an optimization order optimizer instance."""
    return OptimizationOrderOptimizer(optimization_manager, interaction_handler, performance_validator)


def get_recommended_optimization_order(
    optimization_manager: OptimizationManager,
    interaction_handler: OptimizationInteractionHandler,
    performance_validator: PerformanceValidator,
    model: nn.Module,
    test_input: torch.Tensor
) -> List[OptimizationType]:
    """
    Get the recommended optimization order for maximum synergistic effect.
    """
    optimizer = create_optimization_order_optimizer(
        optimization_manager,
        interaction_handler,
        performance_validator
    )
    
    # Use hybrid approach for best balance of accuracy and efficiency
    recommended_order = optimizer.optimize_order(
        strategy=OptimizationOrderStrategy.HYBRID,
        model=model,
        test_input=test_input,
        max_evaluations=15  # Reasonable default for evaluation
    )
    
    return recommended_order