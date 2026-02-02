"""
Disk-Based Inference Pipeline System

This module implements a disk-based inference pipeline system where each stage
is executed separately with intermediate results saved to disk and retrieved as needed.
"""

import hashlib
import json
import logging
import os
import pickle
import queue
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PipelineStage:
    """
    Represents a single stage in the inference pipeline.
    """

    def __init__(
        self,
        name: str,
        function: Callable,
        input_keys: List[str],
        output_keys: List[str],
        checkpoint_dir: str = "./pipeline_checkpoints",
        cache_intermediates: bool = True,
    ):
        """
        Initialize a pipeline stage.

        Args:
            name: Name of the stage
            function: Function to execute for this stage
            input_keys: Keys of data required as input
            output_keys: Keys of data produced as output
            checkpoint_dir: Directory to store intermediate results
            cache_intermediates: Whether to cache intermediate results to disk
        """
        self.name = name
        self.function = function
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.checkpoint_dir = Path(checkpoint_dir) / name
        self.cache_intermediates = cache_intermediates

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline stage with the given inputs.

        Args:
            inputs: Dictionary of input data

        Returns:
            Dictionary of output data
        """
        # Validate inputs
        for key in self.input_keys:
            if key not in inputs:
                raise ValueError(
                    f"Missing required input key '{key}' for stage '{self.name}'"
                )

        # Execute the function
        try:
            outputs = self.function(
                **{k: v for k, v in inputs.items() if k in self.input_keys}
            )

            # Ensure outputs is a dictionary
            if not isinstance(outputs, dict):
                if len(self.output_keys) == 1:
                    outputs = {self.output_keys[0]: outputs}
                else:
                    raise ValueError(
                        f"Function output must be a dictionary or single value for stage '{self.name}'"
                    )

            # Validate outputs
            for key in self.output_keys:
                if key not in outputs:
                    raise ValueError(
                        f"Function did not produce required output key '{key}' for stage '{self.name}'"
                    )

            return outputs
        except Exception as e:
            logger.error(f"Error executing stage '{self.name}': {str(e)}")
            raise

    def save_intermediate_result(self, data: Any, stage_input_hash: str) -> str:
        """
        Save intermediate result to disk.

        Args:
            data: Data to save
            stage_input_hash: Hash of the input that produced this result

        Returns:
            Path to the saved file
        """
        if not self.cache_intermediates:
            return None

        filename = f"{stage_input_hash}.pkl"
        filepath = self.checkpoint_dir / filename

        try:
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
            logger.debug(
                f"Saved intermediate result for stage '{self.name}' to {filepath}"
            )
            return str(filepath)
        except Exception as e:
            logger.error(
                f"Failed to save intermediate result for stage '{self.name}': {str(e)}"
            )
            return None

    def load_intermediate_result(self, stage_input_hash: str) -> Optional[Any]:
        """
        Load intermediate result from disk.

        Args:
            stage_input_hash: Hash of the input that produced this result

        Returns:
            Loaded data or None if not found/could not load
        """
        if not self.cache_intermediates:
            return None

        filename = f"{stage_input_hash}.pkl"
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            logger.debug(
                f"Loaded intermediate result for stage '{self.name}' from {filepath}"
            )
            return data
        except Exception as e:
            logger.error(
                f"Failed to load intermediate result for stage '{self.name}': {str(e)}"
            )
            return None


class DiskBasedPipeline:
    """
    Disk-based inference pipeline system that manages pipeline stages and disk storage.
    """

    def __init__(
        self,
        stages: List[PipelineStage],
        checkpoint_dir: str = "./pipeline_checkpoints",
        max_concurrent_stages: int = 1,
        cleanup_after_completion: bool = True,
    ):
        """
        Initialize the disk-based pipeline.

        Args:
            stages: List of pipeline stages to execute
            checkpoint_dir: Directory to store intermediate results
            max_concurrent_stages: Maximum number of stages to run concurrently
            cleanup_after_completion: Whether to clean up intermediate files after completion
        """
        self.stages = stages
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_concurrent_stages = max_concurrent_stages
        self.cleanup_after_completion = cleanup_after_completion
        self.stage_results = {}  # Cache for stage results
        self._lock = threading.Lock()

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _compute_input_hash(self, inputs: Dict[str, Any]) -> str:
        """
        Compute a hash of the input data for caching purposes.

        Args:
            inputs: Input data dictionary

        Returns:
            SHA256 hash of the input data
        """
        # Convert inputs to a consistent format for hashing
        # Handle special objects like tensors and numpy arrays
        serializable_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                # For tensors, use their shape and a hash of their values
                tensor_data = {
                    "shape": v.shape,
                    "dtype": str(v.dtype),
                    "hash": hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest(),
                }
                serializable_inputs[k] = tensor_data
            elif isinstance(v, np.ndarray):
                # For numpy arrays, use their shape and a hash of their values
                array_data = {
                    "shape": v.shape,
                    "dtype": str(v.dtype),
                    "hash": hashlib.sha256(v.tobytes()).hexdigest(),
                }
                serializable_inputs[k] = array_data
            else:
                # For other types, try to serialize them
                try:
                    # Attempt to serialize to JSON first for consistency
                    json.dumps(v, default=str)
                    serializable_inputs[k] = v
                except TypeError:
                    # If not JSON serializable, convert to string representation
                    serializable_inputs[k] = str(v)

        # Serialize the entire dictionary and compute hash
        serialized = json.dumps(serializable_inputs, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def execute_pipeline(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the entire pipeline with the given initial inputs.

        Args:
            initial_inputs: Initial input data for the pipeline

        Returns:
            Final output of the pipeline
        """
        logger.info(f"Starting pipeline execution with {len(self.stages)} stages")

        # Combine initial inputs with previously computed results
        pipeline_data = initial_inputs.copy()

        # Execute each stage in sequence
        for stage in self.stages:
            logger.info(f"Executing stage: {stage.name}")

            # Prepare inputs for this stage
            stage_inputs = {}
            for key in stage.input_keys:
                if key in pipeline_data:
                    stage_inputs[key] = pipeline_data[key]
                else:
                    raise ValueError(
                        f"Missing required input '{key}' for stage '{stage.name}'"
                    )

            # Compute hash of inputs for caching
            input_hash = self._compute_input_hash(stage_inputs)

            # Check if result is already cached
            cached_result = stage.load_intermediate_result(input_hash)
            if cached_result is not None:
                logger.info(f"Using cached result for stage '{stage.name}'")
                stage_outputs = cached_result
            else:
                # Execute the stage
                stage_outputs = stage.execute(stage_inputs)

                # Save result to disk for future use
                stage.save_intermediate_result(stage_outputs, input_hash)

            # Update pipeline data with stage outputs
            for key, value in stage_outputs.items():
                pipeline_data[key] = value

            logger.info(f"Completed stage: {stage.name}")

        logger.info("Pipeline execution completed")

        # Return the final outputs
        final_outputs = {}
        for stage in self.stages:
            for key in stage.output_keys:
                if key in pipeline_data:
                    final_outputs[key] = pipeline_data[key]

        return final_outputs

    def execute_pipeline_async(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline asynchronously with potential for parallel stage execution.

        Args:
            initial_inputs: Initial input data for the pipeline

        Returns:
            Final output of the pipeline
        """
        logger.info(f"Starting async pipeline execution with {len(self.stages)} stages")

        # For now, execute sequentially since stages depend on each other
        # In a more complex system, we could identify independent stages
        return self.execute_pipeline(initial_inputs)

    def cleanup_checkpoints(self):
        """
        Clean up all checkpoint files created by this pipeline.
        """
        import shutil

        try:
            if self.checkpoint_dir.exists():
                shutil.rmtree(self.checkpoint_dir)
                logger.info(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up checkpoints: {str(e)}")

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline execution.

        Returns:
            Dictionary containing pipeline statistics
        """
        stats = {
            "num_stages": len(self.stages),
            "checkpoint_dir": str(self.checkpoint_dir),
            "stages": [],
        }

        for stage in self.stages:
            stage_stats = {
                "name": stage.name,
                "input_keys": stage.input_keys,
                "output_keys": stage.output_keys,
                "checkpoint_dir": str(stage.checkpoint_dir),
            }
            stats["stages"].append(stage_stats)

        return stats


class PipelineManager:
    """
    Manager class to handle multiple disk-based pipelines.
    """

    def __init__(self, base_checkpoint_dir: str = "./pipeline_checkpoints"):
        """
        Initialize the pipeline manager.

        Args:
            base_checkpoint_dir: Base directory for all pipeline checkpoints
        """
        self.base_checkpoint_dir = Path(base_checkpoint_dir)
        self.pipelines = {}
        self.active_executions = {}

        # Create base directory if it doesn't exist
        self.base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def register_pipeline(self, name: str, pipeline: DiskBasedPipeline):
        """
        Register a pipeline with the manager.

        Args:
            name: Name to register the pipeline under
            pipeline: Pipeline instance to register
        """
        self.pipelines[name] = pipeline

    def execute_pipeline(
        self, name: str, initial_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a registered pipeline.

        Args:
            name: Name of the pipeline to execute
            initial_inputs: Initial input data for the pipeline

        Returns:
            Final output of the pipeline
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not registered")

        pipeline = self.pipelines[name]
        return pipeline.execute_pipeline(initial_inputs)

    def execute_pipeline_async(
        self, name: str, initial_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a registered pipeline asynchronously.

        Args:
            name: Name of the pipeline to execute
            initial_inputs: Initial input data for the pipeline

        Returns:
            Final output of the pipeline
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not registered")

        pipeline = self.pipelines[name]
        return pipeline.execute_pipeline_async(initial_inputs)

    def get_pipeline_stats(self, name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific pipeline.

        Args:
            name: Name of the pipeline

        Returns:
            Dictionary containing pipeline statistics
        """
        if name not in self.pipelines:
            raise ValueError(f"Pipeline '{name}' not registered")

        pipeline = self.pipelines[name]
        return pipeline.get_pipeline_stats()

    def cleanup_pipeline(self, name: str):
        """
        Clean up checkpoints for a specific pipeline.

        Args:
            name: Name of the pipeline to clean up
        """
        if name in self.pipelines:
            self.pipelines[name].cleanup_checkpoints()

    def cleanup_all(self):
        """
        Clean up all pipeline checkpoints.
        """
        for name in self.pipelines:
            self.cleanup_pipeline(name)


# Utility functions for creating common pipeline stages
def create_tokenization_stage(
    tokenizer_func: Callable, checkpoint_dir: str = "./pipeline_checkpoints"
) -> PipelineStage:
    """
    Create a tokenization stage for the pipeline.

    Args:
        tokenizer_func: Function that takes text and returns tokens
        checkpoint_dir: Directory to store intermediate results

    Returns:
        PipelineStage for tokenization
    """

    def tokenize_stage(text: str) -> Dict[str, Any]:
        tokens = tokenizer_func(text)
        return {"tokens": tokens}

    return PipelineStage(
        name="tokenization",
        function=tokenize_stage,
        input_keys=["text"],
        output_keys=["tokens"],
        checkpoint_dir=checkpoint_dir,
    )


def create_model_inference_stage(
    model_func: Callable, checkpoint_dir: str = "./pipeline_checks"
) -> PipelineStage:
    """
    Create a model inference stage for the pipeline.

    Args:
        model_func: Function that takes tokens and returns model outputs
        checkpoint_dir: Directory to store intermediate results

    Returns:
        PipelineStage for model inference
    """

    def inference_stage(tokens: Any) -> Dict[str, Any]:
        outputs = model_func(tokens)
        return {"model_outputs": outputs}

    return PipelineStage(
        name="model_inference",
        function=inference_stage,
        input_keys=["tokens"],
        output_keys=["model_outputs"],
        checkpoint_dir=checkpoint_dir,
    )


def create_decoding_stage(
    decoder_func: Callable, checkpoint_dir: str = "./pipeline_checks"
) -> PipelineStage:
    """
    Create a decoding stage for the pipeline.

    Args:
        decoder_func: Function that takes model outputs and returns text
        checkpoint_dir: Directory to store intermediate results

    Returns:
        PipelineStage for decoding
    """

    def decode_stage(model_outputs: Any) -> Dict[str, Any]:
        decoded_text = decoder_func(model_outputs)
        return {"decoded_text": decoded_text}

    return PipelineStage(
        name="decoding",
        function=decode_stage,
        input_keys=["model_outputs"],
        output_keys=["decoded_text"],
        checkpoint_dir=checkpoint_dir,
    )


__all__ = [
    "PipelineStage",
    "DiskBasedPipeline",
    "PipelineManager",
    "create_tokenization_stage",
    "create_model_inference_stage",
    "create_decoding_stage",
]
