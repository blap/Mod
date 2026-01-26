"""
Intelligent Tensor Pagination System for Multimodal Data

This module implements an intelligent pagination system optimized for multimodal data
(text, image, video, audio) with different memory access patterns and requirements.
"""

import logging
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class DataType(Enum):
    """Enumeration of different data types for multimodal pagination."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    ACTIVATIONS = "activations"
    KV_CACHE = "kv_cache"


class PaginationPriority(Enum):
    """Priority levels for tensor pagination."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TensorPage:
    """Represents a memory page for tensor pagination."""
    id: str
    tensor: Optional[Tensor] = None
    data_type: Optional[DataType] = None
    priority: PaginationPriority = PaginationPriority.MEDIUM
    size_bytes: int = 0
    device: Optional[torch.device] = None
    file_path: Optional[str] = None
    last_access_time: float = 0.0
    creation_time: float = 0.0
    pinned: bool = False
    access_pattern: str = "unknown"
    layer_index: Optional[int] = None
    sequence_position: Optional[int] = None
    temporal_position: Optional[int] = None  # For video/audio data
    spatial_position: Optional[Tuple[int, int]] = None  # For image data
    access_frequency: float = 0.0
    temporal_locality: float = 0.0  # How recently accessed
    reuse_probability: float = 0.0
    offload_history: List[float] = None  # Times when this page was offloaded
    restore_history: List[float] = None  # Times when this page was restored

    def __post_init__(self):
        if self.offload_history is None:
            self.offload_history = []
        if self.restore_history is None:
            self.restore_history = []
        if self.last_access_time == 0.0:
            self.last_access_time = time.time()
        if self.creation_time == 0.0:
            self.creation_time = time.time()


class AccessPatternAnalyzer:
    """Analyze access patterns to predict future page accesses."""
    
    def __init__(self):
        self.access_history = deque(maxlen=1000)  # Keep last 1000 access events
        self.page_frequency = defaultdict(int)
        self.page_recency = {}  # Maps page_id to last access time
        self.access_intervals = defaultdict(deque)  # Maps page_id to access intervals
        self.access_patterns = defaultdict(lambda: "unknown")  # Maps page_id to access pattern
        self.lock = threading.Lock()

    def record_access(self, page_id: str, timestamp: float, access_pattern: str = "unknown"):
        """Record a page access event."""
        with self.lock:
            self.access_history.append((timestamp, page_id))
            self.page_frequency[page_id] += 1
            self.page_recency[page_id] = timestamp
            self.access_patterns[page_id] = access_pattern

            # Calculate access intervals for frequently accessed pages
            if page_id in self.access_intervals:
                prev_access = self.access_intervals[page_id][-1] if self.access_intervals[page_id] else None
                if prev_access is not None:
                    interval = timestamp - prev_access
                    self.access_intervals[page_id].append(interval)
                    if len(self.access_intervals[page_id]) > 10:  # Keep last 10 intervals
                        self.access_intervals[page_id].popleft()
                else:
                    self.access_intervals[page_id] = deque([timestamp], maxlen=10)
            else:
                self.access_intervals[page_id] = deque([timestamp], maxlen=10)

    def predict_next_access(self, page_id: str, current_time: float) -> float:
        """Predict when a page will be accessed next based on historical patterns."""
        if page_id not in self.access_intervals or not self.access_intervals[page_id]:
            return current_time + 10.0  # Default prediction: 10 seconds

        intervals = list(self.access_intervals[page_id])
        avg_interval = sum(intervals) / len(intervals) if intervals else 1.0
        
        last_access = self.page_recency.get(page_id, current_time)
        predicted_next = last_access + avg_interval
        
        return predicted_next

    def get_access_score(self, page_id: str, current_time: float) -> float:
        """Calculate a score representing how soon a page will be accessed."""
        with self.lock:
            # Frequency affects score (more frequent = less likely to be evicted)
            freq_score = min(self.page_frequency[page_id] / 10.0, 1.0)  # Normalize
            
            # Recency affects score (recently accessed pages are more likely to be accessed again)
            last_access = self.page_recency.get(page_id, 0)
            recency_score = max(0.0, (current_time - last_access) / 10.0)  # Normalize by 10 seconds
            
            # Predicted access time affects score (sooner = lower score = less likely to be evicted)
            predicted_next = self.predict_next_access(page_id, current_time)
            time_score = max(0.0, (predicted_next - current_time) / 5.0)  # Normalize by 5 seconds
            
            # Combine scores (higher score = more likely to be kept in memory)
            combined_score = freq_score * 0.4 + recency_score * 0.3 + time_score * 0.3
            
            return combined_score

    def get_access_frequency(self, page_id: str, current_time: float) -> float:
        """Get the access frequency for a page."""
        with self.lock:
            if page_id not in self.page_frequency:
                return 0.0
            
            # Calculate frequency in the last minute
            recent_accesses = sum(1 for t, pid in self.access_history 
                                if pid == page_id and current_time - t <= 60)
            
            return recent_accesses / 60.0  # Per second


class TensorPaginationSystem:
    """Intelligent tensor pagination system optimized for multimodal data."""
    
    def __init__(
        self,
        swap_directory: Optional[Union[str, Path]] = None,
        page_size_mb: int = 16,
        eviction_policy: str = "intelligent",  # Options: lru, fifo, priority, intelligent
        max_memory_ratio: float = 0.8,
        enable_clustering: bool = True,
        cluster_count: int = 5,
        enable_adaptive: bool = True
    ):
        self.swap_directory = Path(swap_directory) if swap_directory else Path("./tensor_swap")
        self.swap_directory.mkdir(exist_ok=True)
        
        self.page_size_bytes = page_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.max_memory_ratio = max_memory_ratio
        self.enable_clustering = enable_clustering
        self.cluster_count = cluster_count
        self.enable_adaptive = enable_adaptive
        
        # Track memory pages
        self.pages: Dict[str, TensorPage] = {}
        self.ram_pages: List[str] = []  # Pages currently in RAM
        self.disk_pages: List[str] = []  # Pages currently on disk
        self.access_times: Dict[str, float] = {}  # Last access time for each page
        
        # Statistics
        self.stats = {
            'pages_swapped_in': 0,
            'pages_swapped_out': 0,
            'page_faults': 0,
            'total_pages': 0,
            'ram_pages': 0,
            'disk_pages': 0,
            'total_swapped_bytes': 0,
            'total_restored_bytes': 0
        }
        
        # Access pattern analyzer
        self.access_analyzer = AccessPatternAnalyzer()
        
        # Clustering model for similar pages
        self.clustering_model = None
        if self.enable_clustering:
            try:
                from sklearn.cluster import KMeans
                self.KMeans = KMeans
            except ImportError:
                logger.warning("sklearn not available, clustering disabled")
                self.enable_clustering = False
        
        # Adaptive management thread
        self.management_thread = None
        self.management_stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized TensorPaginationSystem with swap_directory={self.swap_directory}, "
                   f"page_size={page_size_mb}MB, eviction_policy={eviction_policy}, "
                   f"max_memory_ratio={max_memory_ratio}, clustering={enable_clustering}")

    def _get_current_memory_usage_ratio(self) -> float:
        """Get the current memory usage ratio."""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            max_memory = torch.cuda.max_memory_allocated()
            if max_memory == 0:
                max_memory = torch.cuda.get_device_properties(0).total_memory * self.max_memory_ratio
            return current_memory / max_memory if max_memory > 0 else 0.0
        else:
            # For CPU, estimate based on system memory
            import psutil
            return psutil.virtual_memory().percent / 100.0

    def allocate_page(
        self, 
        tensor: Tensor, 
        page_id: str, 
        data_type: DataType,
        priority: PaginationPriority = PaginationPriority.MEDIUM,
        access_pattern: str = "unknown",
        layer_index: Optional[int] = None,
        sequence_position: Optional[int] = None,
        temporal_position: Optional[int] = None,
        spatial_position: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Allocate a memory page for a tensor."""
        current_time = time.time()
        
        if page_id in self.pages:
            logger.warning(f"Page {page_id} already exists, overwriting")
            self.deallocate_page(page_id)
        
        # Calculate tensor size
        size_bytes = tensor.element_size() * tensor.nelement()
        
        # Create page object
        page = TensorPage(
            id=page_id,
            tensor=tensor,
            data_type=data_type,
            priority=priority,
            size_bytes=size_bytes,
            device=tensor.device,
            last_access_time=current_time,
            creation_time=current_time,
            access_pattern=access_pattern,
            layer_index=layer_index,
            sequence_position=sequence_position,
            temporal_position=temporal_position,
            spatial_position=spatial_position
        )
        
        self.pages[page_id] = page
        self.ram_pages.append(page_id)
        self.access_times[page_id] = page.last_access_time
        
        self.stats['total_pages'] += 1
        self.stats['ram_pages'] += 1
        
        # Record access pattern
        self.access_analyzer.record_access(page_id, current_time, access_pattern)
        
        # Check if we need to evict pages due to memory pressure
        self._handle_memory_pressure()
        
        logger.debug(f"Allocated page {page_id} ({size_bytes} bytes) in RAM, data_type={data_type.value}")
        return True

    def deallocate_page(self, page_id: str) -> bool:
        """Deallocate a memory page."""
        if page_id not in self.pages:
            logger.warning(f"Page {page_id} does not exist")
            return False
        
        page = self.pages[page_id]
        
        # Remove from appropriate list
        if page_id in self.ram_pages:
            self.ram_pages.remove(page_id)
            self.stats['ram_pages'] -= 1
        elif page_id in self.disk_pages:
            self.disk_pages.remove(page_id)
            self.stats['disk_pages'] -= 1
        
        # Delete offload file if exists
        if page.file_path and os.path.exists(page.file_path):
            try:
                os.remove(page.file_path)
            except OSError as e:
                logger.error(f"Failed to delete swap file {page.file_path}: {e}")
        
        # Remove from access times
        if page_id in self.access_times:
            del self.access_times[page_id]
        
        # Delete the page
        del self.pages[page_id]
        
        logger.debug(f"Deallocated page {page_id}")
        return True

    def swap_page_to_disk(self, page_id: str) -> bool:
        """Swap a page from RAM to disk."""
        if page_id not in self.pages:
            logger.error(f"Page {page_id} does not exist")
            return False
        
        page = self.pages[page_id]
        
        if page_id not in self.ram_pages:
            logger.warning(f"Page {page_id} is not in RAM, nothing to swap")
            return False
        
        # Create swap file
        swap_file = self.swap_directory / f"page_{page_id}.pkl"
        
        try:
            # Move tensor to CPU and save
            original_device = page.tensor.device
            with open(swap_file, 'wb') as f:
                pickle.dump({
                    'tensor': page.tensor.cpu() if page.tensor is not None else None,
                    'device': original_device
                }, f)
            
            # Update page info
            page.file_path = str(swap_file)
            page.tensor = None  # Free RAM
            page.device = None
            
            # Update lists
            self.ram_pages.remove(page_id)
            self.disk_pages.append(page_id)
            
            # Update stats
            self.stats['pages_swapped_out'] += 1
            self.stats['ram_pages'] -= 1
            self.stats['disk_pages'] += 1
            self.stats['total_swapped_bytes'] += page.size_bytes
            
            # Record offload time
            page.offload_history.append(time.time())
            
            logger.debug(f"Swapped page {page_id} to disk: {swap_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to swap page {page_id} to disk: {e}")
            return False

    def swap_page_to_ram(self, page_id: str) -> bool:
        """Swap a page from disk to RAM."""
        if page_id not in self.pages:
            logger.error(f"Page {page_id} does not exist")
            return False
        
        page = self.pages[page_id]
        
        if page_id not in self.disk_pages:
            logger.warning(f"Page {page_id} is not on disk, nothing to swap")
            return False
        
        if not page.file_path or not os.path.exists(page.file_path):
            logger.error(f"Swap file for page {page_id} does not exist: {page.file_path}")
            return False
        
        try:
            # Load tensor from disk
            with open(page.file_path, 'rb') as f:
                data = pickle.load(f)
                tensor = data['tensor']
                original_device = data['device']
            
            # Move tensor to original device
            if original_device.type == 'cuda' and torch.cuda.is_available():
                tensor = tensor.to(original_device)
            elif original_device.type == 'mps' and torch.backends.mps.is_available():
                tensor = tensor.to(original_device)
            
            # Update page info
            page.tensor = tensor
            page.device = original_device
            
            # Update lists
            self.disk_pages.remove(page_id)
            self.ram_pages.append(page_id)
            
            # Update stats
            self.stats['pages_swapped_in'] += 1
            self.stats['ram_pages'] += 1
            self.stats['disk_pages'] -= 1
            self.stats['total_restored_bytes'] += page.size_bytes
            
            # Record restore time
            page.restore_history.append(time.time())
            
            logger.debug(f"Swapped page {page_id} to RAM from: {page.file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to swap page {page_id} to RAM: {e}")
            return False

    def access_page(self, page_id: str) -> Optional[Tensor]:
        """Access a page, ensuring it's in RAM. This may trigger swaps."""
        if page_id not in self.pages:
            logger.error(f"Page {page_id} does not exist")
            return None
        
        page = self.pages[page_id]
        
        current_time = time.time()
        page.last_access_time = current_time
        self.access_times[page_id] = page.last_access_time
        
        # Record access pattern
        self.access_analyzer.record_access(page_id, current_time, page.access_pattern)
        
        # If page is on disk, swap it to RAM
        if page_id in self.disk_pages:
            self.stats['page_faults'] += 1
            if not self.swap_page_to_ram(page_id):
                logger.error(f"Failed to swap page {page_id} to RAM")
                return None
        
        return page.tensor

    def _handle_memory_pressure(self):
        """Handle memory pressure by swapping out pages if needed."""
        current_memory_ratio = self._get_current_memory_usage_ratio()
        
        if current_memory_ratio < self.max_memory_ratio:
            return  # No memory pressure
        
        # Get pages to consider for swapping
        pages_to_consider = []
        for page_id in self.ram_pages:
            page = self.pages[page_id]
            # Don't swap pinned pages
            if not page.pinned:
                pages_to_consider.append((page_id, page))
        
        if not pages_to_consider:
            logger.warning("Memory pressure detected but no swappable pages available")
            return
        
        # Sort pages based on the selected strategy
        if self.eviction_policy == "lru":
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "fifo":
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        elif self.eviction_policy == "priority":
            # Evict lowest priority pages first
            pages_to_consider.sort(key=lambda x: x[1].priority.value)
        elif self.eviction_policy == "intelligent":
            # Use predictive algorithm to determine which pages to evict
            pages_to_consider.sort(key=lambda x: self._calculate_intelligent_score(x[0], time.time()))
        else:
            # Default to LRU
            pages_to_consider.sort(key=lambda x: self.access_times[x[0]])
        
        # Swap out pages until memory pressure is relieved
        max_swaps_per_call = min(len(pages_to_consider), 10)  # Limit to 10 swaps per call
        
        for i, (page_id, page) in enumerate(pages_to_consider):
            if i >= max_swaps_per_call:
                break
                
            # Don't swap pinned pages
            if page.pinned:
                continue
                
            current_memory_ratio = self._get_current_memory_usage_ratio()
            if current_memory_ratio < self.max_memory_ratio * 0.9:  # Leave some buffer
                break
                
            if self.swap_page_to_disk(page_id):
                logger.debug(f"Evicted page {page_id} due to memory pressure using {self.eviction_policy} strategy")
            else:
                logger.warning(f"Failed to evict page {page_id} due to memory pressure")

    def _calculate_intelligent_score(self, page_id: str, current_time: float) -> float:
        """Calculate an intelligent score for page eviction based on multiple factors."""
        page = self.pages[page_id]
        
        # Get access score from analyzer
        access_score = self.access_analyzer.get_access_score(page_id, current_time)
        
        # Priority factor (higher priority = less likely to be evicted)
        priority_factor = page.priority.value  # Higher priority = less likely to be evicted
        
        # Size factor (larger pages = more likely to be evicted)
        size_factor = page.size_bytes / (1024 * 1024)  # Size in MB, larger = more likely to be evicted
        
        # Data type factor (some types are more critical than others)
        type_factor = {
            DataType.KV_CACHE: 0.8,  # KV cache is important but can be evicted
            DataType.ACTIVATIONS: 0.6,  # Activations can be recomputed
            DataType.TEXT: 0.7,  # Text embeddings somewhat important
            DataType.IMAGE: 0.5,  # Image features can be reprocessed
            DataType.VIDEO: 0.4,  # Video features can be reprocessed
            DataType.AUDIO: 0.5,  # Audio features can be reprocessed
            DataType.EMBEDDINGS: 0.7  # Embeddings are important
        }.get(page.data_type, 0.6)
        
        # Temporal locality (how recently accessed)
        time_since_access = current_time - page.last_access_time
        temporal_factor = min(time_since_access / 10.0, 1.0)  # Normalize to 0-1
        
        # Combine factors (lower score = more likely to be evicted)
        score = (access_score * 0.3 + 
                (5 - priority_factor) * 0.2 +  # Invert priority (lower priority = higher score)
                size_factor * 0.2 +
                (1 - type_factor) * 0.2 +
                temporal_factor * 0.1)
        
        return score

    def get_page_stats(self) -> Dict[str, Any]:
        """Get statistics about memory pages."""
        total_size = sum(page.size_bytes for page in self.pages.values())
        ram_size = sum(self.pages[pid].size_bytes for pid in self.ram_pages)
        disk_size = sum(self.pages[pid].size_bytes for pid in self.disk_pages)
        
        return {
            'total_pages': len(self.pages),
            'ram_pages': len(self.ram_pages),
            'disk_pages': len(self.disk_pages),
            'total_size_bytes': total_size,
            'ram_size_bytes': ram_size,
            'disk_size_bytes': disk_size,
            'stats': self.stats.copy()
        }

    def start_proactive_management(self, interval: float = 5.0):
        """Start proactive memory management in a background thread."""
        if self.management_thread is not None and self.management_thread.is_alive():
            logger.warning("Proactive management already running")
            return
        
        self.management_stop_event.clear()
        self.management_thread = threading.Thread(
            target=self._proactive_management_loop,
            args=(interval,),
            daemon=True
        )
        self.management_thread.start()
        logger.info(f"Started proactive memory management with interval {interval}s")

    def stop_proactive_management(self):
        """Stop proactive memory management."""
        if self.management_thread is not None:
            self.management_stop_event.set()
            self.management_thread.join(timeout=2.0)
            logger.info("Stopped proactive memory management")

    def _proactive_management_loop(self, interval: float):
        """Background loop for proactive memory management."""
        while not self.management_stop_event.is_set():
            try:
                # Wait for interval or stop event
                if self.management_stop_event.wait(timeout=interval):
                    break  # Stop event was set
                
                # Perform proactive management
                self._perform_proactive_management()
                
            except Exception as e:
                logger.error(f"Error in proactive management loop: {e}")

    def _perform_proactive_management(self):
        """Perform proactive memory management."""
        current_time = time.time()
        
        # Identify pages that are unlikely to be accessed soon
        pages_to_swap = []
        for page_id in self.ram_pages:
            if not self.pages[page_id].pinned:  # Don't swap pinned pages
                access_score = self.access_analyzer.get_access_score(page_id, current_time)
                
                # If access score indicates low likelihood of near-future access
                if access_score < 0.2:  # Threshold for proactive swapping
                    pages_to_swap.append((page_id, access_score))
        
        # Sort by access score (lowest first)
        pages_to_swap.sort(key=lambda x: x[1])
        
        # Swap out pages until we reach a safe memory level
        current_memory_ratio = self._get_current_memory_usage_ratio()
        target_ratio = self.max_memory_ratio * 0.8  # Target 80% of max memory
        
        for page_id, access_score in pages_to_swap:
            if current_memory_ratio < target_ratio:
                break
                
            if self.swap_page_to_disk(page_id):
                logger.debug(f"Proactively swapped page {page_id} (access_score: {access_score:.2f})")
                current_memory_ratio = self._get_current_memory_usage_ratio()
            else:
                logger.warning(f"Failed to proactively swap page {page_id}")

    def cleanup(self):
        """Clean up resources."""
        self.stop_proactive_management()
        
        # Deallocate all pages
        pages_to_delete = list(self.pages.keys())
        for page_id in pages_to_delete:
            self.deallocate_page(page_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)


class MultimodalTensorPager:
    """Specialized pager for multimodal data with different optimization strategies."""
    
    def __init__(self, pagination_system: TensorPaginationSystem):
        self.pagination_system = pagination_system
        self.tensor_mappings: Dict[str, str] = {}  # tensor_id -> page_id
        self.page_mappings: Dict[str, str] = {}    # page_id -> tensor_id
    
    def page_tensor(
        self,
        tensor: Tensor,
        tensor_id: str,
        data_type: DataType,
        priority: PaginationPriority = PaginationPriority.MEDIUM,
        access_pattern: str = "unknown",
        layer_index: Optional[int] = None,
        sequence_position: Optional[int] = None,
        temporal_position: Optional[int] = None,
        spatial_position: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Page a tensor to the pagination system."""
        page_id = f"mm_tensor_{data_type.value}_{tensor_id}_{id(tensor)}"
        
        success = self.pagination_system.allocate_page(
            tensor, 
            page_id, 
            data_type, 
            priority, 
            access_pattern, 
            layer_index, 
            sequence_position, 
            temporal_position, 
            spatial_position
        )
        
        if success:
            self.tensor_mappings[tensor_id] = page_id
            self.page_mappings[page_id] = tensor_id
            
            # Update the page's strategy based on data type
            if page_id in self.pagination_system.pages:
                page = self.pagination_system.pages[page_id]
                page.access_frequency = self._get_data_type_frequency(data_type)
        
        return success
    
    def unpage_tensor(self, tensor_id: str) -> bool:
        """Remove a tensor from pagination."""
        if tensor_id not in self.tensor_mappings:
            logger.warning(f"Tensor {tensor_id} not found in pagination")
            return False
        
        page_id = self.tensor_mappings[tensor_id]
        success = self.pagination_system.deallocate_page(page_id)
        
        if success:
            del self.tensor_mappings[tensor_id]
            del self.page_mappings[page_id]
        
        return success
    
    def access_tensor(self, tensor_id: str) -> Optional[Tensor]:
        """Access a paged tensor, ensuring it's in RAM."""
        if tensor_id not in self.tensor_mappings:
            logger.error(f"Tensor {tensor_id} not found in pagination")
            return None
        
        page_id = self.tensor_mappings[tensor_id]
        return self.pagination_system.access_page(page_id)
    
    def pin_tensor(self, tensor_id: str) -> bool:
        """Pin a tensor to prevent it from being paged out."""
        if tensor_id not in self.tensor_mappings:
            logger.error(f"Tensor {tensor_id} not found in pagination")
            return False
        
        page_id = self.tensor_mappings[tensor_id]
        
        if page_id in self.pagination_system.pages:
            self.pagination_system.pages[page_id].pinned = True
            return True
        
        return False
    
    def unpin_tensor(self, tensor_id: str) -> bool:
        """Unpin a tensor to allow it to be paged out."""
        if tensor_id not in self.tensor_mappings:
            logger.error(f"Tensor {tensor_id} not found in pagination")
            return False
        
        page_id = self.tensor_mappings[tensor_id]
        
        if page_id in self.pagination_system.pages:
            self.pagination_system.pages[page_id].pinned = False
            return True
        
        return False
    
    def get_tensor_priority(self, tensor_id: str) -> Optional[PaginationPriority]:
        """Get the priority of a paged tensor."""
        if tensor_id not in self.tensor_mappings:
            return None
        
        page_id = self.tensor_mappings[tensor_id]
        
        if page_id in self.pagination_system.pages:
            return self.pagination_system.pages[page_id].priority
        
        return None
    
    def set_tensor_priority(self, tensor_id: str, priority: PaginationPriority) -> bool:
        """Set the priority of a paged tensor."""
        if tensor_id not in self.tensor_mappings:
            logger.error(f"Tensor {tensor_id} not found in pagination")
            return False
        
        page_id = self.tensor_mappings[tensor_id]
        
        if page_id in self.pagination_system.pages:
            self.pagination_system.pages[page_id].priority = priority
            return True
        
        return False
    
    def _get_data_type_frequency(self, data_type: DataType) -> float:
        """Get expected access frequency for a data type."""
        frequency_map = {
            DataType.KV_CACHE: 0.9,  # Very high frequency
            DataType.ACTIVATIONS: 0.7,  # High frequency
            DataType.TEXT: 0.6,  # Medium-high frequency
            DataType.IMAGE: 0.4,  # Medium frequency
            DataType.EMBEDDINGS: 0.5,  # Medium frequency
            DataType.VIDEO: 0.3,  # Lower frequency
            DataType.AUDIO: 0.3   # Lower frequency
        }
        return frequency_map.get(data_type, 0.5)


def create_multimodal_pagination_system(
    swap_directory: Optional[Union[str, Path]] = None,
    page_size_mb: int = 16,
    eviction_policy: str = "intelligent",
    max_memory_ratio: float = 0.8
) -> Tuple[TensorPaginationSystem, MultimodalTensorPager]:
    """Create a multimodal pagination system with pager."""
    pagination_system = TensorPaginationSystem(
        swap_directory=swap_directory,
        page_size_mb=page_size_mb,
        eviction_policy=eviction_policy,
        max_memory_ratio=max_memory_ratio
    )
    
    pager = MultimodalTensorPager(pagination_system)
    
    return pagination_system, pager


__all__ = [
    "DataType",
    "PaginationPriority", 
    "TensorPage",
    "AccessPatternAnalyzer",
    "TensorPaginationSystem",
    "MultimodalTensorPager",
    "create_multimodal_pagination_system"
]