"""
Advanced Hierarchical Caching and Buffering System for Qwen3-VL Model
Optimized for Intel i5-10210U + NVIDIA SM61 + NVMe SSD hardware

This system implements a three-level cache hierarchy with L1 (GPU HBM), 
L2 (CPU RAM), and L3 (NVMe SSD) caches, along with prediction algorithms 
and prefetching mechanisms.
"""

import math
import threading
import time
import os
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import numpy as np
import heapq
from pathlib import Path

class CacheLevel(Enum):
    """Níveis de cache na hierarquia"""
    L1_GPU = "L1_GPU"  # GPU HBM (ou VRAM)
    L2_CPU = "L2_CPU"  # CPU RAM
    L3_DISK = "L3_DISK"  # NVMe SSD

class TensorType(Enum):
    """Enumeração para diferentes tipos de tensores no modelo Qwen3-VL"""
    KV_CACHE = "kv_cache"
    IMAGE_FEATURES = "image_features"
    TEXT_EMBEDDINGS = "text_embeddings"
    GRADIENTS = "gradients"
    ACTIVATIONS = "activations"
    PARAMETERS = "parameters"

@dataclass
class CacheEntry:
    """Representa uma entrada no cache"""
    tensor_id: str
    tensor_type: TensorType
    data: Any  # O tensor propriamente dito
    size_bytes: int
    access_time: float
    access_count: int
    last_access_pattern: List[float]  # Histórico de acessos
    predicted_next_access: Optional[float] = None
    cache_level: CacheLevel = CacheLevel.L2_CPU  # Nível padrão
    hash_value: Optional[str] = None

    def __post_init__(self):
        """Calcula o hash do tensor para identificação única"""
        if self.hash_value is None:
            self.hash_value = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calcula o hash do tensor para identificação única"""
        # Para dados numéricos (como tensores), converter para bytes
        if hasattr(self.data, 'tobytes'):
            data_bytes = self.data.tobytes()
        elif isinstance(self.data, bytes):
            data_bytes = self.data
        else:
            # Para outros tipos de dados, usar pickle
            data_bytes = pickle.dumps(self.data)
        
        return hashlib.sha256(data_bytes).hexdigest()


class AccessPatternPredictor:
    """
    Algoritmo de predição de padrões de acesso baseado em histórico
    """
    def __init__(self):
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.pattern_models: Dict[str, Dict] = defaultdict(dict)
        self.lock = threading.Lock()

    def record_access(self, tensor_id: str, access_time: float):
        """Registra um acesso para um tensor"""
        with self.lock:
            self.access_history[tensor_id].append(access_time)
            
            # Manter apenas os últimos 20 acessos para manter a memória sob controle
            if len(self.access_history[tensor_id]) > 20:
                self.access_history[tensor_id] = self.access_history[tensor_id][-20:]

    def predict_next_access(self, tensor_id: str) -> Optional[float]:
        """
        Prediz o próximo acesso baseado em padrões históricos
        """
        with self.lock:
            if tensor_id not in self.access_history or len(self.access_history[tensor_id]) < 2:
                return None

            # Calcular intervalos entre acessos
            accesses = self.access_history[tensor_id]
            intervals = [accesses[i+1] - accesses[i] for i in range(len(accesses)-1)]
            
            if not intervals:
                return None

            # Prever o próximo acesso com base no intervalo médio
            avg_interval = sum(intervals) / len(intervals)
            last_access = accesses[-1]
            predicted_time = last_access + avg_interval
            
            return predicted_time

    def get_access_frequency(self, tensor_id: str) -> float:
        """Obtém a frequência de acesso para um tensor"""
        with self.lock:
            if tensor_id not in self.access_history:
                return 0.0
            
            accesses = self.access_history[tensor_id]
            if len(accesses) < 2:
                return 0.0
            
            # Frequência = número de acessos / período total
            time_span = accesses[-1] - accesses[0] if accesses[-1] != accesses[0] else 1.0
            return len(accesses) / time_span if time_span > 0 else 0.0


class PrefetchingManager:
    """
    Gerenciador de prefetching avançado
    """
    def __init__(self, cache_manager: 'HierarchicalCacheManager'):
        self.cache_manager = cache_manager
        self.prefetch_queue: List[Tuple[float, str]] = []  # (predicted_time, tensor_id)
        self.prefetch_lock = threading.Lock()
        self.prefetch_thread = None
        self.running = False
        
    def start_prefetching(self):
        """Inicia o thread de prefetching"""
        self.running = True
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def stop_prefetching(self):
        """Para o thread de prefetching"""
        self.running = False
        if self.prefetch_thread:
            self.prefetch_thread.join()
    
    def schedule_prefetch(self, tensor_id: str, predicted_time: float, prefetch_distance: float = 1.0):
        """Agenda um tensor para prefetching"""
        # Ajustar o tempo previsto com base na distância de prefetch e no hardware
        adjusted_time = predicted_time - (prefetch_distance * self.cache_manager.prefetch_distance_multiplier)

        with self.prefetch_lock:
            heapq.heappush(self.prefetch_queue, (adjusted_time, tensor_id))
    
    def _prefetch_worker(self):
        """Thread worker para executar prefetching"""
        while self.running:
            with self.prefetch_lock:
                if self.prefetch_queue:
                    next_time, tensor_id = self.prefetch_queue[0]
                    current_time = time.time()
                    
                    if current_time >= next_time:
                        # Remover da fila
                        heapq.heappop(self.prefetch_queue)
                        
                        # Tentar carregar o tensor no cache superior
                        self._execute_prefetch(tensor_id)
            
            # Esperar um pouco antes da próxima verificação
            time.sleep(0.01)
    
    def _execute_prefetch(self, tensor_id: str):
        """Executa o prefetching real de um tensor"""
        # Verificar se o tensor ainda está no cache L3 (SSD)
        l3_entry = self.cache_manager.l3_cache.get(tensor_id)
        if l3_entry:
            # Mover para L2 (RAM) ou L1 (GPU) se for apropriado
            self.cache_manager._move_to_higher_cache_level(l3_entry)


class LRUCache:
    """
    Implementação de cache LRU (Least Recently Used)
    """
    def __init__(self, capacity_bytes: int):
        self.capacity_bytes = capacity_bytes
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size = 0
        self.lock = threading.Lock()
    
    def get(self, tensor_id: str) -> Optional[CacheEntry]:
        """Obtém um tensor do cache"""
        with self.lock:
            if tensor_id in self.cache:
                entry = self.cache[tensor_id]
                # Mover para o final (mais recente)
                self.cache.move_to_end(tensor_id)
                # Atualizar contagem de acesso
                entry.access_count += 1
                entry.access_time = time.time()
                return entry
            return None
    
    def put(self, entry: CacheEntry) -> bool:
        """Adiciona um tensor ao cache"""
        with self.lock:
            # Se já existir, atualizar
            if entry.tensor_id in self.cache:
                old_entry = self.cache[entry.tensor_id]
                self.current_size -= old_entry.size_bytes
            
            # Verificar se há espaço suficiente
            if entry.size_bytes > self.capacity_bytes:
                return False  # Tensor é muito grande para este cache
            
            # Remover entradas antigas até haver espaço
            while (self.current_size + entry.size_bytes > self.capacity_bytes and 
                   len(self.cache) > 0):
                oldest_key, oldest_entry = next(iter(self.cache.items()))
                self.cache.pop(oldest_key)
                self.current_size -= oldest_entry.size_bytes
            
            # Adicionar nova entrada
            if self.current_size + entry.size_bytes <= self.capacity_bytes:
                self.cache[entry.tensor_id] = entry
                self.current_size += entry.size_bytes
                return True
            else:
                return False  # Não há espaço suficiente
    
    def evict(self, tensor_id: str) -> bool:
        """Remove um tensor específico do cache"""
        with self.lock:
            if tensor_id in self.cache:
                entry = self.cache[tensor_id]
                del self.cache[tensor_id]
                self.current_size -= entry.size_bytes
                return True
            return False
    
    def get_size(self) -> int:
        """Retorna o tamanho atual do cache"""
        with self.lock:
            return self.current_size
    
    def get_capacity(self) -> int:
        """Retorna a capacidade do cache"""
        return self.capacity_bytes


class HierarchicalCacheManager:
    """
    Gerenciador de cache hierárquico com L1 (GPU), L2 (CPU) e L3 (SSD)
    Otimizado para Intel i5-10210U + NVIDIA SM61 + NVMe SSD
    """
    def __init__(self,
                 l1_gpu_size: int = 512 * 1024 * 1024,  # 512MB GPU memory
                 l2_cpu_size: int = 2 * 1024 * 1024 * 1024,  # 2GB RAM
                 l3_disk_path: str = "./cache_l3_disk",  # Pasta para cache SSD
                 l3_disk_size: int = 10 * 1024 * 1024 * 1024,  # 10GB SSD
                 hardware_profile: Optional[Dict] = None):  # Perfil de hardware
        """
        Inicializa o gerenciador de cache hierárquico

        Args:
            l1_gpu_size: Tamanho do cache L1 (GPU)
            l2_cpu_size: Tamanho do cache L2 (CPU)
            l3_disk_path: Caminho para armazenamento do cache L3
            l3_disk_size: Tamanho máximo do cache L3 (SSD)
            hardware_profile: Perfil de hardware para otimizações específicas
        """
        # Perfil de hardware para otimizações
        self.hardware_profile = hardware_profile or {
            'cpu_cores': 4,  # i5-10210U tem 4 cores
            'cpu_frequency': 4.2e9,  # 4.2 GHz
            'cpu_l1_cache': 128 * 1024,  # 128KB
            'cpu_l2_cache': 1 * 1024 * 1024,  # 1MB
            'cpu_l3_cache': 6 * 1024 * 1024,  # 6MB
            'cpu_ram': 16 * 1024 * 1024 * 1024,  # 16GB (exemplo)
            'gpu_memory': 4 * 1024 * 1024 * 1024,  # 4GB VRAM (exemplo para SM61)
            'gpu_bandwidth': 25.6e9,  # 25.6 GB/s (exemplo para SM61)
            'ssd_read_speed': 3500 * 1024 * 1024,  # 3500 MB/s
            'ssd_write_speed': 3000 * 1024 * 1024,  # 3000 MB/s
            'ssd_latency': 0.000025  # 25μs de latência típica
        }

        # Inicializar caches com otimizações de hardware
        self.l1_cache = LRUCache(l1_gpu_size)  # GPU cache (simulado)
        self.l2_cache = LRUCache(l2_cpu_size)  # CPU cache (RAM)

        # Calcular tamanhos ideais de cache baseado no hardware
        self._calculate_optimal_cache_sizes()

        # Criar diretório para cache em disco
        self.l3_disk_path = Path(l3_disk_path)
        self.l3_disk_path.mkdir(exist_ok=True)
        self.l3_cache: Dict[str, CacheEntry] = {}  # Cache em disco (simulado com arquivos)
        self.l3_cache_size = 0
        self.l3_max_size = l3_disk_size
        self.l3_lock = threading.Lock()

        # Componentes auxiliares
        self.access_predictor = AccessPatternPredictor()
        self.prefetch_manager = PrefetchingManager(self)
        self.prefetch_manager.start_prefetching()

        # Estatísticas
        self.stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0,
            'prefetch_hits': 0,
            'total_requests': 0
        }
        self.stats_lock = threading.Lock()

        # Locks para proteção de acesso
        self.main_lock = threading.Lock()

        # Configurações específicas de hardware para otimização
        self._setup_hardware_optimizations()

    def _calculate_optimal_cache_sizes(self):
        """Calcula tamanhos ideais de cache baseado no hardware disponível"""
        # Ajustar tamanhos com base nas limitações de hardware
        cpu_ram = self.hardware_profile.get('cpu_ram', 8 * 1024 * 1024 * 1024)  # 8GB padrão
        gpu_memory = self.hardware_profile.get('gpu_memory', 2 * 1024 * 1024 * 1024)  # 2GB padrão

        # Limitar L1 (GPU) a 10% da memória GPU ou 512MB, o que for menor
        max_l1_size = min(
            int(gpu_memory * 0.1),
            512 * 1024 * 1024
        )
        if self.l1_cache.capacity_bytes > max_l1_size:
            self.l1_cache = LRUCache(max_l1_size)

        # Limitar L2 (CPU) a 10% da RAM ou 2GB, o que for menor
        max_l2_size = min(
            int(cpu_ram * 0.1),
            2 * 1024 * 1024 * 1024
        )
        if self.l2_cache.capacity_bytes > max_l2_size:
            self.l2_cache = LRUCache(max_l2_size)

    def _setup_hardware_optimizations(self):
        """Configura otimizações específicas para o hardware"""
        # Configurar thresholds de acesso baseado na velocidade do hardware
        cpu_frequency = self.hardware_profile.get('cpu_frequency', 2.5e9)  # 2.5 GHz padrão
        gpu_bandwidth = self.hardware_profile.get('gpu_bandwidth', 100e9)  # 100 GB/s padrão
        ssd_latency = self.hardware_profile.get('ssd_latency', 0.0001)  # 100μs padrão

        # Ajustar parâmetros de acesso com base na velocidade do hardware
        self.access_frequency_threshold = 10.0  # Base, ajustado abaixo
        self.prefetch_distance_multiplier = 1.0  # Base, ajustado abaixo

        # Ajustar thresholds com base na frequência do CPU
        if cpu_frequency > 3.0e9:  # CPU mais rápido
            self.access_frequency_threshold *= 1.5
            self.prefetch_distance_multiplier *= 1.2

        # Ajustar prefetching com base na largura de banda da GPU
        if gpu_bandwidth > 200e9:  # GPU com alta largura de banda
            self.prefetch_distance_multiplier *= 1.5

        # Ajustar políticas com base na latência do SSD
        if ssd_latency < 0.00005:  # SSD muito rápido
            self.prefetch_distance_multiplier *= 1.3
    
    def get_tensor(self, tensor_id: str) -> Optional[CacheEntry]:
        """
        Obtém um tensor da hierarquia de cache
        
        Args:
            tensor_id: ID do tensor a ser recuperado
            
        Returns:
            CacheEntry se encontrado, None caso contrário
        """
        with self.main_lock:
            self._update_stats('total_requests', 1)
            
            # Tentar L1 primeiro (GPU cache)
            entry = self.l1_cache.get(tensor_id)
            if entry is not None:
                self._update_stats('l1_hits', 1)
                # Atualizar predição de acesso
                predicted_time = self.access_predictor.predict_next_access(tensor_id)
                if predicted_time:
                    entry.predicted_next_access = predicted_time
                    # Usar distância de prefetch otimizada para o hardware
                    prefetch_distance = self._get_optimal_prefetch_distance(entry.tensor_type)
                    self.prefetch_manager.schedule_prefetch(tensor_id, predicted_time, prefetch_distance)
                return entry
            
            self._update_stats('l1_misses', 1)
            
            # Tentar L2 (CPU cache)
            entry = self.l2_cache.get(tensor_id)
            if entry is not None:
                self._update_stats('l2_hits', 1)
                # Mover para L1 se for um acesso frequente
                if self._should_promote_to_l1(entry):
                    self._move_to_higher_cache_level(entry)

                # Atualizar predição de acesso
                predicted_time = self.access_predictor.predict_next_access(tensor_id)
                if predicted_time:
                    entry.predicted_next_access = predicted_time
                    # Usar distância de prefetch otimizada para o hardware
                    prefetch_distance = self._get_optimal_prefetch_distance(entry.tensor_type)
                    self.prefetch_manager.schedule_prefetch(tensor_id, predicted_time, prefetch_distance)
                return entry
            
            self._update_stats('l2_misses', 1)
            
            # Tentar L3 (SSD cache)
            entry = self._get_from_l3_cache(tensor_id)
            if entry is not None:
                self._update_stats('l3_hits', 1)
                # Mover para L2 (e possivelmente L1)
                self._move_to_higher_cache_level(entry)

                # Atualizar predição de acesso
                predicted_time = self.access_predictor.predict_next_access(tensor_id)
                if predicted_time:
                    entry.predicted_next_access = predicted_time
                    # Usar distância de prefetch otimizada para o hardware
                    prefetch_distance = self._get_optimal_prefetch_distance(entry.tensor_type)
                    self.prefetch_manager.schedule_prefetch(tensor_id, predicted_time, prefetch_distance)
                return entry
            
            self._update_stats('l3_misses', 1)
            return None
    
    def put_tensor(self, entry: CacheEntry) -> bool:
        """
        Armazena um tensor na hierarquia de cache
        
        Args:
            entry: Entrada de cache contendo o tensor
            
        Returns:
            True se armazenado com sucesso, False caso contrário
        """
        with self.main_lock:
            # Registrar acesso para predição
            self.access_predictor.record_access(entry.tensor_id, entry.access_time)
            
            # Determinar o nível apropriado para armazenamento inicial
            if self._should_store_in_l1(entry):
                # Tentar armazenar em L1 (GPU cache)
                if self.l1_cache.put(entry):
                    return True
            elif self._should_store_in_l2(entry):
                # Tentar armazenar em L2 (CPU cache)
                if self.l2_cache.put(entry):
                    return True
            
            # Armazenar em L3 (SSD cache) como fallback
            return self._put_in_l3_cache(entry)
    
    def _should_store_in_l1(self, entry: CacheEntry) -> bool:
        """
        Determina se um tensor deve ser armazenado no cache L1 (GPU)
        """
        # Critérios para armazenamento em L1:
        # - Acesso muito frequente
        # - Tamanho pequeno
        # - Tipo de tensor que se beneficia de GPU
        freq = self.access_predictor.get_access_frequency(entry.tensor_id)
        is_gpu_beneficial = entry.tensor_type in [TensorType.IMAGE_FEATURES,
                                                  TensorType.ACTIVATIONS,
                                                  TensorType.KV_CACHE]

        # Usar threshold otimizado para o hardware
        return (freq > self.access_frequency_threshold and  # Frequência baseada no hardware
                entry.size_bytes < 10 * 1024 * 1024 and  # Menos de 10MB
                is_gpu_beneficial)
    
    def _should_store_in_l2(self, entry: CacheEntry) -> bool:
        """
        Determina se um tensor deve ser armazenado no cache L2 (CPU)
        """
        # Critérios para armazenamento em L2:
        # - Acesso frequente
        # - Tamanho moderado
        freq = self.access_predictor.get_access_frequency(entry.tensor_id)
        
        return (freq > 1.0 and  # Mais de 1 acesso por segundo
                entry.size_bytes < 100 * 1024 * 1024)  # Menos de 100MB
    
    def _should_promote_to_l1(self, entry: CacheEntry) -> bool:
        """
        Determina se um tensor deve ser promovido para o cache L1
        """
        # Promover se for acessado com alta frequência
        freq = self.access_predictor.get_access_frequency(entry.tensor_id)
        # Usar threshold otimizado para o hardware (mais 50% que o threshold base)
        return freq > self.access_frequency_threshold * 1.5
    
    def _move_to_higher_cache_level(self, entry: CacheEntry):
        """
        Move um tensor para um nível de cache superior
        """
        # Remover do nível atual
        if entry.cache_level == CacheLevel.L3_DISK:
            self._remove_from_l3_cache(entry.tensor_id)
        elif entry.cache_level == CacheLevel.L2_CPU:
            self.l2_cache.evict(entry.tensor_id)
        
        # Tentar promover para L1
        if self._should_promote_to_l1(entry):
            # Atualizar o nível do cache
            new_entry = CacheEntry(
                tensor_id=entry.tensor_id,
                tensor_type=entry.tensor_type,
                data=entry.data,
                size_bytes=entry.size_bytes,
                access_time=entry.access_time,
                access_count=entry.access_count,
                last_access_pattern=entry.last_access_pattern,
                predicted_next_access=entry.predicted_next_access,
                cache_level=CacheLevel.L1_GPU
            )
            
            if self.l1_cache.put(new_entry):
                return
        
        # Tentar promover para L2
        if entry.cache_level != CacheLevel.L2_CPU:
            new_entry = CacheEntry(
                tensor_id=entry.tensor_id,
                tensor_type=entry.tensor_type,
                data=entry.data,
                size_bytes=entry.size_bytes,
                access_time=entry.access_time,
                access_count=entry.access_count,
                last_access_pattern=entry.last_access_pattern,
                predicted_next_access=entry.predicted_next_access,
                cache_level=CacheLevel.L2_CPU
            )
            
            if self.l2_cache.put(new_entry):
                return
    
    def _get_from_l3_cache(self, tensor_id: str) -> Optional[CacheEntry]:
        """
        Recupera um tensor do cache L3 (SSD)
        """
        with self.l3_lock:
            if tensor_id in self.l3_cache:
                # Carregar dados do arquivo se necessário
                entry = self.l3_cache[tensor_id]
                if entry.data is None:
                    file_path = self.l3_disk_path / f"{tensor_id}.cache"
                    if file_path.exists():
                        try:
                            with open(file_path, 'rb') as f:
                                entry.data = pickle.load(f)
                        except Exception:
                            return None
                
                # Atualizar contagem de acesso
                entry.access_count += 1
                entry.access_time = time.time()
                
                return entry
            return None
    
    def _put_in_l3_cache(self, entry: CacheEntry) -> bool:
        """
        Armazena um tensor no cache L3 (SSD)
        """
        with self.l3_lock:
            # Verificar se há espaço suficiente
            if entry.size_bytes > self.l3_max_size:
                return False
            
            # Remover entradas antigas até haver espaço
            while (self.l3_cache_size + entry.size_bytes > self.l3_max_size and 
                   len(self.l3_cache) > 0):
                # Remover a entrada mais antiga
                oldest_id = min(self.l3_cache.keys(), 
                               key=lambda k: self.l3_cache[k].access_time)
                self._remove_from_l3_cache(oldest_id)
            
            # Armazenar dados em arquivo
            file_path = self.l3_disk_path / f"{entry.tensor_id}.cache"
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.data, f)
                
                # Armazenar apenas metadados na memória
                metadata_entry = CacheEntry(
                    tensor_id=entry.tensor_id,
                    tensor_type=entry.tensor_type,
                    data=None,  # Dados estão no disco
                    size_bytes=entry.size_bytes,
                    access_time=entry.access_time,
                    access_count=entry.access_count,
                    last_access_pattern=entry.last_access_pattern,
                    predicted_next_access=entry.predicted_next_access,
                    cache_level=CacheLevel.L3_DISK,
                    hash_value=entry.hash_value
                )
                
                if self.l3_cache_size + entry.size_bytes <= self.l3_max_size:
                    self.l3_cache[entry.tensor_id] = metadata_entry
                    self.l3_cache_size += entry.size_bytes
                    return True
                else:
                    # Remover arquivo se não houver espaço
                    if file_path.exists():
                        file_path.unlink()
                    return False
            except Exception:
                return False
    
    def _remove_from_l3_cache(self, tensor_id: str) -> bool:
        """
        Remove um tensor do cache L3 (SSD)
        """
        with self.l3_lock:
            if tensor_id in self.l3_cache:
                entry = self.l3_cache[tensor_id]
                del self.l3_cache[tensor_id]
                self.l3_cache_size -= entry.size_bytes
                
                # Remover arquivo físico
                file_path = self.l3_disk_path / f"{tensor_id}.cache"
                if file_path.exists():
                    file_path.unlink()
                
                return True
            return False
    
    def _get_optimal_prefetch_distance(self, tensor_type: TensorType) -> float:
        """
        Retorna a distância de prefetch otimizada para um tipo específico de tensor
        com base nas características do hardware
        """
        # Ajustar distância de prefetch com base no tipo de tensor e hardware
        base_distance = 1.0

        # Tensores KV cache geralmente têm padrões previsíveis e se beneficiam de prefetching
        if tensor_type == TensorType.KV_CACHE:
            return base_distance * 2.0 * self.prefetch_distance_multiplier

        # Features de imagem podem ser acessadas em sequência
        elif tensor_type == TensorType.IMAGE_FEATURES:
            return base_distance * 3.0 * self.prefetch_distance_multiplier

        # Embeddings de texto geralmente têm acesso mais aleatório
        elif tensor_type == TensorType.TEXT_EMBEDDINGS:
            return base_distance * 1.5 * self.prefetch_distance_multiplier

        # Gradients e ativações podem ter padrões de acesso mais complexos
        elif tensor_type in [TensorType.GRADIENTS, TensorType.ACTIVATIONS]:
            return base_distance * 1.8 * self.prefetch_distance_multiplier

        # Parâmetros geralmente são acessados de forma mais previsível
        elif tensor_type == TensorType.PARAMETERS:
            return base_distance * 2.5 * self.prefetch_distance_multiplier

        else:
            return base_distance * self.prefetch_distance_multiplier

    def _update_stats(self, stat_name: str, increment: int = 1):
        """Atualiza as estatísticas"""
        with self.stats_lock:
            if stat_name in self.stats:
                self.stats[stat_name] += increment
            else:
                self.stats[stat_name] = increment
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema de cache"""
        with self.stats_lock:
            total_requests = self.stats['total_requests']
            l1_hit_rate = self.stats['l1_hits'] / total_requests if total_requests > 0 else 0
            l2_hit_rate = self.stats['l2_hits'] / total_requests if total_requests > 0 else 0
            l3_hit_rate = self.stats['l3_hits'] / total_requests if total_requests > 0 else 0
            overall_hit_rate = (self.stats['l1_hits'] + self.stats['l2_hits'] + self.stats['l3_hits']) / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'l1_hit_rate': l1_hit_rate,
                'l2_hit_rate': l2_hit_rate,
                'l3_hit_rate': l3_hit_rate,
                'overall_hit_rate': overall_hit_rate,
                'l1_cache_size': self.l1_cache.get_size(),
                'l1_cache_capacity': self.l1_cache.get_capacity(),
                'l2_cache_size': self.l2_cache.get_size(),
                'l2_cache_capacity': self.l2_cache.get_capacity(),
                'l3_cache_size': self.l3_cache_size,
                'l3_cache_capacity': self.l3_max_size
            }
    
    def clear_cache(self, cache_level: Optional[CacheLevel] = None):
        """Limpa um ou todos os níveis de cache"""
        with self.main_lock:
            if cache_level is None or cache_level == CacheLevel.L1_GPU:
                self.l1_cache = LRUCache(self.l1_cache.capacity_bytes)
            
            if cache_level is None or cache_level == CacheLevel.L2_CPU:
                self.l2_cache = LRUCache(self.l2_cache.capacity_bytes)
            
            if cache_level is None or cache_level == CacheLevel.L3_DISK:
                with self.l3_lock:
                    # Remover todos os arquivos de cache
                    for file_path in self.l3_disk_path.glob("*.cache"):
                        file_path.unlink()
                    
                    self.l3_cache.clear()
                    self.l3_cache_size = 0
    
    def shutdown(self):
        """Desliga o gerenciador de cache e seus componentes"""
        self.prefetch_manager.stop_prefetching()
        # Limpar arquivos temporários do cache L3
        with self.l3_lock:
            for file_path in self.l3_disk_path.glob("*.cache"):
                file_path.unlink()
            self.l3_cache.clear()


class Qwen3VLCacheOptimizer:
    """
    Otimizador específico para o modelo Qwen3-VL com o hardware Intel i5-10210U + NVIDIA SM61 + NVMe SSD
    """
    def __init__(self, cache_manager: HierarchicalCacheManager):
        self.cache_manager = cache_manager
        self.hardware_profile = {
            'cpu_cores': 4,  # i5-10210U tem 4 cores
            'cpu_frequency': 4.2e9,  # 4.2 GHz
            'cpu_l1_cache': 128 * 1024,  # 128KB
            'cpu_l2_cache': 1 * 1024 * 1024,  # 1MB
            'cpu_l3_cache': 6 * 1024 * 1024,  # 6MB
            'cpu_ram': 16 * 1024 * 1024 * 1024,  # 16GB (exemplo)
            'gpu_memory': 4 * 1024 * 1024 * 1024,  # 4GB VRAM (exemplo para SM61)
            'gpu_bandwidth': 25.6e9,  # 25.6 GB/s (exemplo para SM61)
            'ssd_read_speed': 3500 * 1024 * 1024,  # 3500 MB/s
            'ssd_write_speed': 3000 * 1024 * 1024,  # 3000 MB/s
        }
    
    def optimize_cache_for_tensor_type(self, tensor_type: TensorType) -> Dict[str, Any]:
        """
        Otimiza os parâmetros de cache para um tipo específico de tensor
        """
        if tensor_type == TensorType.KV_CACHE:
            # KV cache é acessado com alta frequência
            return {
                'l1_priority': True,
                'prefetch_distance': 2,  # Prefetch 2 próximos tensores
                'access_pattern_threshold': 5.0,  # Considerar padrão após 5 acessos
            }
        elif tensor_type == TensorType.IMAGE_FEATURES:
            # Features de imagem são grandes e acessadas sequencialmente
            return {
                'l1_priority': False,
                'prefetch_distance': 3,  # Prefetch mais à frente
                'access_pattern_threshold': 2.0,
            }
        elif tensor_type == TensorType.TEXT_EMBEDDINGS:
            # Embeddings de texto são de tamanho fixo e frequentemente acessados
            return {
                'l1_priority': True,
                'prefetch_distance': 1,
                'access_pattern_threshold': 3.0,
            }
        elif tensor_type == TensorType.GRADIENTS:
            # Gradients são atualizados frequentemente
            return {
                'l1_priority': False,
                'prefetch_distance': 1,
                'access_pattern_threshold': 4.0,
            }
        elif tensor_type == TensorType.ACTIVATIONS:
            # Ativações são temporárias e intensas
            return {
                'l1_priority': True,
                'prefetch_distance': 1,
                'access_pattern_threshold': 3.0,
            }
        elif tensor_type == TensorType.PARAMETERS:
            # Parâmetros são grandes e persistentes
            return {
                'l1_priority': False,
                'prefetch_distance': 5,  # Prefetch mais longe
                'access_pattern_threshold': 1.0,  # Mesmo padrão simples é útil
            }
        else:
            return {
                'l1_priority': False,
                'prefetch_distance': 1,
                'access_pattern_threshold': 3.0,
            }
    
    def get_optimal_cache_sizes(self, total_available_memory: int) -> Tuple[int, int, int]:
        """
        Calcula tamanhos ideais de cache baseado na memória disponível e hardware
        """
        # Alocar memória com base nas características do hardware
        # L1 (GPU): 10% da VRAM ou 512MB, o que for menor
        l1_size = min(
            int(self.hardware_profile['gpu_memory'] * 0.1),
            512 * 1024 * 1024
        )
        
        # L2 (CPU): 10% da RAM ou 2GB, o que for menor
        l2_size = min(
            int(self.hardware_profile['cpu_ram'] * 0.1),
            2 * 1024 * 1024 * 1024
        )
        
        # L3 (SSD): O restante, com limite de 20GB
        l3_size = min(
            total_available_memory - l1_size - l2_size,
            20 * 1024 * 1024 * 1024
        )
        
        return l1_size, l2_size, l3_size


# Exemplo de uso e demonstração
if __name__ == "__main__":
    print("Inicializando o sistema de cache hierárquico para Qwen3-VL...")
    
    # Criar o gerenciador de cache hierárquico
    cache_manager = HierarchicalCacheManager(
        l1_gpu_size=256 * 1024 * 1024,  # 256MB
        l2_cpu_size=1 * 1024 * 1024 * 1024,  # 1GB
        l3_disk_path="./qwen3_vl_cache",
        l3_disk_size=5 * 1024 * 1024 * 1024  # 5GB
    )
    
    # Criar otimizador para Qwen3-VL
    optimizer = Qwen3VLCacheOptimizer(cache_manager)
    
    print("Testando operações de cache...")
    
    # Criar alguns tensores de exemplo
    import numpy as np
    
    # Tensor KV cache (muito acessado)
    kv_tensor = np.random.random((1000, 512)).astype(np.float32)
    kv_entry = CacheEntry(
        tensor_id="kv_tensor_1",
        tensor_type=TensorType.KV_CACHE,
        data=kv_tensor,
        size_bytes=kv_tensor.nbytes,
        access_time=time.time(),
        access_count=0,
        last_access_pattern=[]
    )
    
    # Tensor de features de imagem (acessado sequencialmente)
    img_tensor = np.random.random((3, 224, 224)).astype(np.float32)
    img_entry = CacheEntry(
        tensor_id="img_tensor_1",
        tensor_type=TensorType.IMAGE_FEATURES,
        data=img_tensor,
        size_bytes=img_tensor.nbytes,
        access_time=time.time(),
        access_count=0,
        last_access_pattern=[]
    )
    
    # Armazenar tensores no cache
    print("Armazenando tensores no cache...")
    cache_manager.put_tensor(kv_entry)
    cache_manager.put_tensor(img_entry)
    
    # Recuperar tensores do cache
    print("Recuperando tensores do cache...")
    retrieved_kv = cache_manager.get_tensor("kv_tensor_1")
    retrieved_img = cache_manager.get_tensor("img_tensor_1")
    
    if retrieved_kv:
        print(f"KV tensor recuperado: {retrieved_kv.size_bytes} bytes")
    if retrieved_img:
        print(f"Image tensor recuperado: {retrieved_img.size_bytes} bytes")
    
    # Acessar novamente para testar predição
    print("Testando predição de acesso...")
    time.sleep(0.1)  # Aguardar um pouco
    cache_manager.get_tensor("kv_tensor_1")  # Acessar novamente
    time.sleep(0.1)
    cache_manager.get_tensor("kv_tensor_1")  # Acessar novamente
    
    # Mostrar estatísticas
    print("\nEstatísticas do cache:")
    stats = cache_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Mostrar otimizações específicas
    print("\nOtimizações para diferentes tipos de tensores:")
    for tensor_type in TensorType:
        opts = optimizer.optimize_cache_for_tensor_type(tensor_type)
        print(f"  {tensor_type.value}: {opts}")
    
    # Finalizar
    cache_manager.shutdown()
    print("\nSistema de cache hierárquico finalizado.")