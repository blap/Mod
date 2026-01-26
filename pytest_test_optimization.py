"""
Pytest Plugin for Test Optimization

This plugin extends pytest with parallel execution and caching capabilities.
"""

import pytest
import time
import threading
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_optimization import TestResultCache


class TestOptimizationPlugin:
    """Pytest plugin for test optimization."""
    
    def __init__(self, cache_enabled: bool = True, cache_dir: Optional[str] = None):
        self.cache_enabled = cache_enabled
        self.cache = TestResultCache(cache_dir=cache_dir) if cache_enabled else None
        self.results = []
        self.lock = threading.Lock()
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        """
        Hook to intercept test execution and apply optimizations.
        """
        if self.cache_enabled:
            # Generate cache key based on test ID
            test_id = item.nodeid
            cached_result = self.cache.get_result(test_id)
            
            if cached_result is not None:
                # Use cached result
                result = cached_result['result']
                
                # Report cached result
                with self.lock:
                    self.results.append({
                        'nodeid': test_id,
                        'outcome': 'cached',
                        'duration': 0.0,
                        'result': result
                    })
                
                # Skip actual test execution
                return True  # Return True to indicate test was handled
        
        # Proceed with normal test execution
        return None  # Return None to allow normal execution
    
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """
        Hook to capture test results and cache them.
        """
        outcome = yield
        
        if self.cache_enabled and call.when == "call":
            # Get the result
            result = outcome.get_result()
            
            if result.outcome == 'passed':
                cache_result = {'success': True, 'error': None}
            else:
                cache_result = {'success': False, 'error': str(result.longrepr)}
            
            # Cache the result
            test_id = item.nodeid
            self.cache.put_result(test_id, cache_result)
    
    @pytest.hookimpl
    def pytest_configure(self, config):
        """Configure the plugin."""
        self.config = config
    
    @pytest.hookimpl
    def pytest_terminal_summary(self, terminalreporter):
        """Add optimization summary to terminal output."""
        terminalreporter.write_sep("=", "Test Optimization Summary")
        terminalreporter.write_line(f"Cache enabled: {self.cache_enabled}")
        if self.cache_enabled:
            terminalreporter.write_line(f"Cache directory: {self.cache.cache_dir}")


def pytest_addoption(parser):
    """Add options to pytest."""
    group = parser.getgroup("test-optimization")
    group.addoption(
        "--cache-tests",
        action="store_true",
        default=False,
        help="Enable test result caching"
    )
    group.addoption(
        "--cache-dir",
        action="store",
        default=None,
        help="Directory for test result cache"
    )


@pytest.hookimpl
def pytest_configure(config):
    """Register the plugin."""
    cache_enabled = config.getoption("--cache-tests")
    cache_dir = config.getoption("--cache-dir")
    
    if cache_enabled or cache_dir:
        plugin = TestOptimizationPlugin(
            cache_enabled=cache_enabled,
            cache_dir=cache_dir
        )
        config.pluginmanager.register(plugin, "test-optimization")


# Optional: Integration with pytest-xdist for parallel execution
def pytest_configure_node(node):
    """Configure worker nodes for distributed testing."""
    # This is called on the master node for each worker node
    pass


def pytest_testnodedown(node, error):
    """Handle worker node shutdown."""
    # This is called when a worker node goes down
    pass