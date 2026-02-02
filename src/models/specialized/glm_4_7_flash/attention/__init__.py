"""
GLM-4.7 Attention Package - Redirect to Plugin

This module redirects to the GLM-4.7 attention implementations in the plugin directory.
"""

# Redirect imports to the plugin directory
from ..plugin.glm47_attention import *
from ..plugin.glm47_multi_query_attention import *
from ..plugin.glm47_paged_attention import *
from ..plugin.glm47_sliding_window_attention import *
from ..plugin.glm47_sparse_attention import *
