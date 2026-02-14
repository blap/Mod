from ..base import AMDBasePlugin
from ...common.utils.lib_loader import load_backend_lib
import logging

logger = logging.getLogger(__name__)

class AMDRX550Plugin(AMDBasePlugin):
    """
    Optimized AMD Plugin for Polaris (RX 550).
    """
    def _load_library(self):
        try:
            self.lib = load_backend_lib("amd", "rx550")
            logger.info("Loaded optimized AMD RX550 backend.")
        except Exception:
            logger.info("RX550 specific backend not found, falling back to generic AMD.")
            try:
                self.lib = load_backend_lib("amd")
            except Exception as e:
                logger.warning(f"Failed to load generic AMD backend: {e}")

        self._setup_signatures()

    def get_device_info(self):
        return {"vendor": "AMD", "backend": "OpenCL", "arch": "Polaris (RX 550)"}
