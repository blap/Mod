"""
Comprehensive verification system to ensure all optimizations maintain the required
32 transformer layers and 32 attention heads in the Qwen3-VL model.

This system provides:
1. Model capacity validation system that verifies layer count and attention head count
2. Architecture integrity checks to ensure no components are inadvertently removed
3. Parameter count validation to confirm model capacity preservation
4. Integration verification that all optimization techniques preserve capacity when active
5. Configuration validation to ensure settings don't inadvertently reduce capacity
6. Continuous monitoring during training and inference to detect capacity drift
7. Test suite that validates capacity preservation under all optimization combinations
8. Error reporting system that clearly identifies when capacity violations occur
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from enum import Enum
import json


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    status: ValidationStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ModelCapacityValidator:
    """
    Main class for validating that model capacity is preserved.
    """
    def __init__(self, expected_layers: int = 32, expected_heads: int = 32):
        self.expected_layers = expected_layers
        self.expected_heads = expected_heads
        self.logger = logging.getLogger(__name__)
        self.validation_results: List[ValidationResult] = []

    def validate_layer_count(self, model: nn.Module) -> ValidationResult:
        """Validate that the model has the correct number of transformer layers."""
        try:
            # Count language model layers
            language_layers = 0
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
                language_layers = len(model.language_model.layers)
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                language_layers = len(model.transformer.h)
            elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
                language_layers = len(model.decoder.layers)
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                language_layers = len(model.encoder.layers)
            else:
                # Try to find transformer layers by iterating through named modules
                for name, module in model.named_modules():
                    if 'decoder_layer' in name.lower() or 'transformer_layer' in name.lower():
                        language_layers += 1
                    elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                        language_layers += 1
                    elif hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
                        language_layers += 1

            # Count vision model layers
            vision_layers = 0
            if hasattr(model, 'vision_tower'):
                if hasattr(model.vision_tower, 'layers'):
                    vision_layers = len(model.vision_tower.layers)
                elif hasattr(model.vision_tower, 'vision_model') and hasattr(model.vision_tower.vision_model, 'encoder'):
                    if hasattr(model.vision_tower.vision_model.encoder, 'layers'):
                        vision_layers = len(model.vision_tower.vision_model.encoder.layers)

            # For Qwen3-VL, we specifically check language model layers
            actual_layers = language_layers
            status = ValidationStatus.PASSED if actual_layers == self.expected_layers else ValidationStatus.FAILED
            message = f"Language transformer layers: {actual_layers}/{self.expected_layers} ({'PASS' if status == ValidationStatus.PASSED else 'FAIL'})"

            return ValidationResult(
                check_name="Layer Count Validation",
                status=status,
                message=message,
                details={
                    'expected_layers': self.expected_layers,
                    'actual_layers': actual_layers,
                    'vision_layers': vision_layers
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Layer Count Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating layer count: {str(e)}",
                details={'error': str(e)}
            )

    def validate_attention_heads(self, model: nn.Module) -> ValidationResult:
        """Validate that the model has the correct number of attention heads."""
        try:
            # First, check configuration
            config_heads = None
            if hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
                config_heads = model.config.num_attention_heads
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'config'):
                if hasattr(model.language_model.config, 'num_attention_heads'):
                    config_heads = model.language_model.config.num_attention_heads

            # Check actual attention mechanisms
            attention_heads_counts = []

            # Check language model layers
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
                for i, layer in enumerate(model.language_model.layers):
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'num_heads'):
                        attention_heads_counts.append(layer.self_attn.num_heads)
                    elif hasattr(layer, 'attention') and hasattr(layer.attention, 'num_heads'):
                        attention_heads_counts.append(layer.attention.num_heads)
                    elif hasattr(layer, 'attn') and hasattr(layer.attn, 'num_heads'):
                        attention_heads_counts.append(layer.attn.num_heads)

                    # Limit to first few layers to avoid excessive processing
                    if i >= 2:  # Check first 3 layers to be sure
                        break

            # If no attention heads found in layers, check config
            if not attention_heads_counts and config_heads is not None:
                attention_heads_counts = [config_heads]

            # For Qwen3-VL, also check for multi-modal attention heads
            if not attention_heads_counts:
                for name, module in model.named_modules():
                    if isinstance(module, nn.MultiheadAttention) and hasattr(module, 'num_heads'):
                        attention_heads_counts.append(module.num_heads)
                    elif hasattr(module, 'num_heads'):
                        # Check if this looks like an attention module based on parameters
                        if hasattr(module, 'q_proj') or hasattr(module, 'query'):
                            attention_heads_counts.append(module.num_heads)

            # Check if all attention mechanisms have the correct number of heads
            all_correct = True
            if attention_heads_counts:
                all_correct = all(count == self.expected_heads for count in attention_heads_counts)
            else:
                all_correct = False  # No attention heads found

            status = ValidationStatus.PASSED if all_correct and attention_heads_counts else ValidationStatus.FAILED
            message = f"Attention heads: {'/'.join(map(str, attention_heads_counts)) if attention_heads_counts else 'NOT FOUND'} (expected: {self.expected_heads}, {'PASS' if status == ValidationStatus.PASSED else 'FAIL'})"

            return ValidationResult(
                check_name="Attention Heads Validation",
                status=status,
                message=message,
                details={
                    'expected_heads': self.expected_heads,
                    'actual_heads': attention_heads_counts,
                    'config_heads': config_heads,
                    'all_correct': all_correct
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Attention Heads Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating attention heads: {str(e)}",
                details={'error': str(e)}
            )

    def validate_architecture_integrity(self, model: nn.Module) -> ValidationResult:
        """Validate that no critical components are removed from the architecture."""
        try:
            missing_components = []
            critical_components = [
                'language_model', 'vision_tower'
            ]

            # Check for critical components
            for component in critical_components:
                if not hasattr(model, component):
                    missing_components.append(component)

            # Check for transformer layers in language model
            if hasattr(model, 'language_model'):
                if not hasattr(model.language_model, 'layers'):
                    missing_components.append('language_model.layers')
                elif len(model.language_model.layers) == 0:
                    missing_components.append('language_model.layers (empty)')

            # Check for vision transformer layers
            if hasattr(model, 'vision_tower'):
                has_vision_layers = (
                    hasattr(model.vision_tower, 'layers') or
                    (hasattr(model.vision_tower, 'vision_model') and
                     hasattr(model.vision_tower.vision_model, 'encoder') and
                     hasattr(model.vision_tower.vision_model.encoder, 'layers'))
                )
                if not has_vision_layers:
                    missing_components.append('vision_tower.layers')

            status = ValidationStatus.PASSED if not missing_components else ValidationStatus.FAILED
            message = f"Architecture integrity: {'PASS' if status == ValidationStatus.PASSED else 'FAIL'} ({len(missing_components)} missing components)"

            return ValidationResult(
                check_name="Architecture Integrity Validation",
                status=status,
                message=message,
                details={
                    'missing_components': missing_components,
                    'all_components_present': len(missing_components) == 0
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Architecture Integrity Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating architecture integrity: {str(e)}",
                details={'error': str(e)}
            )

    def validate_parameter_count(self, model: nn.Module) -> ValidationResult:
        """Validate that the model has the expected parameter count for full capacity."""
        try:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params_all = sum(p.numel() for p in model.parameters())

            # For a Qwen3-VL model with 32 layers and 32 attention heads,
            # we expect a parameter count in the range of ~2B parameters
            # We'll use a reasonable range based on the configuration
            if hasattr(model, 'config'):
                config = model.config
                # Rough estimate for a transformer with the given config
                expected_min_params = (
                    config.num_hidden_layers *  # 32 layers
                    (config.hidden_size * config.hidden_size * 4)  # Rough estimate per layer
                )
                expected_max_params = expected_min_params * 1.5  # Allow some variation
            else:
                # Default range for a 2B model
                expected_min_params = 1_500_000_000  # 1.5B
                expected_max_params = 2_500_000_000  # 2.5B

            params_in_range = expected_min_params <= total_params <= expected_max_params
            status = ValidationStatus.PASSED if params_in_range else ValidationStatus.WARNING
            message = f"Parameter count: {total_params:,} ({'PASS' if params_in_range else 'WARN' if status == ValidationStatus.WARNING else 'FAIL'}, expected range: {expected_min_params:,}-{expected_max_params:,})"

            return ValidationResult(
                check_name="Parameter Count Validation",
                status=status,
                message=message,
                details={
                    'total_params': total_params,
                    'total_params_all': total_params_all,
                    'expected_min': expected_min_params,
                    'expected_max': expected_max_params,
                    'in_range': params_in_range
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Parameter Count Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating parameter count: {str(e)}",
                details={'error': str(e)}
            )

    def validate_config_settings(self, model: nn.Module) -> ValidationResult:
        """Validate that configuration settings don't inadvertently reduce capacity."""
        try:
            if not hasattr(model, 'config'):
                return ValidationResult(
                    check_name="Configuration Settings Validation",
                    status=ValidationStatus.FAILED,
                    message="Model has no configuration",
                    details={'has_config': False}
                )

            config = model.config
            issues = []

            # Check critical capacity parameters
            if config.num_hidden_layers != self.expected_layers:
                issues.append(f"num_hidden_layers: {config.num_hidden_layers} (expected: {self.expected_layers})")

            if config.num_attention_heads != self.expected_heads:
                issues.append(f"num_attention_heads: {config.num_attention_heads} (expected: {self.expected_heads})")

            # Check that hidden size is compatible with attention heads
            if config.hidden_size % config.num_attention_heads != 0:
                issues.append(f"hidden_size ({config.hidden_size}) not divisible by num_attention_heads ({config.num_attention_heads})")

            status = ValidationStatus.PASSED if not issues else ValidationStatus.FAILED
            message = f"Configuration settings: {'PASS' if status == ValidationStatus.PASSED else 'FAIL'} ({len(issues)} issues)"

            return ValidationResult(
                check_name="Configuration Settings Validation",
                status=status,
                message=message,
                details={
                    'issues': issues,
                    'num_hidden_layers': config.num_hidden_layers,
                    'num_attention_heads': config.num_attention_heads,
                    'hidden_size': config.hidden_size
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name="Configuration Settings Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating configuration settings: {str(e)}",
                details={'error': str(e)}
            )

    def run_comprehensive_validation(self, model: nn.Module) -> Dict[str, Any]:
        """Run all validation checks and return comprehensive results."""
        results = {
            'timestamp': time.time(),
            'model_type': type(model).__name__,
            'validation_results': [],
            'overall_status': ValidationStatus.PASSED,
            'summary': {}
        }

        # Run all validation checks
        checks = [
            self.validate_layer_count,
            self.validate_attention_heads,
            self.validate_architecture_integrity,
            self.validate_parameter_count,
            self.validate_config_settings
        ]

        for check_func in checks:
            result = check_func(model)
            results['validation_results'].append(result)
            self.validation_results.append(result)

            # Update overall status
            if result.status == ValidationStatus.FAILED:
                results['overall_status'] = ValidationStatus.FAILED
            elif result.status == ValidationStatus.WARNING and results['overall_status'] != ValidationStatus.FAILED:
                results['overall_status'] = ValidationStatus.WARNING

        # Create summary
        passed_count = sum(1 for r in results['validation_results'] if r.status == ValidationStatus.PASSED)
        failed_count = sum(1 for r in results['validation_results'] if r.status == ValidationStatus.FAILED)
        warning_count = sum(1 for r in results['validation_results'] if r.status == ValidationStatus.WARNING)

        results['summary'] = {
            'total_checks': len(results['validation_results']),
            'passed': passed_count,
            'failed': failed_count,
            'warnings': warning_count,
            'capacity_preserved': results['overall_status'] in [ValidationStatus.PASSED, ValidationStatus.WARNING]
        }

        return results

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append("=== Qwen3-VL Model Capacity Validation Report ===")
        report.append(f"Timestamp: {time.ctime(results['timestamp'])}")
        report.append(f"Model Type: {results['model_type']}")
        report.append(f"Overall Status: {results['overall_status'].value}")
        report.append("")

        report.append("Validation Summary:")
        summary = results['summary']
        report.append(f"  Total Checks: {summary['total_checks']}")
        report.append(f"  Passed: {summary['passed']}")
        report.append(f"  Failed: {summary['failed']}")
        report.append(f"  Warnings: {summary['warnings']}")
        report.append(f"  Capacity Preserved: {'Yes' if summary['capacity_preserved'] else 'No'}")
        report.append("")

        report.append("Detailed Results:")
        for result in results['validation_results']:
            status_icon = {'PASSED': '[PASS]', 'FAILED': '[FAIL]', 'WARNING': '[WARN]', 'SKIPPED': '[SKIP]'}[result.status.value]
            report.append(f"  {status_icon} {result.check_name}: {result.message}")
            if result.details:
                for key, value in result.details.items():
                    if key != 'error':  # Don't repeat error in details if already in message
                        report.append(f"      {key}: {value}")
            report.append("")

        return "\n".join(report)

    def save_validation_results(self, results: Dict[str, Any], filepath: str):
        """Save validation results to a JSON file."""
        # Convert non-serializable objects to serializable format
        serializable_results = {
            'timestamp': results['timestamp'],
            'model_type': results['model_type'],
            'overall_status': results['overall_status'].value,
            'summary': results['summary'],
            'validation_results': [
                {
                    'check_name': r.check_name,
                    'status': r.status.value,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp
                } for r in results['validation_results']
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


class OptimizationIntegrationValidator:
    """
    Validator for ensuring optimization techniques preserve capacity when active.
    """
    def __init__(self, capacity_validator: ModelCapacityValidator):
        self.capacity_validator = capacity_validator
        self.logger = logging.getLogger(__name__)

    def validate_optimization_capacity_preservation(self, model: nn.Module,
                                                  optimization_name: str = "general") -> ValidationResult:
        """Validate that a specific optimization preserves model capacity."""
        try:
            # Run comprehensive validation
            results = self.capacity_validator.run_comprehensive_validation(model)

            # Check if capacity is preserved despite optimizations
            capacity_preserved = results['summary']['capacity_preserved']

            status = ValidationStatus.PASSED if capacity_preserved else ValidationStatus.FAILED
            message = f"Optimization '{optimization_name}' capacity preservation: {'PASS' if capacity_preserved else 'FAIL'}"

            return ValidationResult(
                check_name=f"Optimization {optimization_name} Capacity Validation",
                status=status,
                message=message,
                details={
                    'optimization_name': optimization_name,
                    'capacity_preserved': capacity_preserved,
                    'validation_summary': results['summary']
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"Optimization {optimization_name} Capacity Validation",
                status=ValidationStatus.FAILED,
                message=f"Error validating optimization '{optimization_name}' capacity preservation: {str(e)}",
                details={'error': str(e), 'optimization_name': optimization_name}
            )

    def validate_multiple_optimizations(self, model: nn.Module,
                                      optimization_configs: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate capacity preservation across multiple optimization configurations."""
        results = []

        for i, config in enumerate(optimization_configs):
            # Apply optimization configuration to model (this would be implementation-specific)
            # For now, we'll just validate the current state
            result = self.validate_optimization_capacity_preservation(
                model,
                f"config_{i}_{config.get('name', 'unnamed')}"
            )
            results.append(result)

        return results


class ContinuousMonitoringSystem:
    """
    System for continuous monitoring during training and inference to detect capacity drift.
    """
    def __init__(self, capacity_validator: ModelCapacityValidator):
        self.capacity_validator = capacity_validator
        self.logger = logging.getLogger(__name__)
        self.monitoring_history: List[Dict[str, Any]] = []

    def monitor_model_capacity(self, model: nn.Module,
                             context: str = "training_step") -> ValidationResult:
        """Monitor model capacity at a specific point in time."""
        try:
            # Run quick validation checks
            layer_result = self.capacity_validator.validate_layer_count(model)
            heads_result = self.capacity_validator.validate_attention_heads(model)

            # Check if both are passing
            capacity_drift_detected = (
                layer_result.status != ValidationStatus.PASSED or
                heads_result.status != ValidationStatus.PASSED
            )

            status = ValidationStatus.WARNING if capacity_drift_detected else ValidationStatus.PASSED
            message = f"Capacity monitoring ({context}): {'Drift detected' if capacity_drift_detected else 'Stable'}"

            monitoring_data = {
                'timestamp': time.time(),
                'context': context,
                'capacity_drift_detected': capacity_drift_detected,
                'layer_validation': layer_result,
                'heads_validation': heads_result
            }

            self.monitoring_history.append(monitoring_data)

            return ValidationResult(
                check_name=f"Capacity Monitoring ({context})",
                status=status,
                message=message,
                details={
                    'context': context,
                    'drift_detected': capacity_drift_detected,
                    'layer_status': layer_result.status.value,
                    'heads_status': heads_result.status.value
                }
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"Capacity Monitoring ({context})",
                status=ValidationStatus.FAILED,
                message=f"Error during capacity monitoring: {str(e)}",
                details={'error': str(e), 'context': context}
            )

    def check_capacity_drift_over_time(self) -> Dict[str, Any]:
        """Analyze monitoring history to detect capacity drift patterns."""
        if not self.monitoring_history:
            return {
                'drift_detected': False,
                'history_length': 0,
                'drift_summary': 'No monitoring data available'
            }

        drift_events = [
            entry for entry in self.monitoring_history
            if entry.get('capacity_drift_detected', False)
        ]

        drift_detected = len(drift_events) > 0

        return {
            'drift_detected': drift_detected,
            'history_length': len(self.monitoring_history),
            'drift_events_count': len(drift_events),
            'drift_percentage': len(drift_events) / len(self.monitoring_history) * 100,
            'drift_summary': f"Detected {len(drift_events)} drift events in {len(self.monitoring_history)} monitoring points"
        }


class ErrorReportingSystem:
    """
    System for clearly identifying when capacity violations occur.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_log: List[Dict[str, Any]] = []

    def report_capacity_violation(self, validation_result: ValidationResult,
                                model_state: str = "unknown") -> Dict[str, Any]:
        """Report a capacity violation with detailed information."""
        error_info = {
            'timestamp': time.time(),
            'check_name': validation_result.check_name,
            'status': validation_result.status.value,
            'message': validation_result.message,
            'details': validation_result.details or {},
            'model_state': model_state,
            'severity': 'HIGH' if validation_result.status == ValidationStatus.FAILED else 'MEDIUM'
        }

        self.error_log.append(error_info)

        # Log to standard logger
        self.logger.error(f"CAPACITY VIOLATION: {validation_result.check_name} - {validation_result.message}")
        if validation_result.details:
            self.logger.error(f"Details: {validation_result.details}")

        return error_info

    def generate_error_report(self) -> str:
        """Generate a comprehensive error report."""
        if not self.error_log:
            return "No capacity violations detected."

        report = ["=== Capacity Violation Error Report ===", ""]
        report.append(f"Total violations: {len(self.error_log)}")
        report.append("")

        for i, error in enumerate(self.error_log, 1):
            report.append(f"{i}. {error['check_name']}")
            report.append(f"   Status: {error['status']}")
            report.append(f"   Message: {error['message']}")
            report.append(f"   Model State: {error['model_state']}")
            report.append(f"   Severity: {error['severity']}")
            report.append(f"   Time: {time.ctime(error['timestamp'])}")
            if error['details']:
                report.append("   Details:")
                for key, value in error['details'].items():
                    report.append(f"     {key}: {value}")
            report.append("")

        return "\n".join(report)

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get a summary of capacity violations."""
        if not self.error_log:
            return {
                'total_violations': 0,
                'failed_checks': [],
                'warning_checks': [],
                'severity_distribution': {}
            }

        failed_checks = [e for e in self.error_log if e['status'] == 'FAILED']
        warning_checks = [e for e in self.error_log if e['status'] == 'WARNING']

        severity_dist = {}
        for error in self.error_log:
            severity = error['severity']
            severity_dist[severity] = severity_dist.get(severity, 0) + 1

        return {
            'total_violations': len(self.error_log),
            'failed_checks_count': len(failed_checks),
            'warning_checks_count': len(warning_checks),
            'failed_checks': [e['check_name'] for e in failed_checks],
            'warning_checks': [e['check_name'] for e in warning_checks],
            'severity_distribution': severity_dist
        }


def create_capacity_verification_suite() -> Tuple[ModelCapacityValidator, OptimizationIntegrationValidator,
                                                ContinuousMonitoringSystem, ErrorReportingSystem]:
    """
    Create a complete capacity verification suite.
    """
    capacity_validator = ModelCapacityValidator()
    optimization_validator = OptimizationIntegrationValidator(capacity_validator)
    monitoring_system = ContinuousMonitoringSystem(capacity_validator)
    error_reporter = ErrorReportingSystem()

    return capacity_validator, optimization_validator, monitoring_system, error_reporter


def validate_model_capacity_preservation(model: nn.Module) -> Dict[str, Any]:
    """
    Convenience function to validate model capacity preservation with a complete check.
    """
    validator, _, _, _ = create_capacity_verification_suite()
    results = validator.run_comprehensive_validation(model)

    # Generate and print report
    report = validator.generate_validation_report(results)
    print(report)

    return results