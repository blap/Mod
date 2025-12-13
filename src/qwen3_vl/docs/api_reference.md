# Qwen3-VL API Reference

This document provides a reference for the main APIs available in the Qwen3-VL package.

## Main Package

### `Qwen3VLConfig`
Main configuration class for the Qwen3-VL model.

**Parameters:**
- `vocab_size`: Size of the vocabulary (default: 152064)
- `hidden_size`: Size of hidden layers (default: 2048)
- `num_hidden_layers`: Number of hidden layers (default: 32)
- `num_attention_heads`: Number of attention heads (default: 32)
- `max_position_embeddings`: Maximum position embeddings (default: 32768)
- Additional parameters for vision and multimodal components

### `Qwen3VLModel`
Main Qwen3-VL model implementation.

**Methods:**
- `forward(input_ids, attention_mask=None, position_ids=None, pixel_values=None)`: Forward pass
- `generate(input_ids, max_length=512, temperature=1.0, do_sample=True, ...)`: Text generation

## Configuration

### `ConfigFactory`
Factory class for creating configuration instances.

### `ConfigValidator`
Class for validating configuration objects.

## Models

### `get_model(config=None, pretrained_model_name_or_path=None)`
Factory function to get a Qwen3-VL model instance.

## Components

### `AdapterConfig`
Configuration for adapter layers.

### `AdapterLayer`
Implementation of adapter layers for parameter-efficient fine-tuning.

## Inference

### `generate_text(model, input_ids, max_length=512, ...)`
Generate text using the Qwen3-VL model.