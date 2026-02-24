"""
Patch transformers to support dinov3_vit model type in older versions.

This patch registers dinov3_vit to use dinov2 classes, which should be compatible
since DinoV3 is based on DinoV2 architecture.
"""
import warnings

try:
    # Import transformers first to ensure modules are loaded
    import transformers
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING, MODEL_NAMES_MAPPING, CONFIG_MAPPING_NAMES
    from transformers.models.dinov2 import Dinov2Config, Dinov2Model
    
    # Only patch if dinov3_vit is not already registered
    if 'dinov3_vit' not in CONFIG_MAPPING:
        # Use _extra_content for _LazyConfigMapping
        CONFIG_MAPPING._extra_content['dinov3_vit'] = Dinov2Config
        
        # Also add to CONFIG_MAPPING_NAMES and MODEL_NAMES_MAPPING
        CONFIG_MAPPING_NAMES['dinov3_vit'] = 'Dinov2Config'
        MODEL_NAMES_MAPPING['dinov3_vit'] = 'Dinov2Model'
        
        # Also register in MODEL_MAPPING if available
        try:
            from transformers.models.auto.modeling_auto import MODEL_MAPPING
            if hasattr(MODEL_MAPPING, '_extra_content'):
                MODEL_MAPPING._extra_content['dinov3_vit'] = (Dinov2Config, Dinov2Model)
            elif hasattr(MODEL_MAPPING, 'register'):
                MODEL_MAPPING.register('dinov3_vit', (Dinov2Config, Dinov2Model))
        except (ImportError, AttributeError):
            pass  # MODEL_MAPPING might not exist in older versions
        
        # Monkey-patch AutoModel.from_pretrained to automatically add ignore_mismatched_sizes
        # for dinov3 models to handle shape mismatches
        try:
            from transformers import AutoModel
            original_from_pretrained = AutoModel.from_pretrained
            
            @classmethod
            def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                # Check if this is a dinov3 model
                if isinstance(pretrained_model_name_or_path, str) and 'dinov3' in pretrained_model_name_or_path.lower():
                    # Add ignore_mismatched_sizes if not explicitly set
                    if 'ignore_mismatched_sizes' not in kwargs:
                        kwargs['ignore_mismatched_sizes'] = True
                return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            
            AutoModel.from_pretrained = patched_from_pretrained
        except Exception:
            pass  # If monkey-patching fails, continue anyway
        
        warnings.warn(
            "Patched transformers to support dinov3_vit using dinov2 classes. "
            "Consider upgrading transformers to >=4.47.0 for native support.",
            UserWarning
        )
except ImportError as e:
    warnings.warn(f"Could not patch transformers: {e}", UserWarning)
