"""
Signatures Subpackage - Observable Signature Models

This subpackage contains signature models for defense eVTOL:

Unified Signature Model (signature_model.py):
- Integrated RCS, IR, and acoustic signatures
- Frequency-dependent detection modeling
- Threat characterization

Component Models (legacy, preserved for compatibility):

RCS Model (rcs.py):
- Radar cross-section as function of aspect angle
- Frequency-dependent behavior
- Rotor modulation effects

IR Model (infrared.py):
- Thermal emission modeling
- Engine/exhaust signatures
- Solar reflection

Acoustic Model (acoustic.py):
- Rotor noise (BVI, loading, broadband)
- Motor/inverter noise
- Propagation and attenuation

For defense operations, these signatures determine detectability:
    - Radar: Long-range detection, all-weather
    - IR: Medium-range, passive detection
    - Acoustic: Close-range, passive detection
"""

# Unified model (new)
try:
    from .signature_model import (
        VehicleSignatureModel,
        VehicleSignatureConfig,
        SignatureState,
        RadarSignature,
        InfraredSignature,
        AcousticSignature,
        RadarBand,
    )
except ImportError:
    pass

# Legacy models (backward compatibility)
try:
    from .rcs import RCSModel, RCSConfig, RCSState
    from .infrared import IRModel, IRConfig, IRState, IRBand
    from .acoustic import AcousticModel, AcousticConfig, AcousticState, NoiseSource
except ImportError:
    pass

__all__ = [
    # Unified model
    'VehicleSignatureModel',
    'VehicleSignatureConfig',
    'SignatureState',
    'RadarSignature',
    'InfraredSignature',
    'AcousticSignature',
    'RadarBand',
    # RCS/IR/Acoustic models
    'RCSState',
    'IRModel',
    'IRState',
    'AcousticModel',
    'AcousticState',
]
