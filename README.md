# Model Summary

This directory summarizes three representative model integrations used in SURE-EVAL phase-1 onboarding:

- **FireRedVAD**: voice activity detection / audio event detection
- **Qwen2-VL-2B-Instruct**: vision-language understanding
- **WeSpeaker**: speaker verification

## Overview

These three models cover different multimodal/audio-related tasks:

| Model | Task | Phase-1 Status | Notes |
|------|------|----------------|------|
| FireRedVAD | VAD / AED | Passed | Supports streaming and non-streaming VAD, plus AED |
| Qwen2-VL-2B-Instruct | VLM | Passed | Validated on minimal local image-text understanding |
| WeSpeaker | Speaker Verification | Failed | Blocked at import stage due to upstream dependency issues |

## Model Notes

### FireRedVAD
FireRedVAD is an industrial-grade VAD/AED model supporting:
- non-streaming VAD
- streaming VAD
- non-streaming AED

It is designed for multilingual speech/singing/music detection and provides both command-line and Python API usage.

### Qwen2-VL-2B-Instruct
Qwen2-VL-2B-Instruct is a local vision-language model integrated for minimal image understanding in SURE-EVAL.  
Phase-1 successfully validated:
- import
- model load
- single-image inference
- JSON output contract

The validated output format is:

```json
{"text": "<non-empty string>"}
