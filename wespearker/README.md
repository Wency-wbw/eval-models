# WeSpeaker Model for SURE-EVAL

WeSpeaker speaker verification phase-1 onboarding record for SURE-EVAL.

## Model Information

- **Model**: [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker)
- **Runtime Target**: `wespeaker.load_model("english")`
- **Display Name**: `WeSpeaker-English-ResNet221_LM-SV`
- **Task**: Speaker Verification
- **Deployment**: Local
- **Phase-1 Status**: Failed at `VALIDATE_IMPORT`
- **Final Verdict**: See [verdict.json](./verdict.json)

## Phase-1 Target

This onboarding only targets the minimal repo-native speaker verification path:

```python
import wespeaker

model = wespeaker.load_model("english")
model.set_device("cpu")
score = model.compute_similarity(
    "tests/fixtures/shared/speaker_verification/spk1_enroll.wav",
    "tests/fixtures/shared/speaker_verification/spk1_trial.wav",
)
```

Out of scope for this phase:

- diarization quality validation
- speaker registration workflows
- ONNX runtime pipelines beyond import side effects
- PLDA or score calibration
- training or benchmark evaluation

## Fixture

Phase-1 uses the task-specific shared speaker verification fixtures:

- `tests/fixtures/shared/speaker_verification/spk1_enroll.wav`
- `tests/fixtures/shared/speaker_verification/spk1_trial.wav`
- `tests/fixtures/shared/speaker_verification/spk2_trial.wav` for optional negative sanity-check

All three files are local mono 16kHz PCM WAV files.

## What Was Tried

The executed workflow followed:

`DISCOVER -> CLASSIFY -> PLAN -> VALIDATE_SPEC -> BUILD_ENV -> VALIDATE_IMPORT -> DIAGNOSE -> REPLAN`

Environment/build choices:

- backend: `pip`
- actual Python: `3.10.20`
- model-local venv: `src/sure_eval/models/wespeaker/.venv`
- model-local cache target: `src/sure_eval/models/wespeaker/pretrained_models/wespeaker`

Targeted mitigations that were attempted:

1. Install `wespeaker` from git plus already diagnosed missing deps: `PyYAML`, `requests`, `onnxruntime`
2. Install `s3prl` after import failed on an eager optional frontend dependency
3. Pin `torch==2.1.2` and `torchaudio==2.1.2` after a torchaudio compatibility failure inside `s3prl`

## Why Phase-1 Failed

The failure is not a fixture problem. The blocker is the repo-native import chain:

1. `import wespeaker` eagerly imports `wespeaker.cli.speaker`
2. `wespeaker.cli.speaker` eagerly imports `wespeaker.frontend`
3. `wespeaker.frontend` eagerly imports multiple optional frontend families
4. Those frontends pull undeclared or unstable optional dependencies before the minimal english similarity path can even load

Observed import-stage failures in order:

1. `ModuleNotFoundError: No module named 's3prl'`
2. `AttributeError: module 'torchaudio' has no attribute 'set_audio_backend'`
3. `ModuleNotFoundError: No module named 'whisper'`

Because the same checkpoint kept exposing new optional frontend dependencies, the run was escalated instead of continuing blind dependency expansion.

## Key Findings

- Upstream README shows a development route based on `python=3.9` and `pip install -r requirements.txt`.
- The base package install path is thinner than the development route and does not declare all import-time dependencies exposed by current eager imports.
- `commit: null` increases volatility because newer frontend additions can widen the `import wespeaker` dependency surface.
- No weights were downloaded in this run because import never completed, so `wespeaker.load_model("english")` never started.


## Recommended Next Steps

Suggested next moves before retrying phase-1:

1. Pin a stable upstream commit instead of leaving `repo.commit` as `null`
2. Decide whether phase-1 should follow the upstream development install route instead of only package install
3. Decide whether to approve a recorded local patch that narrows eager frontend imports to the minimal speaker verification path

If a new run is attempted, the first checks should focus on:

- integration and import-path breadth
- dependency declaration gaps
- frontend dependency compatibility

Fixture mismatch should be considered a lower-priority hypothesis for this model.

