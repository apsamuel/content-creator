# `calibrate` Command

Compare old (LLM-only) and new (ensemble-enhanced) preclassification outputs for model calibration and validation.

## Purpose

The `calibrate` command enables you to:
- **Validate ensemble improvements** by comparing how the new ensemble preclassification differs from the old LLM-only method
- **Tune risk thresholds** by understanding which signals changed and by how much
- **Audit model behavior** by side-by-side comparison of risk scores, risk levels, and visual intensity recommendations
- **Track quality metrics** across runs to measure the impact of ensemble enhancements

## Usage

### Basic Usage

```bash
content-creator calibrate \
  --manifest-old <path-to-old-manifest.json> \
  --manifest-new <path-to-new-manifest.json>
```

### Output to File

```bash
content-creator calibrate \
  --manifest-old output/cleaned/manifest.json \
  --manifest-new output/cleaned-new/manifest.json \
  --output calibration-report.json
```

## Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--manifest-old` | Path | ✅ Yes | Path to baseline manifest (old/LLM-only preclassification) |
| `--manifest-new` | Path | ✅ Yes | Path to new manifest (ensemble-enhanced preclassification) |
| `--output` | Path | ❌ No | Save report as JSON file; prints to stdout if omitted |

## Output

### Console Output

The command prints a summary to stdout showing key changes:

```
✅ Calibration report written to: calibration-report.json

📊 Calibration Summary:
  Mood: Hopeful → Hopeful
  Risk Score: 0.35 ↑ 0.52 (Δ=+0.170)
  Risk Level: Low → Medium
  Visual Intensity: balanced → expressive
  Signals: 5 → 6
```

### JSON Report Format

When saved to a file, the report includes detailed metrics:

```json
{
  "calibration_report": true,
  "old_manifest": "output/cleaned/manifest.json",
  "new_manifest": "output/cleaned-new/manifest.json",
  "comparison": {
    "mood": {
      "old": "Hopeful",
      "new": "Hopeful",
      "changed": false
    },
    "truthfulness": {
      "old": "MixedOrUnverifiable",
      "new": "MixedOrUnverifiable",
      "old_confidence": 0.58,
      "new_confidence": 0.58
    },
    "ensemble_scoring": {
      "old_risk_score": 0.35,
      "new_risk_score": 0.52,
      "risk_score_delta": 0.17,
      "old_risk_level": "Low",
      "new_risk_level": "Medium",
      "risk_level_changed": true,
      "old_visual_intensity": "balanced",
      "new_visual_intensity": "expressive",
      "visual_intensity_changed": true
    },
    "signal_count": {
      "old_signals": 5,
      "new_signals": 6
    }
  },
  "new_warnings": [
    "Secondary safety model signal unavailable."
  ]
}
```

## Interpretation Guide

### Risk Score Delta

- **Positive (+)**: Ensemble detected higher risk than LLM-only method
- **Negative (-)**: Ensemble detected lower risk (more permissive)
- **Zero**: No change in overall risk assessment

### Risk Level Changes

- `Low → Medium` or `Low → High`: Ensemble flagged content as riskier
- `High → Medium` or `High → Low`: Ensemble flagged content as safer
- No change: Ensemble agrees with LLM baseline

### Visual Intensity Recommendations

Changes in intensity guide artistic style adaptation:
- `restrained → balanced/expressive/vivid`: Use more visual energy/saturation
- `vivid → balanced/restrained`: Use more subdued colors/motion
- No change: Consistent artistic direction

### Signal Changes

- **Increased signal count**: Ensemble had more available models (fewer failures)
- **Decreased signal count**: Some models were unavailable during scoring

## Practical Workflow

1. **Generate baseline** with `from-audio` or `from-text` (with old code)
2. **Generate new version** with ensemble-enhanced code
3. **Run calibration** to compare outputs
4. **Review risk score deltas** to understand ensemble impact
5. **Adjust config** (e.g., model weights, thresholds) if needed
6. **Iterate** and re-calibrate

## Examples

### Compare Single Run

```bash
content-creator calibrate \
  --manifest-old output/baseline/manifest.json \
  --manifest-new output/with-ensemble/manifest.json
```

### Save Multi-Run Comparison

```bash
for i in 1 2 3; do
  content-creator calibrate \
    --manifest-old "runs/baseline_run_$i/manifest.json" \
    --manifest-new "runs/ensemble_run_$i/manifest.json" \
    --output "reports/calibration_run_$i.json"
done
```

## Related Commands

- [`from-audio`](from-audio.md) — Generate video from audio with preclassification
- [`from-text`](from-text.md) — Generate video from text with preclassification
- [`doctor`](doctor.md) — Diagnose system configuration and model availability
