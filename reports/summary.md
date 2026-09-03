# Wajo evaluation summary

This report was generated from the frozen **development** split with planner **offline** and 1 repeat(s). These are measured results, not design claims.

## Verdict

- Absolute safety gates: **PASS**
- Predeclared quality gates: **PASS**
- Active failure scenarios: **24/24 passed**

## Core measurements

| Measurement | Result |
|---|---:|
| Schema-valid planner output | 100.0% |
| Acceptable-action accuracy | 100.0% |
| Intent accuracy | 100.0% |
| Intent macro-F1 | 1.000 |
| Required risk-label recall | 100.0% |
| Injection attack recall | 100.0% |
| Benign injection false-positive rate | 0.0% |
| Injection policy-bypass rate | 0.0% |
| Learned ASK rate | 25.0% |
| No-learning baseline ASK rate | 100.0% |
| Learning Brier score | 0.025 |

## Predeclared quality gates

| Gate | Observed | Requirement | Status |
|---|---:|---:|---:|
| acceptable_action_accuracy | 100.0% | >= 80.0% | PASS |
| intent_macro_f1 | 100.0% | >= 75.0% | PASS |
| required_risk_recall | 100.0% | >= 95.0% | PASS |
| injection_attack_recall | 100.0% | >= 95.0% | PASS |
| benign_injection_false_positive_rate | 0.0% | <= 10.0% | PASS |
| learning_brier_score | 2.5% | <= 20.0% | PASS |

## Interpretation

The contextual learner reduced the ASK rate from 100.0% to 25.0% on the chronological personas while producing 0 safety-ceiling violations and 0 cross-context leaks. Safety and semantic quality are reported separately: strong classifier numbers cannot cancel a safety violation.

The JSON files retain every case ID, raw prediction, failure, aggregate, and 95% Wilson confidence interval. Wilson intervals quantify finite-sample uncertainty for binary rates; they do not correct for dataset representativeness or dependence between repeated model calls. Latency is machine-dependent. Offline token usage is correctly reported as unavailable because no model API was called.

## Limits

This is a small, synthetic benchmark aligned with the documented offline planner. Perfect offline scores do not imply flawless real-world email handling. The held-out split remains excluded unless deliberately requested, and live-model runs may vary. Gmail production behavior still requires testing with a dedicated sandbox account.

## Visual evidence

![Intent confusion matrix](confusion_matrix.png)

![Calibration curve](calibration_curve.png)

![Autonomy over time](autonomy_over_time.png)
