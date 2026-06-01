# Passive Partner ML Results

Generated UTC: 2026-06-01T19:37:35.415879+00:00

## Modeling choices

- Data source: Training_Eligible sheet from each passive partner model workbook.
- Target: event-level passive-financial-partner flag and partner-level passive-financial flag.
- Feature policy: strict tabular features only; IDs, URLs, source fields, derived outcome evidence, and near-label text fields are excluded.
- Models compared: Dummy prior baseline, balanced logistic regression, balanced random forest, balanced extra trees, and balanced histogram gradient boosting.
- Primary selection metric: mean cross-validated average precision, suitable for rare positive labels.

## Best model summary

| Dataset | Rows | Positives | Positive rate | Best model | CV avg precision | Holdout avg precision | Holdout F1 | Holdout recall |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| event_level | 438 | 46 | 0.105 | logistic_regression_balanced | 0.536 | 0.716 | 0.640 | 0.667 |
| partner_level | 1042 | 48 | 0.046 | random_forest_balanced | 0.297 | 0.440 | 0.385 | 0.417 |

## Graphs

PNG charts are saved under `graphs/` and are regenerated on every run.

- graphs/overall_best_model_summary.png
- graphs/event_level_class_balance.png
- graphs/event_level_cv_model_comparison.png
- graphs/event_level_holdout_model_comparison.png
- graphs/event_level_precision_recall_curve.png
- graphs/event_level_roc_curve.png
- graphs/event_level_best_model_confusion_matrix.png
- graphs/event_level_feature_importance.png
- graphs/partner_level_class_balance.png
- graphs/partner_level_cv_model_comparison.png
- graphs/partner_level_holdout_model_comparison.png
- graphs/partner_level_precision_recall_curve.png
- graphs/partner_level_roc_curve.png
- graphs/partner_level_best_model_confusion_matrix.png
- graphs/partner_level_feature_importance.png

## Files

- model_summary.csv: compact comparison and chosen model for each dataset.
- *_cv_metrics.csv: stratified cross-validation metrics by model.
- *_holdout_metrics.csv: stratified holdout metrics by model.
- *_holdout_predictions.csv: holdout predictions and scores by model.
- *_feature_importance.csv: permutation importance for the selected model on the holdout split.
- *_best_model.joblib: final selected pipeline retrained on all eligible rows.
- graphs/*.png: visual summaries, model comparisons, curves, confusion matrices, and feature importance charts.
- run_metadata.json: Python, package, feature, and runtime details.

## Caution

The positive class is rare in both datasets. Treat holdout results as directional, and prefer average precision, recall, and precision over plain accuracy.
