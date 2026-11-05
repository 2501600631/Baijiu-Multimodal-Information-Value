# Measurement and result dictionary

The observational unit is one author-confirmed independent bottle. There are 139 matched bottle identities. All modalities and reference concentrations are joined by `sample_id`, never by an assumed row position. Three technical measurements were averaged per bottle and modality; technical scans are not independent bottles and are not supplied individually.

## Analysis-ready data

`executed/data/chemistry.csv` contains `sample_id` followed by eleven concentration columns, in this fixed order: acetaldehyde, methanol, n_propanol, ethyl_acetate, isobutanol, acetal, n_butanol, isoamyl_alcohol, ethyl_butyrate, ethyl_lactate and ethyl_hexanoate. The unit is the source-reported g/L. The last identifier is provisional pending confirmation of an inconsistent source label. No primary constituent is privileged.

Each `spectra_*.npz` contains `sample_id`, `axis`, and `spectra`. The spectrum array has shape bottle by coordinate. UV-visible has 911 coordinates from 190 to 1100, step 1; Raman has 1019 coordinates from 21.6323 to 3002.1052 with nonuniform spacing; NIR has 1001 coordinates from 4000 to 12000, step 8. UV-visible coordinates are consistent with wavelength in nm. Native Raman and NIR units must be confirmed from acquisition records; physical-axis labels must not be inferred from numerical range alone.

`outer_splits.json` stores the historical three-repeat, five-fold bottle partitions. No outer test label is used for preprocessing, model fitting or inner selection. IDs are identifiers, not predictor variables.

## English measurement workbooks

The four English workbooks preserve the supplied table layout and all numeric cells, numeric text, blank cells and bottle identities. Human-readable labels are translated; workbook author metadata is replaced with neutral export metadata. The original workbook hashes, translated-cell counts and value/type equality checks are in `provenance/english_export_validation.json`.

The chemistry workbook includes source metadata such as grade, alcohol content, weight and ratios. Those metadata are preserved for source completeness but are not used as predictors, labels for subgroup selection or model selection inputs. Units and meanings of auxiliary source coefficients are not established by this repository. The source category `jiu er` is retained as a transliterated category rather than interpreted as a new quantitative variable.

## Prediction records

`stage`, `condition`, `training_size`, `repeat`, `fold` and `bootstrap_refit` identify the computation. `sample_id`, `target` and `model` identify a held-out prediction. `y_true` and `y_pred` are source-unit concentration and prediction; `train_sd` is the receiving-fit response SD. `outer_train_sd` is the complete outer-training response SD used only for common-scale learning-curve assessment. `n_train` counts fit rows and `n_unique_train` counts unique fit bottles. They can differ for bootstrap refits. `n_cal` is zero in every included point-prediction task.

Task caches contain the selected parameter and preprocessing records, inner candidate scores, fit/test bottle roles and payload hashes. Raw CSV streams are byte-preserved after gzip decompression. Root derived selection records are filtered by included stage only; per-stage derived tables retain their saved values.

## Strategies and metrics

`uv`, `raman` and `nir` are single-modality inputs; combinations joined with `+` are concatenated or block-modelled inputs. `select_single` and `select_subset` choose per target using inner validation. `panel_t000`, `panel_t002`, `panel_t005`, `panel_t010` choose one common mask for all eleven targets, preferring fewer modalities within the declared inner standardized-loss tolerance.

RMSE, MAE and bias retain g/L. R2 retains negative values. Macro standardized MSE averages squared errors normalized by the corresponding receiving-training response variance and then weights each of the eleven targets equally. Learning tables with `common_scale` use complete outer-training SD held fixed across sizes. A positive RMSE gain means lower candidate error. Refit minimum/maximum columns describe five observed refit-index groups, not a confidence level.
