# Real-time organic aerosol source estimation

Core analysis code for comparing PMF-trained machine learning with historical-profile fitting and rolling PMF.

- `src/oa_realtime/pmf.py`: ME-2 windows, runs, solution selection and factor matching.
- `src/oa_realtime/models.py`: fixed-profile and weighted elastic-net fitting, and seven ML regressors.
- `src/oa_realtime/transformer.py`: causal Transformer and spectral history.
- `src/oa_realtime/evaluation.py`: two-hour sampling, agreement, day-weighted error, single-observation timing, and hardware details.
- `src/oa_realtime/source_response.py`: source-spectrum training additions, source-increment errors, and source-informed PMF constraint centers.
- `src/oa_realtime/config.py`: selected Barcelona autumn model settings.

Install the dependencies in `requirements.txt` and add `src` to `PYTHONPATH`. The PMF interface requires a separately obtained EPA PMF 5.0 ME-2 runtime and Wine. Supply the spectra, uncertainties, reference profiles, and historical PMF contributions locally. The supplied settings describe the Barcelona autumn experiment; other periods use their own historical model settings.

The weighted elastic-net comparison uses the settings of an earlier CASA fitting function: [Gherras et al.](https://doi.org/10.1021/acsestair.6c00094). Provide normalized historical profiles and the aligned mass-to-charge channels to `fit_weighted_elastic_net`.

For signal-normalized models, `fit_normalized_trees` and `predict_normalized_trees` divide each spectrum and its historical PMF contributions by the spectrum's positive signal sum, then rescale the prediction. The corresponding regressor functions use the same transformation. If uncertainties are supplied, they are divided by the same signal sum. `prepare_history_constrained_run` prepares a continuation around historical PMF profiles; `source_total_error` evaluates estimates against known or explicitly specified reference contributions.

`NearestShapeRegressor` estimates contributions from the five closest normalized historical spectra. `TabICLSourceRegressor` evaluates the pretrained TabICLv2 model with either direct or signal-normalized inputs; its optional quantiles are not calibrated prediction intervals.

`SourceScalingRegressor` compares raw or normalized inputs, scaled contributions, and optional signal-squared sample weights using a supplied estimator.

`SequentialInterval` adjusts later intervals using a reference only after it becomes available; it evaluates agreement with the supplied reference.

`fit_profile_regularized` combines current-spectrum fitting with existing source estimates, including grouped factors.

`match_profiles` aligns current factors with historical profiles for correspondence sensitivity comparisons.

For the study-held-out library comparison, `replace_source_profiles` changes source spectra within existing factor and intensity strata, and `augment_with_assignment` rebuilds the corresponding training data. `expanded_constraint_centers` combines the original and added source-category means with historical profiles; `prepare_source_constrained_run` applies these bounds to matched PMF factors while leaving other factors free.

`add_source_to_window` applies a source addition to consecutive observations and optionally changes the corresponding uncertainty weights; `source_increment` scores the resulting factor response.

`source_addition_reference` combines background factor estimates with a known source addition. `source_total_error` compares the complete estimate with that conditional reference.

`source_constraint_envelope` retains the range of available source spectra; `prepare_source_bounded_run` applies its channelwise bounds to a PMF continuation.

`leave_one_study_out_selection` screens fixed methods using source-increment errors from the other studies and reports each held-out comparison.

Scaling each spectrum, its uncertainties, and its contributions together leaves the standard weighted PMF residuals unchanged when constraints are compatible.
