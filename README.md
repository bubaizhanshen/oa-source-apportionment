# Real-time organic aerosol source estimation

Core analysis code for comparing PMF-trained machine learning with historical-profile fitting and rolling PMF.

- `src/oa_realtime/pmf.py`: ME-2 windows, runs, solution selection and factor matching.
- `src/oa_realtime/models.py`: fixed-profile and weighted elastic-net fitting, and seven ML regressors.
- `src/oa_realtime/transformer.py`: causal Transformer and spectral history.
- `src/oa_realtime/evaluation.py`: two-hour sampling, agreement, day-weighted error, single-observation timing, and hardware details.
- `src/oa_realtime/config.py`: selected Barcelona autumn model settings.

Install the dependencies in `requirements.txt` and add `src` to `PYTHONPATH`. The PMF interface requires a separately obtained EPA PMF 5.0 ME-2 runtime and Wine. Supply the spectra, uncertainties, reference profiles, and historical PMF contributions locally. The supplied settings describe the Barcelona autumn experiment; other periods use their own historical model settings.

The weighted elastic-net comparison uses the settings of an earlier CASA fitting function: [Gherras et al.](https://doi.org/10.1021/acsestair.6c00094). Provide normalized historical profiles and the aligned mass-to-charge channels to `fit_weighted_elastic_net`.
