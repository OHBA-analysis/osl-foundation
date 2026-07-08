Sign Flipping
-------------

The training data for MEG-GPT used sign flipped parcellated data (Glasser52). ``template_cov.npy`` contains the covariance for the template session. [osl-dynamics](https://github.com/OHBA-analysis/osl-dynamics) can be used to find the parcels to flip for a new session:

```
from osl_dynamics.data import sign_flipping

template_cov = np.load("/path/to/template_cov.npy")

data = ... # (n_parcels, n_samples)
flipped_data, flips, corr = sign_flipping.sign_flip(data, template_cov)
```

Also see this [tutorial](https://osl-dynamics.readthedocs.io/en/latest/tutorials_build/0-4_sign_flipping.html).
