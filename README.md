# Policy Analytics Assessment (IDB)

Base repository for the Data Scientist Technical Assessment (IDB).  
Provides a reproducible structure, Conda environment (Python 3.12), and a bootstrap script to create/update the environment.

## Quick start

```bash
# create or update the environment
bash run.sh

# activate it
conda activate policy-analytics

# optional: register Jupyter kernel
python -m ipykernel install --user --name policy-analytics --display-name "policy-analytics (py312)"
```

## Notes
- Raw data is not versioned. Use external links and local paths (see `data_links.txt`).
- Place your code in `code/` and write outputs to `outputs/`.
- Documentation and slides should be stored in `docs/`.
