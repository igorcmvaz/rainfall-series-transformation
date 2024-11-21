# Rainfall Series Transformation
Transformation of Brazilian '.nc' precipitation files (such as the ones found at [CLIMBra - Climate Change Dataset for Brazil](https://www.scidb.cn/en/detail?dataSetId=609b7ff93f0d4d1a9ba6eb709027c6ad), under `Gridded data/pr`") into location and model specific output files.

## Setup
```bash
git clone git@github.com:igorcmvaz/rainfall-series-transformation.git
cd rainfall-series-transformation
python -m venv venv     # optional (but recommended) to create a virtual environment
python -m pip install -r requirements.txt
cd src
```

# Commits
When committing to this repository, following convention is advised:

* chore: regular maintenance unrelated to source code (dependencies, config, etc)
* docs: updates to any documentation
* feat: new features
* fix: bug fixes
* ref: refactored code (no new feature or bug fix)
* revert: reverts on previous commits
* test: updates to tests

For further reference on writing good commit messages, see [Conventional Commits](https://www.conventionalcommits.org).
