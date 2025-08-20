# Development

Instructions for developers working on the project.

## Publishing a new version

  1. Increment the version number in `pyproject.toml`
  2. Run `make tag` to push the new version tag to GitHub
  3. Go to https://github.com/allenai/asta-bench/actions and run the `Publish to PyPI` workflow, using the version tag

## Updating git dependencies

PyPI does not allow git dependencies to be specified in `pyproject.toml`. 
We depend on specific git shas of some dependencies, which we publish to our own pypi account

### Updating inspect_evals

  1. Update the git ref in the checkout step of [publish-inspect-evals.yml](.github/workflows/publish-inspect-evals.yml).
  2. Check out that git ref locally and copy its `pyproject.toml` to [.github/workflows/inspect-evals-pyproject.toml](.github/workflows/inspect-evals-pyproject.toml).
  3. Change the `name = "..."` in `inspect-evals-pyproject.toml` to `astabench-inspect_evals`
  3. Bump the `version = "..."` in `inspect-evals-pyproject.toml` to the current version of astabench 
  4. Go to https://github.com/allenai/asta-bench/actions and run the "Publish inspect_evals to PyPI" workflow

### Updating knowledge-storm
  1. Update the git ref in the checkout step of [publish-knowledge-storm.yml](.github/workflows/publish-knowledge-storm.yml).
  2. Check out that git ref locally and copy its `setup.py` to [.github/workflows/knowledge-storm-setup.py](.github/workflows/knowledge-storm-setup.py).
  3. Change the `name="..."` in `knowledge-storm-setup.py` to `astabench-knowledge-storm`
  3. Bump the `version="..."` in `knowledge-storm-setup.py` to the current version of astabench
  4. Go to https://github.com/allenai/asta-bench/actions and run the "Publish knowledge-storm to PyPI" workflow

