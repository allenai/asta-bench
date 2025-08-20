# Development

Instructions for developers working on the project.

## Publishing a new version

  1. Increment the version number in `pyproject.toml`
  2. Run `make tag` to push the new version tag to GitHub
  3. Go to https://github.com/allenai/asta-bench/actions and run the `Publish to PyPI` workflow, using the version tag
