.PHONY: shell format mypy flake test build-image test-expensive

# allow passing extra pytest args, e.g. make test-expensive PYTEST_ARGS="-k EVAL_NAME"
PYTEST_ARGS ?=

ASTABENCH_TAG    := astabench
CONTAINER_NAME   := astabench-container
DOCKER_SOCKET_PATH ?= $(if $(XDG_RUNTIME_DIR),$(XDG_RUNTIME_DIR)/docker.sock,/var/run/docker.sock)

# -----------------------------------------------------------------------------
# Local vs CI environment vars
# -----------------------------------------------------------------------------
ifeq ($(IS_CI),true)
  LOCAL_MOUNTS :=
  ENV_ARGS := -e OPENAI_API_KEY -e HF_TOKEN -e IS_CI
  TEST_RUN := docker run --rm $(ENV_ARGS) -v /var/run/docker.sock:/var/run/docker.sock $(ASTABENCH_TAG)
  BUILD_QUIET := --quiet
else
  LOCAL_MOUNTS := \
    -v $(DOCKER_SOCKET_PATH):/var/run/docker.sock \
    -v $$(pwd)/pyproject.toml:/astabench/pyproject.toml:ro \
    -v $$(pwd)/astabench:/astabench/astabench \
    -v $$(pwd)/tests:/astabench/tests \
    -v $$(pwd)/logs:/astabench/logs \
    -v astabench-cache:/root/.cache
  ENV_ARGS := --env-file .env
  TEST_RUN := docker run --rm $(ENV_ARGS) $(LOCAL_MOUNTS) $(ASTABENCH_TAG)
  BUILD_QUIET ?=
endif

# -----------------------------------------------------------------------------
# Build the Docker image (primary target)
# -----------------------------------------------------------------------------
build-image:
	docker build $(BUILD_QUIET) . --tag $(ASTABENCH_TAG) -f ./docker/Dockerfile

# -----------------------------------------------------------------------------
# Interactive shell in container
# -----------------------------------------------------------------------------
shell: build-image
	@docker run --rm -it --name $(CONTAINER_NAME) \
		$(LOCAL_MOUNTS) \
		-v astabench-home:/root/.astabench \
		$(ENV_ARGS) -p 7575:7575 \
		$(ASTABENCH_TAG) \
		/bin/bash

# -----------------------------------------------------------------------------
#  Formatting and linting
# -----------------------------------------------------------------------------
# NOTE: These commands aim to install only the dev dependencies, without the
# main package depedencies which require more complex setup, e.g., ~ssh.
# Ideally they would install the exact lib versions of the dev dependencies,
# but limiting to only dev dependencies in pyproject.toml in a DRY manner is not
# easy to do since pip has no mechanism and uv requires defining a seeparate section
# in pyproject.toml which pip cannot read.

ifneq ($(IS_CI),true)
format: build-image
endif

format:
	docker run --rm \
		-v $$(pwd):/astabench \
		$(ASTABENCH_TAG) \
		sh -c "pip install --no-cache-dir black && black ."

ifneq ($(IS_CI),true)
mypy: build-image
endif

mypy:
	docker run --rm \
		-v $$(pwd):/astabench \
		$(ASTABENCH_TAG) \
		uv run mypy astabench/ tests/

ifneq ($(IS_CI),true)
flake: build-image
endif

flake:
	docker run --rm \
		-v $$(pwd):/astabench \
		$(ASTABENCH_TAG) \
		uv run flake8 astabench/ tests/

ifneq ($(IS_CI),true)
test: build-image
endif

test:
	@$(TEST_RUN) uv run --no-sync --extra dev --extra inspect_evals --extra smolagents \
		-m pytest $(PYTEST_ARGS) -vv /astabench/tests

ifneq ($(IS_CI),true)
test-expensive: build-image
endif

test-expensive:
	@$(TEST_RUN) uv run --no-sync --extra dev --extra inspect_evals --extra smolagents \
		-m pytest $(PYTEST_ARGS) -vv -o addopts= -m expensive /astabench/tests
