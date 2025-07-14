# syntax=docker/dockerfile:1.4
FROM python:3.11.3

LABEL name="astabench" \
    org="allenai" \
    description="Benchmark and experimentation framework for scientific agents"

RUN apt-get update && apt-get install -y less git curl cmake clang

RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz -o docker.tgz \
    && tar xzvf docker.tgz --strip 1 -C /usr/local/bin docker/docker \
    && rm docker.tgz

RUN mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -SL https://github.com/docker/compose/releases/download/v2.29.6/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

# install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"
# Don't let uv link system packages
ENV UV_LINK_MODE=copy

# install cargo:
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Only copy pyproject.toml; dependencies can be installed this way, but the docker
# cache won't depend on any other files
WORKDIR /astabench
COPY pyproject.toml /astabench/

# Create a basic directory structure for the package
RUN mkdir -p /astabench/astabench

# Split up `sync` commands to avoid conflict errors; the conflicts are the
# result of packages pinning more specific dependency versions than they
# actually require
# Note: all these extras are convenience for solvers/dev; they aren't needed to
# run evals with basic agents like `generate`
RUN uv sync --extra dev --extra smolagents --extra azure --extra sqa --extra super --extra datavoyager_solver

COPY . /astabench

# Get astabench registry synced, but don't uninstall the stuff from before
RUN uv sync --inexact

COPY docker/bashrc /root/.bashrc
