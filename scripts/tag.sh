#!/usr/bin/env bash
set -euo pipefail

# 🧠 Ensure we're on the main branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$current_branch" != "main" ]]; then
  echo "You're on branch '$current_branch'. Please switch to 'main'."
  exit 1
fi

# 🔄 Ensure local main matches remote
git fetch origin main
if ! git diff --quiet HEAD origin/main; then
  echo "Local 'main' is not in sync with 'origin/main'."
  exit 1
fi

# 🧼 Ensure working tree is clean
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "You have uncommitted changes. Commit or stash them before tagging."
  exit 1
fi

# 🔢 Extract version
version=$(grep "version\s*=" pyproject.toml | cut -d '"' -f2)

# ✅ Validate semantic version
if ! [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Version '$version' is not semantic (e.g., 1.2.3)."
  exit 1
fi

# 🏷 Check tag absence
if git rev-parse "$version" >/dev/null 2>&1; then
  echo "Git tag '$version' already exists."
  exit 1
fi

# 🏷 Create and push tag
echo "Tagging version $version..."
git tag "$version"
git push origin "$version"
echo "Successfully tagged version $version"
