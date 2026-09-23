# Security policy

## Scope

This repository contains teaching notebooks, generated slide decks and a static website. It holds **no credentials, no private data and no production services**. Every dataset is public and historical. The notebooks that deploy to the cloud walk through the provider's web console; they never embed tokens.

## What we protect against

- **Leaked secrets.** Every push and pull request is scanned with gitleaks across the full git history. A static check (`scripts/check_notebooks.py`) also rejects notebooks whose code, outputs or metadata match common key and token patterns, or contain absolute local paths.
- **Supply chain.** Runtime dependencies are declared in `requirements.txt` and maintainer tooling in `requirements-dev.txt`. Dependabot proposes monthly updates for Python packages and GitHub Actions. Workflows run with read-only tokens except the Pages deployment, which uses the GitHub-managed OIDC flow.
- **Student safety.** Notebooks instruct students never to paste tokens into cells. The local API examples bind to `127.0.0.1` only.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting on this repository (**Security > Report a vulnerability**). Please include the notebook or file, the commit, and steps to reproduce. You will get an acknowledgement within a week. Please do not open a public issue for anything that could expose a student.

## Supported versions

Only the `main` branch is maintained. Older commits are kept for history and are not patched.
