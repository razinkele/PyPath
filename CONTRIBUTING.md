Contributing to PyPath

Thanks for contributing! A few quick guidelines to make PRs smooth and reviewable.

- Use the provided PR template (.github/PULL_REQUEST_TEMPLATE.md) when opening pull requests.
- Add or update tests for any behavior changes you introduce and run the test suite locally (pytest -q).
- Keep changes small and focused or document them clearly in the PR when larger.

Shinyswatch (Theme) guidance

- shinyswatch is optional for local development. The app has a graceful fallback when shinyswatch is not installed, so you can run the app or tests without it.
- To develop or test theme-related UI (theme picker, `shinyswatch.theme` usage), install the package in your environment:

  pip install shinyswatch

- We run a CI smoke job that installs `shinyswatch` and executes a minimal Shiny integration test: `.github/workflows/ci-shiny-smoke.yml` (this verifies the app imports and the theme picker utilities are available).
- If you add or modify theme-related behavior (e.g., theme picker UI), add or update `tests/test_shinyswatch_integration.py` to cover it.

If you're unsure about any changes or what tests to add, open an issue or ask for guidance on the PR — we're happy to help.