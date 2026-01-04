# Changelog

All notable changes to this project will be documented in this file.

## Unreleased (2026-01-04)

- Fix: Ecosim now requires a balanced Ecopath model and shows a clear error notification if an unbalanced `RpathParams` is present. This prevents runtime errors and clarifies required workflow (balance in Ecopath page before running Ecosim). 🔧
- Fix: Preserve explicit zero inputs in Ecopath parameter edits — blank strings and None are treated as `NaN`, while `'0'` and `0` are preserved as numeric zero. Added unit tests for both behaviors. ✅


## 0.2.2 - 2025-12-XX

- Initial release notes and previous changes (see commit history).
