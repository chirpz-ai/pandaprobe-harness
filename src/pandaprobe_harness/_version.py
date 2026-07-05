"""Single source of truth for the package version.

Bumping ``__version__`` on ``main`` triggers the release workflow
(.github/workflows/release.yml): CI gates, then the package is tagged,
published to GitHub Releases, and uploaded to PyPI.
"""

__version__ = "0.6.1"
