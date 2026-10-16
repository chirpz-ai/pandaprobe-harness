# PandaProbe Harness arXiv preprint

This directory contains the branded, single-column technical preprint for
PandaProbe Harness. It presents the system and benchmark study for agent
builders, researchers, and the open-source community, with practical
integration and reproducibility material alongside the technical mechanism.

## Build

The paper uses the standard `article` class and TeX Live packages only. It does
not require a downloaded conference style or shell escape.

```bash
cd paper/arxiv
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The same build can be run with TeX Live in Docker from the repository root:

```bash
docker run --rm \
  -v "$PWD/paper/arxiv:/work" \
  -w /work texlive/texlive:latest sh -lc \
  'pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
   bibtex main && \
   pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
   pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

Both commands write the PDF to `paper/arxiv/main.pdf`.

`figures/overview.tex` presents the system architecture and evidence-gated
self-heal loop. `paper_tables.md` contains the benchmark tables used
to update the report.

## Project links

- PandaProbe platform: <https://github.com/chirpz-ai/pandaprobe>
- PandaProbe Harness: <https://github.com/chirpz-ai/pandaprobe-harness>
- Documentation: <https://docs.pandaprobe.com/harness>

## arXiv submission notes

- Primary category: `cs.AI`.
- Candidate cross-lists: `cs.SE`, `cs.MA`, and `cs.LG`.
- Upload `main.tex`, `references.bib`, `figures/overview.tex`,
  `logo/android-chrome-512x512.png`, and the generated `main.bbl`.
- The PandaProbe Harness, Chirpz AI Lab, and author attribution are intentional
  and must remain onymized.
- Inspect the final PDF for table overflow, figure legibility, URL wrapping,
  and accessibility of plots.

## Author TODOs before posting

- Confirm the exact spelling of **Sina Tayebati**, ORCID, and any co-authors.
- Check that every result in `main.tex` agrees with `paper_tables.md`.
- Complete the two methodological checkpoints and report null outcomes.
- Verify every reference, package version, model matrix, command, and public
  link at submission time.
- Recheck the integration-verification status recorded in
  `../../benchmarks/IMPLEMENTATION_NOTES.md` after full orchestration runs.
- Rebuild from a clean directory and confirm there are no undefined citations,
  references, missing files, or overfull boxes.

## Availability and reproducibility

PandaProbe Harness is MIT-licensed and maintained by PandaProbe and
Chirpz AI Lab. Study
commands and outputs are documented in `../../benchmarks/README.md` and
`../../benchmarks/RUNNING.md`; implementation maturity and deviations are
recorded in `../../benchmarks/IMPLEMENTATION_NOTES.md`.
