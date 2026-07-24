# PandaProbe Harness arXiv preprint

This directory contains the branded, two-column technical preprint for
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

The same clean build can be run with TeX Live in Docker:

```bash
mkdir -p /tmp/pandaprobe-paper-arxiv
docker run --rm \
  -v "$PWD/paper/arxiv:/work:ro" \
  -v /tmp/pandaprobe-paper-arxiv:/out \
  -w /out texlive/texlive:latest sh -lc \
  'export TEXINPUTS=/work//:; export BIBINPUTS=/work:; \
   pdflatex -interaction=nonstopmode -halt-on-error /work/main.tex && \
   bibtex main && \
   pdflatex -interaction=nonstopmode -halt-on-error /work/main.tex && \
   pdflatex -interaction=nonstopmode -halt-on-error /work/main.tex'
```

`figures/overview.tex` presents the system architecture and evidence-gated
self-heal loop. `figures/results.tex` remains a schematic placeholder until
the benchmark runs are complete.

## Project links

- GitHub: <https://github.com/chirpz-ai/pandaprobe-harness>
- Documentation: <https://docs.pandaprobe.com/harness>

## arXiv submission notes

- Primary category: `cs.AI`.
- Candidate cross-lists: `cs.SE`, `cs.MA`, and `cs.LG`.
- Upload `main.tex`, `references.bib`, both TikZ figure sources, and the
  generated `main.bbl`.
- The PandaProbe Harness, PandaProbe Research Lab, Chirpz AI, and author
  attribution are intentional and must remain onymized.
- Inspect the final PDF for table overflow, figure legibility, URL wrapping,
  and accessibility of plots.

## Author TODOs before posting

- Confirm the exact spelling of **Sina Tayebati**, affiliation, contact email,
  ORCID, and any co-authors.
- Fill every result placeholder from audited benchmark artifacts; never infer
  or fabricate measurements.
- Complete the two methodological checkpoints and report null outcomes.
- Replace the schematic results figure with the measured learning curve.
- Verify every reference, package version, model matrix, command, and public
  link at submission time.
- Recheck the integration-verification status recorded in
  `../../benchmarks/IMPLEMENTATION_NOTES.md` after full orchestration runs.
- Rebuild from a clean directory and confirm there are no undefined citations,
  references, missing files, or overfull boxes.

## Availability and reproducibility

PandaProbe Harness is MIT-licensed and maintained by the PandaProbe research
lab. Study commands and outputs are documented in `../../benchmarks/README.md` and
`../../benchmarks/RUNNING.md`; implementation maturity and deviations are
recorded in `../../benchmarks/IMPLEMENTATION_NOTES.md`.
