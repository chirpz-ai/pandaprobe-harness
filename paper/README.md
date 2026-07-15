# Paper draft — "Self-Healing Agents: An Evidence-Gated Rule-Learning Harness for Reliable LLM Agents"

Anonymized (double-blind) **AAAI 2027** first draft, using the official AAAI 2027 author-kit style
(`aaai2027.sty` + `aaai2027.bst`).

## Build

The AAAI 2027 style **requires pdfLaTeX** (it has an engine guard and will *not* compile under
XeLaTeX / LuaLaTeX / tectonic). On Overleaf, set the compiler to "pdfLaTeX" (the default). Locally:

```bash
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

This produces `main.pdf`. `bibtex` is used (not `biber`). The bibliography style is set
automatically by `aaai2027.sty` (`aaai2027.bst`, author–year) — do **not** add a
`\bibliographystyle` command. **Undefined-citation warnings from the `placeholder-*` bib keys are
expected** — they mean the entry still needs the author to verify/complete its details, not that a
key is missing. There should be no missing-file or undefined-**reference** (cross-ref) errors, and
(per AAAI) **no overfull boxes** — check the log.

## Layout

```
paper/
├── main.tex                     # the full paper (all sections)
├── references.bib               # every \cite key resolves; all flagged for author verification
├── aaai2027.sty                 # official AAAI 2027 style (do not modify)
├── aaai2027.bst                 # official AAAI 2027 bibliography style (do not modify)
├── figures/
│   ├── overview.tex             # Fig 1 — self-heal loop schematic (self-contained TikZ; \input-ed)
│   └── results.tex              # Fig 2 — learning-dynamics placeholder (self-contained TikZ; \input-ed)
└── README.md                    # this file
```

## AAAI compliance notes (already applied)

- `\documentclass[letterpaper]{article}` + `\usepackage[submission]{aaai2027}`; the `submission`
  option auto-anonymizes the author line ("Anonymous submission") and suppresses the copyright slug.
- Forbidden packages removed (`geometry`, `hyperref`, `siunitx`); fonts come from the style
  (`newtxtext`/`helvet`/`courier`) — no font package is loaded.
- Algorithms use the AAAI-recommended `algorithm` + `algorithmic` packages.
- `\setcounter{secnumdepth}{2}` so sections/subsections are numbered (the paper cross-references
  them via `\S\ref`). AAAI does not support numbering below subsection — do not raise this above 2.
- Both figures are wrapped in `\resizebox` so they cannot overflow the column/gutter (AAAI forbids
  overfull boxes).
- `tikz` is used to draw the two schematic figures (it is *not* on the AAAI disallowed list;
  `pgfplots`, which is disallowed, is not used).

## What is real vs. placeholder

- **Real:** the formalism (§4) and the experimental design (§5) are grounded in the actual
  implementation and study configuration; each non-obvious equation has a `% source:` comment in
  `main.tex`. The statistical estimators (§5) match the study's code exactly.
- **Placeholder (must be filled):** every numeric result cell (the results tables) is a `\TBD`
  marker; both figures are compilable schematics of the *expected shape*, not measured data; and the
  abstract/intro headline effect is a `[TODO]`. No quantitative result is invented.

## Author TODO list

1. **Single-file + metadata for final AAAI submission.** AAAI requires the source as a *single*
   `.tex` file (plus the `.bib`) — no `\input`. For the final archive, inline the two TikZ figures
   from `figures/*.tex` directly into `main.tex` (or pre-render them to PDF and use
   `\includegraphics`), then remove the `\input`s. Also clear the exported PDF's metadata with a
   metadata-cleaning tool so it does not de-anonymize you. (During review/authoring, `\input` on
   Overleaf compiles fine.)
2. **Fill the result tables** once the benchmark runs land: `tab:headline` (main results),
   `tab:telemetry` (telemetry/cost), `tab:ablation` (ablations), and the `#learn`/`#eval` counts in
   `tab:config`. Replace every `\TBD`. Fill the headline-effect placeholders in the abstract and
   intro.
3. **Swap the figures.** Redraw Fig 1 (`figures/overview.tex`) as a polished schematic and replace
   Fig 2 (`figures/results.tex`) with the real learning curve (or the Δpass@1-vs-Δpass^k bars); each
   file has in-source notes on exactly what to draw. Keep figure text ≥ 9 pt and embed all fonts.
4. **Complete the citations.** Every entry in `references.bib` carries a `note = {TODO(author):
   verify ...}`. Verify venue/year/pages, complete the two `@misc` benchmark entries
   (terminal-task, tool-dialogue), and remove the notes. Missing required BibTeX fields are
   unacceptable to `aaai2027.bst`.
5. **Finalize naming.** The system is referred to generically as *the Harness* and the evaluation
   platform as *a session-level agent-evaluation service*; the benchmark suite has no brand. If a
   name is desired for camera-ready (post-review), introduce it in the intro and use it consistently.
6. **Run the two checkpoints and report them** (§5.6): metric–failure calibration (precision/recall/
   F1) and the rule-promotion gate (promoted-rule counts). Add their outcomes to §6.
7. **Final anonymization sweep** before every submission build. Grep `paper/` (case-insensitive)
   for the project's identifying strings and confirm it returns nothing. Fill `<identifiers>` with
   your product name, benchmark-suite name, company, author name(s), any project domains, and any
   env-var prefixes (kept out of this README so the sweep does not match itself):
   ```bash
   grep -riE '<identifiers>' paper/   # e.g. product|company|suite|author|domains|ENV_PREFIX
   ```
