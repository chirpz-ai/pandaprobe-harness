# Scoped on-demand rule delivery

## Task-facing contract

Task system context contains only a short, stable statement that optional learned
guidance is available through PandaProbe's four read-only rule tools. It contains
neither rule bodies nor an expanded rule index, and context construction performs
no implicit list, read, search, or retrieval. The task agent decides whether to
list scopes, search live guidance, or read one validated scope.

## Index and regeneration

`harness_guide.md` is the package-owned SKILL-style task guide and index. Its YAML
frontmatter declares the four allowed read-only tools, its stable body explains
the pull workflow, and
its generated References section has one deterministic entry per scope containing
active or provisional rules: `global` first, then all other scopes alphabetically.
Each entry contains the scope key, `rules/<scope>.md`, one bounded safe description,
and active/provisional counts—never rule bodies. Rule
creation, promotion, retirement, and workspace startup regenerate the index and
scope files with atomic replacement. Persisted scope descriptions are kept in a
small metadata file; absent metadata is reconstructed deterministically so 0.8.0
workspaces migrate the legacy generated root and recover without moving or
duplicating their scoped rules.

## Scope metadata flow

Hosts may attach bounded `RuleScopeHint` values to a completed turn. A hint keeps
topic (`key`) separate from applicability (`global`, `topical`, or `task`) and may
carry a short safe description. The hook persists those hints on the resulting
notice. A repair assignment receives every available hint plus one recommended
scope. A precise host topic outranks a generic benchmark label; explicit global
applicability remains available for genuinely cross-domain guidance. `scoped` is
the default for ordinary granular guidance; `global` must be explicit. Scope keys
are an open catalog: a host or managed repair may choose any concise custom name
without a category prefix. PandaProbe normalizes the key only for filename safety,
owns the resulting `rules/<scope>.md` path, and strictly validates task reads.
Neither agent can supply an arbitrary path.

AppWorld derives app hints from application names already visible in initialized
task metadata and API metadata. tau2 supplies its configured domain and bounded
workflow metadata. Terminal-Bench supplies category/task-family metadata when
Harbor exposes it. None of these paths calls a classification model or relies only
on an opaque task id.

## Repair episodes and novelty

Pending notices from the same task session and turn are coalesced only when their
trace/signature evidence overlaps. The resulting repair episode retains all notice
IDs and evidence references, and one resolution atomically acknowledges the whole
group. Timeout, cancellation, or failure acknowledges nothing. One episode may
create at most one candidate.

Before a repair proposal is accepted, the store checks normalized exact text and
live rules in the selected scope using failure signatures, bounded tags, and
deterministic lexical overlap. Covered proposals resolve as duplicate (active) or
already-covered (provisional); unactionable evidence may resolve without a rule.
The package-owned prompt requires searching active and provisional guidance and
treats narrower examples or wording changes as non-novel.

## Compatibility and telemetry

Existing `global`, `scoped`, and custom scope records retain their primary scope.
New optional rule, notice, episode, and scope-metadata fields use forgiving readers
with defaults for older persisted records. Structured journal/benchmark telemetry
records episode IDs, grouped notice IDs, recommended/selected scopes, considered
rules, resolution and suppression details, candidate IDs, repair usage, index
regeneration, and per-scope lifecycle counts without logging unrestricted prompts,
credentials, provider responses, or diagnostic payloads.
