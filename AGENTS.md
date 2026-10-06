# Ablate-to-Validate: agent instructions

Mirrored file: AGENTS.md and CLAUDE.md in this directory are identical copies (Codex reads AGENTS.md, Claude reads CLAUDE.md). Edit AGENTS.md, then copy it over CLAUDE.md in the same commit.
Machine-local pointers: if an untracked AGENTS.local.md (or its identical copy CLAUDE.local.md) exists next to this file, read it at session start; it points to the owner's project state on this machine.

This is the PUBLIC code release and project page of the Ablate-to-Validate paper. It is derived from the authors' private research repository and from the paper: code arrives as ported changes, and every number comes from the paper. Start with `README.md` and `repo_docs/structure.md`.

Live state is kept in the owner's OS project CONTEXT (see AGENTS.local.md); don't start new handoff files.

## Rules

- Never commit personal data, machine-specific paths or cluster names, private repository names, venue submission or review ids, or tokens. Check every added line for them before a commit.
- Data and model weights that are not already public stay out; the owner decides each release.
- Every number in public text (README, `docs/index.html`, the teaser, the guides) must match the paper.
- The local branch `wip/june-2026` holds private paths and is never pushed. Sync rounds with the paper go through a branch and a pull request (as in PR #1); merging a pull request needs the owner's go.
- GitHub Pages publishes `docs/` from `main`; files outside `docs/`, including this one, are not on the site.
- Pushes follow the owner's PERSONAL.md (see AGENTS.local.md): allowed, aggregated, each announced with its purpose. Fetch before the first edit and again before a commit or push. Never rebase or force-push. No AI attribution.
