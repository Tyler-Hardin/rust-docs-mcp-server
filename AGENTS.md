# AGENTS.md

This is a **personal fork** of
[Govcraft/rust-docs-mcp-server](https://github.com/Govcraft/rust-docs-mcp-server).
It is not the upstream project. Do not open upstream PRs from this branch, and do not
expect upstream to match it.

| Remote     | URL                                            | Role                          |
| ---------- | ---------------------------------------------- | ----------------------------- |
| `origin`   | `git@github.com:Tyler-Hardin/rust-docs-mcp-server.git` | This fork — force-push freely |
| `upstream` | `https://github.com/Govcraft/rust-docs-mcp-server.git` | Read-only source of truth |

`main` is a **linear stack**: one or more fork-local commits rebased directly on top of
`upstream/main`. Nothing else lives on `main`.

## The one rule

> **Every commit must look as if it had been written today, against the newly pulled
> `upstream/main`.**

A commit in this fork is not a historical record of what happened. It is a patch that
*should apply cleanly and make sense to a reader who has never seen the old base*.
Concretely, when rebasing:

- **Amend, don't append.** A fixup belongs *inside* the commit it corrects, not in a new
  "fix rebase" commit. History should read like a clean feature branch, not a log of
  maintenance work.
- **Prefer the modern API.** If upstream renamed something, bumped a dependency, or
  changed a signature, rewrite the fork commit to use it directly. No compatibility
  shims, no stale call shapes.
- **No functionality may be dropped.** If a conflict is hard, re-implement the intent
  against the new code — do not delete the feature, the flag, or the test. If something
  genuinely cannot be preserved, stop and ask.
- **Keep upstream's commits untouched.** Only fork-local commits are rewritten.
- **Legitimate upstream bugs may be fixed in a fork-local commit.** This project is old
  and lightly maintained; a broken dev shell or build is fair game to fix, in its own
  commit.
- **Merge conflicts are a signal, not an obstacle.** An auto-merged file can still be
  semantically wrong. A conflict-free rebase is not a correct rebase.

## Maintenance process

```sh
git switch main
git branch backup/pre-rebase-$(date +%Y%m%d-%H%M%S)   # always take a backup
git fetch upstream --prune
git rebase upstream/main
```

1. **Compile, then test.** A clean `git rebase --continue` proves nothing:

   ```sh
   nix develop --command bash -c 'cargo check --locked && cargo test --locked'
   ```

   This crate currently has **no unit tests**, so `cargo check` is the primary signal
   that a rebase did not break something. Exercise the changed paths manually when the
   change is behavioural.

2. **Build through Nix**, since that is how it is deployed:

   ```sh
   nix build .# --no-link --print-out-paths
   nix flake check --no-build
   ```

3. **Fold fixes into the commit they belong to**, then re-test the final tree:

   ```sh
   git add <files> && git commit -m 'fixup! feat: add a multi-crate mode'
   GIT_SEQUENCE_EDITOR=true git rebase --autosquash upstream/main
   ```

4. **Push.** Rewriting history is expected:

   ```sh
   git fetch origin --prune
   git push --force-with-lease origin main
   ```

   Prefer `--force-with-lease` over `--force`; it refuses if anyone else pushed.

## Notes for agents

- Confirm the working tree is clean, and record a `backup/` branch, before any rewrite.
- The dev shell must stay **hermetic**: it should not depend on the caller's
  `~/.rustup`, `~/.cargo/bin`, or on network access at shell entry. Toolchains and cargo
  subcommands come from the flake's pinned nixpkgs.
- Do not commit build artifacts (`target/`, `result` symlinks).
- Documentation-only or process changes (like this file) get their **own commit**.
- Report what you verified and what you did not — an untested `nix build` or a skipped
  test run should be stated explicitly, not implied to pass.
