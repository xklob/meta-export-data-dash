# Contributing

Meta Archive Explorer handles unusually sensitive source material. Changes should preserve the
project's local-first boundary, source provenance, and fail-closed behavior before optimizing for
convenience.

## Development environment

The supported toolchain is pinned in `.python-version`, `.node-version`, and the package-manager
metadata:

```bash
uv sync --all-groups
pnpm install --frozen-lockfile
pnpm --dir web exec playwright install chromium
```

Run the backend and frontend in separate terminals:

```bash
uv run meta-archive serve --archive-root /path/to/an/ignored/export
pnpm dev
```

Use only synthetic fixtures in tests, screenshots, bug reports, and pull requests. Never attach a
real export, catalog, trace, screenshot, baseline manifest, generated library, or model index.

## Validation

Run the fast static and unit gates before opening a pull request:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src scripts
uv run pytest -q
pnpm --dir web run lint
pnpm --dir web run test
pnpm --dir web run test:real-ui-unit
pnpm --dir web run build
```

Changes to routing, archive workflows, destructive operations, accessibility behavior, or visual
output should also run the relevant Playwright suites documented in [README.md](README.md). The
destructive suite creates its own disposable synthetic archive and must never receive a real
archive path.

## Change expectations

- Preserve immutable source coordinates and content hashes through import and migration changes.
- Represent absent or partial export categories as unavailable or partial, never as a zero count.
- Keep the HTTP service bound to loopback and preserve authentication plus CSRF checks.
- Treat export text and extracted attachment content as untrusted data.
- Keep cloud processing disabled by default and make every external-data boundary explicit.
- Add regression coverage for parser variants, path handling, migrations, and deletion behavior.
- Update the README or implementation plan when a public interface or invariant changes.

Security-sensitive findings should follow [SECURITY.md](SECURITY.md), not a public issue.
