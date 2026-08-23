# Security policy

Meta Archive Explorer is designed for local processing of private Facebook export data. Security
reports that could expose archive content should never include real messages, names, attachments,
paths, screenshots, tokens, catalogs, manifests, or derived indexes.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting flow:

<https://github.com/xklob/meta-export-data-dash/security/advisories/new>

Do not open a public issue for a suspected vulnerability. A synthetic reproducer is preferred. If
real-data behavior is essential to explain the issue, describe its structure and redact the values.

## Security boundaries

Reports are especially useful when they involve:

- non-loopback network exposure or authentication/CSRF bypasses;
- path traversal, symlink races, unsafe archive extraction, or writes outside the selected root;
- unintended cloud transmission or disclosure through logs, errors, reports, or API responses;
- deletion, quarantine, restore, or migration behavior that violates documented invariants;
- unsafe attachment parsing, sandbox escape, or unbounded resource consumption;
- secret exposure in source, fixtures, build output, or repository history.

The default branch is the supported version. This project does not claim to erase independent
backups, provider-retained data, or forensic remnants outside the configured archive workspace.
