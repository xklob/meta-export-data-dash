#!/bin/sh

set -eu

project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

secure_artifacts() {
  for artifact_root in \
    "$project_root/.private-review" \
    "$project_root/screenshots" \
    "$project_root/test-results" \
    "$project_root/web/test-results" \
    "$project_root/playwright-report"
  do
    if [ -L "$artifact_root" ]; then
      echo "Refusing symlinked private artifact root: $artifact_root" >&2
      exit 2
    fi
    if [ -e "$artifact_root" ] && [ ! -d "$artifact_root" ]; then
      echo "Refusing non-directory private artifact root: $artifact_root" >&2
      exit 2
    fi
    mkdir -p -m 700 "$artifact_root"
    if find "$artifact_root" -xdev -type l -print -quit | grep -q .; then
      echo "Refusing nested symlink below private artifact root: $artifact_root" >&2
      exit 2
    fi
    if find "$artifact_root" -xdev ! -type d ! -type f -print -quit | grep -q .; then
      echo "Refusing special file below private artifact root: $artifact_root" >&2
      exit 2
    fi
    find "$artifact_root" -xdev -type d -exec chmod 700 {} +
    find "$artifact_root" -xdev -type f -exec chmod 600 {} +
  done
}

if [ "$#" -eq 0 ]; then
  echo "Usage: sh scripts/run-private-ui-review.sh COMMAND [ARG ...]" >&2
  exit 2
fi

umask 077
secure_artifacts
trap secure_artifacts EXIT
"$@"
