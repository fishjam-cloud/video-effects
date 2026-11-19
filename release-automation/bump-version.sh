#!/bin/bash
set -euo pipefail

# Called by the release automation during a release.
# Creates a release branch and bumps the package version without committing;
# the caller commits, pushes and opens the PR. Must print BRANCH_NAME:<branch>.
#
# Usage: ./bump-version.sh <version>

VERSION="${1:-}"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

BRANCH_NAME="release-$VERSION"
git checkout -b "$BRANCH_NAME"

npm version "$VERSION" --no-git-tag-version
echo "Updated package.json to $VERSION"

# Turn the "Unreleased" section into this version's entry, or add an empty entry
# under the title when there is none. perl -pi works the same with GNU and BSD tools.
export VERSION
if grep -q '^## Unreleased$' CHANGELOG.md; then
    perl -pi -e 's/^## Unreleased$/## $ENV{VERSION}/ if !$done++ && /^## Unreleased$/' CHANGELOG.md
    echo "Renamed the Unreleased CHANGELOG.md section to $VERSION"
elif ! grep -q "^## $VERSION\$" CHANGELOG.md; then
    perl -0pi -e 's/\A(# Changelog\n\n)/$1## $ENV{VERSION}\n\n- TODO: describe the changes in this release.\n\n/' CHANGELOG.md
    echo "Added a CHANGELOG.md entry for $VERSION"
fi

echo "BRANCH_NAME:$BRANCH_NAME"
