# Node 26 CI Runtime Design

## Scope

Run CI, smoke, and release workflows on an exact Node 26 version while retaining the package's existing Node `>=22.19.0` compatibility contract and Node 22 type baseline.

## Changes

- Pin every `actions/setup-node` workflow to Node `26.5.0`.
- Update the README's CI-runtime note without changing `package.json` or lockfile engine requirements.
- Accept both npm 11's array and npm 12's single-object `npm pack --json` output in the supply-chain guard.

## Verification

- Cover both package-output shapes in the supply-chain guard tests.
- Run `npm run check`, clean build, supply-chain guard, and package dry-run under Node 26.
- Confirm the package engine remains `>=22.19.0`.

## Non-goals

- Raising the minimum supported Node version.
- Updating `@types/node` from the Node 22 baseline.
- Adding a multi-version CI matrix.
