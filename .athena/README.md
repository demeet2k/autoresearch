# Athena federation contract

This directory makes `demeet2k/autoresearch` a typed participant in the Athena Git Brain.
It does not copy the Athena corpus or grant this repository global authority.

- Resource: `athena.repo.autoresearch@contract-proposal-0.1.0`
- Role: `experiment`
- Authority domain: `experimental-witness`
- Base content witness: `f032120010c570e56451023a8b28a26f401850d8`
- Control-plane schema commit: `3d33fbcd6248fc2dc2991fbbab5e93a7eb184246`
- State: `WITNESS_ONLY_UNTIL_PROMOTED`

`repo.json` declares the local surface. `exports.jsonl` exposes bounded
identities. `imports.lock.json` pins the control-plane schema. `edges.jsonl`
contains the forward declaration and its explicit return edge. `status.json`
preserves blockers instead of promoting them away.
