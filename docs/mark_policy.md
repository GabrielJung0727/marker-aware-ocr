# Mark Policy

## Scope
Resolve final selection when multiple marker symbols exist in one option block.

## Priority Rule (default)
`X > O > V > CHECK > STAR`

## Resolution Procedure
1. Collect all marker labels inside the option block.
2. Normalize marker labels (`marker_x` -> `X`, etc.).
3. Apply priority rule.
4. If no priority match exists, use the last marker as fallback.

## Audit Fields
- `mark`: final resolved mark
- `markers`: normalized raw marker labels
- `reason`: resolution reason (`priority_match:*`, `fallback_last_marker`, `no_marker`)
