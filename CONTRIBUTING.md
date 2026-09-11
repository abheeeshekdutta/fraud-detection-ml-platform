# Contributing

Use Python 3.11 and Node.js 22 or newer. Install the locked environments with
`uv sync --locked --extra dev` and `npm ci --prefix dashboard`.

Run `make check` before opening a pull request. Changes to event delivery should also pass
`make integration` with Kafka running. CI runs backend lint and tests, dashboard tests and builds,
a real Kafka contract test, and the full Docker demo. Add regression coverage for failure behavior and observable
contracts rather than duplicating implementation details.

Keep feature generation shared between training and scoring. Preserve event IDs on retries and
include model, feature-schema, and policy versions in every decision. Document any changes to
artifact compatibility, event fields, delivery semantics, or deployment commands.

Use synthetic data in tests. Never commit downloaded IEEE-CIS files, credentials, fitted model
pickles, or generated operational records. Documentation should describe implemented behavior and
label historical measurements and future extensions explicitly.
