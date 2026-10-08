# Configurable scoring/game-piece implementation plan
Date: 2026-10-07
Status: Implemented on feature/custom-game-pieces-scoring-targets; validation in progress

1. Audit existing React/RK4 pipeline, optimizer client and score classifier. Preserve default hub regressions.
2. Research shooting games in FIRST archives (2006, 2012, 2013, 2016, 2017, 2020, 2022, 2024, 2026) and record representative geometry and approximation limits in the design spec.
3. Define strict schema, input validation, safe storage recovery, and CRUD helpers. No database or external service needed for browser-local saving.
4. Implement 3-D aperture classifiers and target-rendering geometry. Keep historical hub behavior exactly as default.
5. Wire selection into main simulation, results, 2-D & 3-D renderers, optimizer and robust analysis. Add a new UI editing panel.
6. Add Node regression tests for storage reload, custom edits, invalid input, each scoring geometry, and simulation integration.
7. Run CI physics regressions, frontend lint, build and Python tests. Address failures before merge; retain PR for review.
