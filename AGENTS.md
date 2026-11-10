# Context for experiments

This directory contains executable machine learning experiments for preparing and presenting a tech talk. These interactive demos live inside the GameOfSpace Obsidian vault. The root `AGENTS.md` describes the broader career context; when working on these labs, carry out the user's specific technical request. Company interviews and career-gap analysis apply when requested, rather than to every code change.

Write documentation and UI explanations in Russian, except for this English agent instruction file. The demos should explain data, training, prediction, model assumptions, and limitations. Visual clarity must preserve mathematical correctness.

## Experiment map

### sine-regression — the original minimal example

Polynomial regression on `y = sin(πx) + noise`: underfitting, overfitting, regularization, and training versus test errors.

- Plain JavaScript without external dependencies; open `index.html` in a browser, or use the standalone `demo.html`.
- `model.js` — data generation, polynomial model, and metrics; also used by the neighboring JavaScript lab.
- `app.js` — state management and SVG plots; `index.html` — markup and styling.
- `test.cjs` — numerical tests and event-handler checks using a simulated DOM.

### sine-ml-lab — the expanded JavaScript lab

Seven regression methods on shared data: polynomial regression, kNN, decision tree, random forest, RBF, neural network, and Gaussian process. Includes diagrams of actual prediction calculations, periodic input transformations, outliers, and manual observation exclusion.

- `index.html`, `style.css`, `app.js` — UI.
- `models.js` — additional models; the polynomial model, shared data, and metrics come from `../sine-regression/model.js`.
- `periodic.js` — phase transformation and grouping of identical phases.
- `structure.js` — model diagrams based on current parameters.
- `pretrained.js` — saved periodic-network weights and Adam state.
- `train-periodic.cjs` — reproduces the checkpoint; overwrites `pretrained.js` and `periodic-results.json`.
- `build.cjs` — embeds styles and scripts into the standalone `demo.html`.
- `test.cjs` — checks models, neural-network gradients, event handlers, and the standalone build.

### ml-lab-leptos — the Rust and WebAssembly lab

A Rust/Leptos port of the regression lab, with additional clustering and reinforcement-learning tabs. This is a separate application: it does not call the neighboring JavaScript implementations at runtime.

- `src/data.rs` — data, phase, and metrics.
- `src/model.rs`, `src/model/` — models, parameters, and neural-network checkpoint.
- `src/numerics.rs` — numerical methods; `src/lab.rs` — regression experiment state.
- `src/clustering.rs` — k-means, single-link, and DBSCAN.
- `src/reinforcement.rs` — Q-learning, SARSA, and DQN on a shared grid.
- `src/reinforcement/cartpole.rs` and `src/reinforcement/cartpole/` — continuing CartPole dynamics and online DQN training.
- `src/ui/` — Leptos components and SVG; `style.css` — styling.
- `tests/model_contract.rs` — integration tests; source files also contain unit tests.
- `Cargo.toml`, `Cargo.lock`, `Trunk.toml` — dependencies and build configuration. Requires Rust 1.89 or newer, the `wasm32-unknown-unknown` target, and Trunk; see the local `README.md`.

The JavaScript and Rust versions share their origin and educational purpose, but their capabilities differ. Do not treat them as interchangeable or propagate every change to all directories unnecessarily.

## Making changes

1. Read the relevant experiment's `README.md` and code first. Check current defaults in code: saved-network parameters can differ from the initial screen settings.
2. Choose the directory based on the request. If the user does not name a version, use the conversation context; clarify material ambiguity before changing an implementation.
3. Edit source files. In `sine-ml-lab`, regenerate `demo.html` with `build.cjs` rather than editing it separately. `sine-regression` has no dedicated build script: when changing its demo, verify that the standalone file matches the source page.
4. Changes to shared `sine-regression/model.js` affect both JavaScript labs: check both and rebuild `sine-ml-lab/demo.html`.
5. Do not retrain the checkpoint for ordinary UI changes. When changing data, architecture, optimizer, or checkpoint format, check loading conditions and result reproducibility.
6. Update the relevant `README.md` when behavior changes. Extend this map when adding an experiment.
7. Preserve standalone JavaScript demos and browser execution of Rust models. Add an external server, CDN, or framework only when the task justifies it.

## Rust coding rules

Adapted from [minimaxir's Rust code-quality guidelines](https://gist.github.com/minimaxir/068ef4137a1b6c1dcefa785349c91728). Apply these to Rust changes. Project-specific architecture and build instructions take precedence: retain Leptos, local CSS, offline operation, and Trunk. Do not introduce Pico CSS, another UI stack, or server/Python tooling solely to follow the source's preferences. This section does not require a retroactive repository-wide refactor.

- Favor maintainability, reuse, efficient algorithms, borrowing, and minimal allocations. Measure optimizations; consider SIMD/parallelism only where compatible with browser execution.
- Follow idiomatic naming, four-space indentation, and rustfmt's 100-column default. Explain unfamiliar Rust details without redundant comments or emoji.
- Document public APIs, inputs, outputs, errors, and complex examples.
- Model distinctions with types/newtypes and absence with `Option`. Return `Result`; propagate contextual errors. Avoid production/library `unwrap`; reserve descriptive `expect` for invariants. Prefer `thiserror`/`anyhow` where needed.
- Keep functions/types focused; use configuration structs beyond five parameters. Prefer private fields, composition, appropriate derives, early returns, and clear iterators.
- Avoid wildcard imports except preludes/tests. Constrain dependencies. Consider lightweight crates before duplicating functionality.
- Avoid `unsafe`; document necessary safety invariants. Make cloning explicit, matches exhaustive, ownership deliberate, and capacities preallocated when known.
- Test meaningful behavior and edge cases with `#[test]`/`#[cfg(test)]`; isolate external dependencies. Use fmt, Clippy, and warning-free builds.
- Benchmark sequentially, comparably, without manipulated cases or custom `RUSTFLAGS`.
- Use appropriate synchronization, `Send`/`Sync`, and channels; evaluate Tokio/Rayon against target constraints.
- Keep credentials outside code/logs; ignore local secret files. Remove debug output, dead code, and disabled tests. Use descriptive commits.
- Keep computation in Rust; rebuild affected WASM.

### Module boundaries and file organization

- Organize modules around cohesive responsibilities. Do not accumulate unrelated functionality in `main.rs`, `lib.rs`, `mod.rs`, or a generic utility file.
- Keep crate entry points focused on initialization, composition, and wiring.
- Before adding functionality, decide whether it belongs to an existing module or introduces a distinct responsibility that needs a new module.
- Separate domain logic and numerical algorithms from UI rendering, event handling, persistence, and platform integration. Core algorithms must be testable without the UI.
- If a file exceeds approximately 400 lines, or a change introduces a second independent responsibility, review its boundaries. Split it when responsibilities can be named clearly; line count alone does not require splitting.
- Extract cohesive types and their operations together. Avoid creating many tiny files or moving arbitrary code into `utils.rs` merely to shorten a file.
- Keep implementation details private. Use `pub(super)` or `pub(crate)` when sufficient; expose only the API required by callers.
- Prefer explicit dependencies between modules. Resolve circular dependencies by revisiting ownership and shared abstractions.
- When extending an oversized file, extract the relevant responsibility as part of the change when practical. Keep refactoring limited to the affected area.

In this lab, keep model calculations in `src/model/`, regression experiment state in `src/lab.rs`, and UI components and event handlers in `src/ui/`. If `src/reinforcement.rs` grows to contain independent responsibilities, consider a module directory separating the environment, algorithms, and training orchestration.

### Input validation, numerical failures, and browser state

- Validate parameters at system boundaries before computation: layer dimensions, ranges, learning rates, and sample sizes. Reject invalid values with actionable errors rather than silently changing their meaning.
- Handle numerical failures explicitly. Non-finite values (`NaN` or infinity), singular systems, and training divergence must produce a diagnosable state instead of silently propagating through metrics and plots.
- Keep the browser responsive. Run long training operations in bounded batches that yield to the browser and allow cancellation; avoid a single uninterrupted loop that blocks interaction.
- Define which parameter changes invalidate weights, metrics, and cached intermediate results. Keep rendering separate from training: a redraw alone must not restart or advance training. Prevent cancelled or outdated runs from overwriting current results.

## Educational constraints

- Do not use test data or the known target function to train weights. If settings are selected using test metrics, explain that this dataset effectively becomes validation data.
- Preserve seed reproducibility and an independent test-data stream. Switching models or display settings must not silently change the data being compared.
- The period `T = 2` is supplied by a person, not discovered by the algorithm. Periodic extrapolation demonstrates the effect of this assumption.
- R² is not accuracy and can be negative. Near-zero training error does not establish good generalization.
- Artificial outliers are known to the generator. Manual marking and exclusion are not automatic error detection; in periodic mode, exclusion happens before phase grouping.
- Diagrams must reflect real parameters, activations, and prediction paths. Explicitly label hidden portions of large models while retaining their contribution to calculations.
- Preserve numerically stable calculations: the Chebyshev basis and QR/Cholesky methods rather than explicit matrix inversion. Rounded formulas shown in the UI must not drive plot calculations.
- Do not replace algorithm output with a predetermined illustration for visual effect. Limitations and unsuccessful results also have educational value.

## Running and checking

For JavaScript, run from the vault root:

```sh
node --test experiments/sine-regression/test.cjs
node experiments/sine-ml-lab/build.cjs
node --test experiments/sine-ml-lab/test.cjs
```

Open `sine-regression/index.html` or the relevant `demo.html` in a browser. If a tool cannot access `file://`, use a local HTTP server.

For Rust, run from `experiments/ml-lab-leptos`:

```sh
trunk serve --locked
cargo fmt --all -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
cargo clippy --locked --target wasm32-unknown-unknown -- -D warnings
trunk build --release --locked
```

Run checks relevant to the changed behavior. For Rust/WASM changes, also verify the browser build: native Rust tests do not exercise the Leptos UI. The built site is in `dist/` and requires all generated JS and WASM files. Use Trunk for this project's WASM rebuilds.

A simulated DOM does not replace visual inspection. After UI changes, check affected browser scenarios: method switching, parameters, plots and formulas, outlier marking and exclusion, neural-network training/pause/reset; for clustering and RL, stepping and reset. Distinguish completed checks from anything that could not be verified.

## Related vault materials

When working on the presentation scenario as well as code, consult:

- `40 Sources/Техтолк - Машинное обучение/Сценарий лекции.md`.
- `40 Sources/Техтолк - Машинное обучение/Практикум и подготовка.md`.
- `20 Analysis/2026-11-05 - Техтолк о машинном обучении.md`.

These materials explain the audience and presentation context. Do not automatically edit them for a local technical change unless it affects the scenario they describe.
