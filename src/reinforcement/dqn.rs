use super::{ACTIONS, GAMMA, GOAL_COUNT, Random, STATES, State, Transition};

/// Параметры и активации онлайн-сети для одной выбранной клетки.
pub struct NetworkView {
    pub first: [f64; HIDDEN],
    pub second: [f64; HIDDEN],
    pub output_weights: [[f64; HIDDEN]; ACTIONS],
    pub output_bias: [f64; ACTIONS],
    pub replay_size: usize,
    pub updates: usize,
}

const HIDDEN: usize = 64;
const REPLAY: usize = 512;
const BATCH: usize = 16;
const TARGET_INTERVAL: usize = 64;

// Скрытый слой суммирует веса активной клетки и флагов собранных наград.
#[derive(Clone)]
struct Network {
    input: Vec<[f64; HIDDEN]>,
    hidden: Vec<[f64; HIDDEN]>,
    hidden_bias: [f64; HIDDEN],
    output: [[f64; HIDDEN]; ACTIONS],
    bias: [f64; ACTIONS],
}

impl Network {
    fn new(rng: &mut Random) -> Self {
        Self {
            input: (0..STATES + GOAL_COUNT)
                .map(|_| std::array::from_fn(|_| 0.8 * rng.sample() - 0.4))
                .collect(),
            hidden: (0..HIDDEN)
                .map(|_| {
                    std::array::from_fn(|_| {
                        (2.0 * rng.sample() - 1.0) * (6.0 / HIDDEN as f64).sqrt()
                    })
                })
                .collect(),
            hidden_bias: [0.0; HIDDEN],
            output: std::array::from_fn(|_| std::array::from_fn(|_| 0.24 * rng.sample() - 0.12)),
            bias: [0.0; ACTIONS],
        }
    }

    fn inputs(&self, state: State) -> [f64; HIDDEN] {
        let mut input = self.input[state.cell];
        for goal in 0..GOAL_COUNT {
            if state.collected.0 & (1 << goal) != 0 {
                for (sum, weight) in input.iter_mut().zip(self.input[STATES + goal]) {
                    *sum += weight;
                }
            }
        }
        input
    }

    fn activations(&self, state: State) -> ([f64; HIDDEN], [f64; HIDDEN]) {
        let first = self.inputs(state).map(|value| value.max(0.0));
        let second = std::array::from_fn(|i| {
            (self.hidden_bias[i] + crate::numerics::dot(&first, &self.hidden[i])).max(0.0)
        });
        (first, second)
    }

    fn values(&self, state: State) -> [f64; ACTIONS] {
        let (_, second) = self.activations(state);
        std::array::from_fn(|a| self.value(&second, a))
    }

    fn value(&self, second: &[f64; HIDDEN], action: usize) -> f64 {
        self.bias[action] + crate::numerics::dot(second, &self.output[action])
    }
}

pub(super) struct DeepQ {
    online: Network,
    target: Network,
    replay: Vec<Transition>,
    cursor: usize,
    updates: usize,
}

impl DeepQ {
    pub(super) fn new(seed: u32) -> Self {
        let online = Network::new(&mut Random(seed ^ 0x73a1));
        Self {
            target: online.clone(),
            online,
            replay: Vec::with_capacity(REPLAY),
            cursor: 0,
            updates: 0,
        }
    }

    pub(super) fn values(&self, state: State) -> [f64; ACTIONS] {
        self.online.values(state)
    }

    pub(super) fn view(&self, state: State) -> NetworkView {
        let (first, second) = self.online.activations(state);
        NetworkView {
            first,
            second,
            output_weights: self.online.output,
            output_bias: self.online.bias,
            replay_size: self.replay.len(),
            updates: self.updates,
        }
    }

    pub(super) fn learn(&mut self, transition: Transition, rng: &mut Random) {
        if self.replay.len() < REPLAY {
            self.replay.push(transition);
        } else {
            self.replay[self.cursor] = transition;
        }
        self.cursor = (self.cursor + 1) % REPLAY;
        if self.replay.len() < BATCH {
            return;
        }
        let mut grad = Network {
            input: vec![[0.0; HIDDEN]; STATES + GOAL_COUNT],
            hidden: vec![[0.0; HIDDEN]; HIDDEN],
            hidden_bias: [0.0; HIDDEN],
            output: [[0.0; HIDDEN]; ACTIONS],
            bias: [0.0; ACTIONS],
        };
        for _ in 0..BATCH {
            let t = self.replay[rng.index(self.replay.len())];
            let future = if t.terminal {
                0.0
            } else {
                self.target
                    .values(t.to)
                    .into_iter()
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let (first, second) = self.online.activations(t.from);
            let error =
                (t.reward + GAMMA * future - self.online.value(&second, t.action)).clamp(-1.0, 1.0);
            grad.bias[t.action] += error;
            let mut first_delta = [0.0; HIDDEN];
            for (h, activation) in second.into_iter().enumerate() {
                if activation > 0.0 {
                    grad.output[t.action][h] += error * activation;
                    let delta = error * self.online.output[t.action][h];
                    grad.hidden_bias[h] += delta;
                    for (i, &input) in first.iter().enumerate() {
                        grad.hidden[h][i] += delta * input;
                        first_delta[i] += delta * self.online.hidden[h][i];
                    }
                }
            }
            for (i, delta) in first_delta.into_iter().enumerate() {
                if first[i] > 0.0 {
                    grad.input[t.from.cell][i] += delta;
                    for goal in 0..GOAL_COUNT {
                        if t.from.collected.0 & (1 << goal) != 0 {
                            grad.input[STATES + goal][i] += delta;
                        }
                    }
                }
            }
        }
        let rate = 0.08 / BATCH as f64;
        for (weights, deltas) in self
            .online
            .input
            .iter_mut()
            .zip(grad.input)
            .chain(self.online.hidden.iter_mut().zip(grad.hidden))
            .chain(self.online.output.iter_mut().zip(grad.output))
        {
            for (w, delta) in weights.iter_mut().zip(deltas) {
                *w += rate * delta;
            }
        }
        for (w, delta) in self
            .online
            .bias
            .iter_mut()
            .zip(grad.bias)
            .chain(self.online.hidden_bias.iter_mut().zip(grad.hidden_bias))
        {
            *w += rate * delta;
        }
        self.updates += 1;
        if self.updates.is_multiple_of(TARGET_INTERVAL) {
            self.target.clone_from(&self.online);
        }
    }
}
