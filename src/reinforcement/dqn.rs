use super::{ACTIONS, GAMMA, Random, STATES, Transition};

/// Параметры и активации онлайн-сети для одной выбранной клетки.
pub struct NetworkView {
    pub activation: [f64; HIDDEN],
    pub input_weights: [f64; HIDDEN],
    pub output_weights: [[f64; HIDDEN]; ACTIONS],
    pub output_bias: [f64; ACTIONS],
    pub replay_size: usize,
    pub updates: usize,
}

const HIDDEN: usize = 24;
const REPLAY: usize = 512;
const BATCH: usize = 16;
const TARGET_INTERVAL: usize = 64;

// При one-hot входе предактивации скрытого слоя равны весам активной клетки.
#[derive(Clone)]
struct Network {
    input: [[f64; HIDDEN]; STATES],
    output: [[f64; HIDDEN]; ACTIONS],
    bias: [f64; ACTIONS],
}

impl Network {
    fn new(rng: &mut Random) -> Self {
        Self {
            input: std::array::from_fn(|_| std::array::from_fn(|_| 0.8 * rng.sample() - 0.4)),
            output: std::array::from_fn(|_| std::array::from_fn(|_| 0.24 * rng.sample() - 0.12)),
            bias: [0.0; ACTIONS],
        }
    }

    fn values(&self, state: usize) -> [f64; ACTIONS] {
        std::array::from_fn(|a| {
            self.bias[a]
                + self.input[state]
                    .iter()
                    .zip(self.output[a])
                    .map(|(&z, w)| z.max(0.0) * w)
                    .sum::<f64>()
        })
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

    pub(super) fn values(&self, state: usize) -> [f64; ACTIONS] {
        self.online.values(state)
    }

    pub(super) fn view(&self, state: usize) -> NetworkView {
        let input_weights = self.online.input[state];
        NetworkView {
            activation: input_weights.map(|z| z.max(0.0)),
            input_weights,
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
            input: [[0.0; HIDDEN]; STATES],
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
            let error =
                (t.reward + GAMMA * future - self.online.values(t.from)[t.action]).clamp(-1.0, 1.0);
            grad.bias[t.action] += error;
            for h in 0..HIDDEN {
                let z = self.online.input[t.from][h];
                if z > 0.0 {
                    grad.output[t.action][h] += error * z;
                    grad.input[t.from][h] += error * self.online.output[t.action][h];
                }
            }
        }
        let rate = 0.08 / BATCH as f64;
        for (weights, deltas) in self
            .online
            .input
            .iter_mut()
            .zip(grad.input)
            .chain(self.online.output.iter_mut().zip(grad.output))
        {
            for (w, delta) in weights.iter_mut().zip(deltas) {
                *w += rate * delta;
            }
        }
        for (w, delta) in self.online.bias.iter_mut().zip(grad.bias) {
            *w += rate * delta;
        }
        self.updates += 1;
        if self.updates.is_multiple_of(TARGET_INTERVAL) {
            self.target = self.online.clone();
        }
    }
}
