use std::collections::VecDeque;

use super::{Action, NetworkConfig, Random};

const INPUTS: usize = 5;
const ACTIONS: usize = 2;
const MAX_WIDTH: usize = NetworkConfig::MAX_WIDTH;
const MAX_LEVELS: usize = NetworkConfig::MAX_DEPTH + 2;
const CAPACITY: usize = 16_384;
const BATCH: usize = 32;
const GAMMA: f64 = 0.97;
const HORIZON: usize = 5;
const TARGET_INTERVAL: usize = 200;

#[derive(Clone, Copy)]
struct Transition {
    from: [f64; INPUTS],
    action: usize,
    to: [f64; INPUTS],
    reward: f64,
}

struct Layer {
    start: usize,
    inputs: usize,
    outputs: usize,
}

impl Layer {
    fn rows<'a>(&self, values: &'a [f64]) -> std::slice::ChunksExact<'a, f64> {
        let stride = self.inputs + 1;
        values[self.start..self.start + stride * self.outputs].chunks_exact(stride)
    }

    fn rows_mut<'a>(&self, values: &'a mut [f64]) -> std::slice::ChunksExactMut<'a, f64> {
        let stride = self.inputs + 1;
        values[self.start..self.start + stride * self.outputs].chunks_exact_mut(stride)
    }
}

struct Network {
    layers: Vec<Layer>,
    weights: Vec<f64>,
    relu: bool,
}

impl Network {
    fn new(seed: u32, config: NetworkConfig) -> Self {
        let mut rng = Random(seed);
        let mut weights = Vec::new();
        let layers = (0..=config.depth)
            .map(|level| {
                let inputs = if level == 0 { INPUTS } else { config.width };
                let is_output = level == config.depth;
                let outputs = if is_output { ACTIONS } else { config.width };
                let layer = Layer {
                    start: weights.len(),
                    inputs,
                    outputs,
                };
                let scale = if is_output {
                    0.4
                } else if level == 0 {
                    1.0
                } else {
                    2.0 / (inputs as f64).sqrt()
                };
                for _ in 0..outputs {
                    weights.extend((0..inputs).map(|_| (rng.sample() - 0.5) * scale));
                    weights.push(0.0);
                }
                layer
            })
            .collect();
        Self {
            layers,
            weights,
            relu: config.relu,
        }
    }

    fn forward(&self, input: [f64; INPUTS], weights: &[f64]) -> [[f64; MAX_WIDTH]; MAX_LEVELS] {
        let mut activations = [[0.0; MAX_WIDTH]; MAX_LEVELS];
        activations[0][..INPUTS].copy_from_slice(&input);
        for (level, layer) in self.layers.iter().enumerate() {
            let previous = activations[level];
            for (j, row) in layer.rows(weights).enumerate() {
                let z = row[layer.inputs]
                    + row[..layer.inputs]
                        .iter()
                        .zip(previous)
                        .map(|(weight, x)| weight * x)
                        .sum::<f64>();
                activations[level + 1][j] = if level + 1 == self.layers.len() || !z.is_finite() {
                    z
                } else if self.relu {
                    z.max(0.0)
                } else {
                    z.tanh()
                };
            }
        }
        activations
    }

    fn values(&self, input: [f64; INPUTS], weights: &[f64]) -> [f64; ACTIONS] {
        let output = self.forward(input, weights)[self.layers.len()];
        [output[0], output[1]]
    }

    fn gradient(&self, t: Transition, target: f64, gradient: &mut [f64]) {
        let activations = self.forward(t.from, &self.weights);
        let mut delta = [0.0; MAX_WIDTH];
        delta[t.action] = (activations[self.layers.len()][t.action] - target).clamp(-1.0, 1.0);
        for (level, layer) in self.layers.iter().enumerate().rev() {
            let mut previous_delta = [0.0; MAX_WIDTH];
            let inputs = &activations[level][..layer.inputs];
            for ((row, gradient), d) in layer
                .rows(&self.weights)
                .zip(layer.rows_mut(gradient))
                .zip(delta)
            {
                if d == 0.0 {
                    continue;
                }
                gradient[layer.inputs] += d;
                for (i, &input) in inputs.iter().enumerate() {
                    gradient[i] += d * input;
                    if level > 0 {
                        previous_delta[i] += d * row[i];
                    }
                }
            }
            if level > 0 {
                for (d, &activation) in previous_delta.iter_mut().zip(inputs) {
                    if self.relu {
                        if activation <= 0.0 {
                            *d = 0.0;
                        }
                    } else {
                        *d *= 1.0 - activation * activation;
                    }
                }
                delta = previous_delta;
            }
        }
    }
}

pub(super) struct Agent {
    online: Network,
    target: Vec<f64>,
    candidate: Vec<f64>,
    gradient: Vec<f64>,
    config: NetworkConfig,
    replay: Vec<Transition>,
    pending: VecDeque<Transition>,
    cursor: usize,
    updates: usize,
    first: Vec<f64>,
    second: Vec<f64>,
}

impl Agent {
    pub(super) fn new(seed: u32, config: NetworkConfig) -> Self {
        let online = Network::new(seed, config);
        let parameters = online.weights.len();
        Self {
            target: online.weights.clone(),
            online,
            config,
            candidate: vec![0.0; parameters],
            gradient: vec![0.0; parameters],
            replay: Vec::with_capacity(CAPACITY),
            pending: VecDeque::with_capacity(HORIZON),
            cursor: 0,
            updates: 0,
            first: vec![0.0; parameters],
            second: vec![0.0; parameters],
        }
    }

    pub(super) fn values(&self, input: [f64; INPUTS]) -> [f64; 2] {
        self.online.values(input, &self.online.weights)
    }

    pub(super) fn interrupt(&mut self) {
        self.pending.clear();
    }

    pub(super) fn learn(
        &mut self,
        from: [f64; INPUTS],
        action: Action,
        to: [f64; INPUTS],
        reward: f64,
        rng: &mut Random,
    ) -> bool {
        self.pending.push_back(Transition {
            from,
            action: usize::from(action == Action::Right),
            to,
            reward,
        });
        if self.pending.len() < HORIZON {
            return true;
        }
        let mut discount = 1.0;
        let mut total_reward = 0.0;
        for step in &self.pending {
            total_reward += discount * step.reward;
            discount *= GAMMA;
        }
        let Some(first) = self.pending.pop_front() else {
            return true;
        };
        let transition = Transition {
            to,
            reward: total_reward,
            ..first
        };
        if self.replay.len() < CAPACITY {
            self.replay.push(transition);
        } else {
            self.replay[self.cursor] = transition;
        }
        self.cursor = (self.cursor + 1) % CAPACITY;
        if self.replay.len() < 256 {
            return true;
        }
        self.gradient.fill(0.0);
        for _ in 0..BATCH {
            let t = self.replay[rng.index(self.replay.len())];
            let [left, right] = self.online.values(t.to, &self.target);
            if !left.is_finite() || !right.is_finite() {
                return false;
            }
            let target = t.reward + discount * left.max(right);
            self.online.gradient(t, target, &mut self.gradient);
        }
        self.updates += 1;
        let bias_first = 1.0 - 0.9_f64.powf(self.updates as f64);
        let bias_second = 1.0 - 0.999_f64.powf(self.updates as f64);
        for gradient in &mut self.gradient {
            *gradient /= BATCH as f64;
        }
        if self.config.l2 != 0.0 {
            for layer in &self.online.layers {
                for (row, gradient) in layer
                    .rows(&self.online.weights)
                    .zip(layer.rows_mut(&mut self.gradient))
                {
                    for (g, w) in gradient[..layer.inputs].iter_mut().zip(row) {
                        *g += self.config.l2 * w;
                    }
                }
            }
        }
        for (i, &g) in self.gradient.iter().enumerate() {
            self.first[i] = 0.9 * self.first[i] + 0.1 * g;
            self.second[i] = 0.999 * self.second[i] + 0.001 * g * g;
            self.candidate[i] = self.online.weights[i]
                - self.config.rate * (self.first[i] / bias_first)
                    / ((self.second[i] / bias_second).sqrt() + 1e-8);
        }
        if self.candidate.iter().any(|value| !value.is_finite()) {
            return false;
        }
        std::mem::swap(&mut self.online.weights, &mut self.candidate);
        if self.updates.is_multiple_of(TARGET_INTERVAL) {
            self.target.copy_from_slice(&self.online.weights);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multilayer_gradient_matches_finite_differences_for_both_activations() {
        for relu in [true, false] {
            let config = NetworkConfig {
                depth: 3,
                width: 4,
                relu,
                ..NetworkConfig::default()
            };
            let mut network = Network::new(72, config);
            let input = [0.17, -0.24, 0.46, -0.83, 0.31];
            let transition = Transition {
                from: input,
                action: 1,
                to: input,
                reward: 0.0,
            };
            let target = 0.25;
            let mut gradient = vec![0.0; network.weights.len()];
            network.gradient(transition, target, &mut gradient);
            let loss = |net: &Network| 0.5 * (net.values(input, &net.weights)[1] - target).powi(2);
            let step = 1e-6;
            for (i, analytic) in gradient.into_iter().enumerate() {
                let weight = network.weights[i];
                network.weights[i] = weight + step;
                let plus = loss(&network);
                network.weights[i] = weight - step;
                let minus = loss(&network);
                network.weights[i] = weight;
                assert!((analytic - (plus - minus) / (2.0 * step)).abs() < 1e-8);
            }
        }
    }
}
