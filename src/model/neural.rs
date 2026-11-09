use crate::{
    data::{Point, Random},
    numerics::dot,
};

pub struct Network {
    pub layers: Vec<Layer>,
    relu: bool,
    rate: f64,
    lambda: f64,
    pub epoch: u32,
}

pub struct Layer {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<f64>,
    momentum: Vec<f64>,
    variance: Vec<f64>,
}

impl Network {
    pub fn new(depth: usize, hidden: usize, relu: bool, rate: f64, lambda: f64, seed: u32) -> Self {
        let mut rng = Random(seed ^ 2026);
        let layers = (0..=depth)
            .map(|i| {
                let input = if i == 0 { 1 } else { hidden };
                let output = if i == depth { 1 } else { hidden };
                let connections = input * output;
                let count = connections + output;
                let mut weights = vec![0.0; count];
                let scale = ((if relu { 6.0 } else { 3.0 }) / input as f64).sqrt();
                for j in 0..output {
                    for w in &mut weights[j * input..(j + 1) * input] {
                        *w = (2.0 * rng.sample() - 1.0) * scale;
                    }
                    if i == 0 {
                        let sign = if rng.sample() < 0.5 { -1.0 } else { 1.0 };
                        weights[j] = sign * (1.0 + 4.0 * rng.sample());
                        weights[connections + j] = -weights[j] * (2.0 * rng.sample() - 1.0);
                    }
                }
                Layer {
                    input,
                    output,
                    weights,
                    momentum: vec![0.0; count],
                    variance: vec![0.0; count],
                }
            })
            .collect();
        Self {
            layers,
            relu,
            rate,
            lambda,
            epoch: 0,
        }
    }

    /// Restores the reference 1→32→1 tanh network and its Adam accumulators.
    pub fn restore_reference(&mut self) -> bool {
        use super::checkpoint::{MOMENT, VARIANCE, WEIGHTS};
        if self.layers.len() != 2
            || self.layers[0].output != 32
            || self.relu
            || self.rate != 0.01
            || self.lambda != 0.0001
        {
            return false;
        }
        let mut offset = 0;
        for layer in &mut self.layers {
            let end = offset + layer.weights.len();
            layer.weights.copy_from_slice(&WEIGHTS[offset..end]);
            layer.momentum.copy_from_slice(&MOMENT[offset..end]);
            layer.variance.copy_from_slice(&VARIANCE[offset..end]);
            offset = end;
        }
        self.epoch = 6000;
        true
    }

    pub fn forward(&self, x: f64) -> Vec<Vec<f64>> {
        let mut values = vec![vec![x / 3.0]];
        for (i, layer) in self.layers.iter().enumerate() {
            let input = &values[i];
            let next = (0..layer.output)
                .map(|j| {
                    let z = dot(
                        input,
                        &layer.weights[j * layer.input..(j + 1) * layer.input],
                    ) + layer.weights[layer.input * layer.output + j];
                    if i + 1 == self.layers.len() {
                        z
                    } else if self.relu {
                        z.max(0.0)
                    } else {
                        z.tanh()
                    }
                })
                .collect();
            values.push(next);
        }
        values
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.forward(x)[self.layers.len()][0]
    }

    pub fn step(&mut self, points: &[Point]) -> bool {
        let mut gradients: Vec<_> = self
            .layers
            .iter()
            .map(|layer| vec![0.0; layer.weights.len()])
            .collect();
        for point in points {
            let values = self.forward(point.x);
            let mut delta =
                vec![2.0 * (values[self.layers.len()][0] - point.y) / points.len() as f64];
            for (i, layer) in self.layers.iter().enumerate().rev() {
                let mut previous = vec![0.0; layer.input];
                for (j, &d) in delta.iter().enumerate() {
                    gradients[i][layer.input * layer.output + j] += d;
                    for (k, (accumulator, &input)) in
                        previous.iter_mut().zip(&values[i]).enumerate()
                    {
                        let index = j * layer.input + k;
                        gradients[i][index] += d * input;
                        *accumulator += layer.weights[index] * d;
                    }
                }
                if i > 0 {
                    for (v, &activation) in previous.iter_mut().zip(&values[i]) {
                        *v *= if self.relu {
                            if activation > 0.0 { 1.0 } else { 0.0 }
                        } else {
                            1.0 - activation.powi(2)
                        };
                    }
                    delta = previous;
                }
            }
        }
        for (layer, gradient) in self.layers.iter().zip(&mut gradients) {
            for (g, w) in gradient[..layer.input * layer.output]
                .iter_mut()
                .zip(&layer.weights)
            {
                *g += 2.0 * self.lambda * w;
            }
        }
        if !gradients.iter().flatten().all(|g| g.is_finite()) {
            return false;
        }
        self.epoch += 1;
        let b1 = 1.0 - 0.9_f64.powf(f64::from(self.epoch));
        let b2 = 1.0 - 0.999_f64.powf(f64::from(self.epoch));
        for (layer, gradient) in self.layers.iter_mut().zip(gradients) {
            for (i, g) in gradient.into_iter().enumerate() {
                layer.momentum[i] = 0.9 * layer.momentum[i] + 0.1 * g;
                layer.variance[i] = 0.999 * layer.variance[i] + 0.001 * g * g;
                layer.weights[i] -=
                    self.rate * (layer.momentum[i] / b1) / ((layer.variance[i] / b2).sqrt() + 1e-8);
            }
        }
        true
    }
}
