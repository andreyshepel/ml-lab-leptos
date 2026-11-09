use std::collections::BTreeMap;
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

pub(crate) struct Random(pub u32);

impl Random {
    pub fn sample(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = (self.0 ^ (self.0 >> 15)).wrapping_mul(self.0 | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }

    pub fn index(&mut self, length: usize) -> usize {
        (self.sample() * length as f64) as usize
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct DataConfig {
    pub count: usize,
    pub noise: f64,
    pub seed: u32,
    pub outliers: bool,
    pub outlier_count: usize,
    pub strength: f64,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            count: 24,
            noise: 0.5,
            seed: 72,
            outliers: true,
            outlier_count: 3,
            strength: 3.0,
        }
    }
}

pub struct Dataset {
    pub train: Vec<Point>,
    pub test: Vec<Point>,
}

pub fn target(x: f64) -> f64 {
    (PI * x).sin()
}

pub fn grid(start: f64, end: f64, count: usize) -> impl Iterator<Item = f64> {
    (0..count).map(move |i| start + (end - start) * i as f64 / (count - 1).max(1) as f64)
}

impl Dataset {
    pub fn generate(config: DataConfig) -> Self {
        let mut train_rng = Random(config.seed);
        let mut test_rng = Random(config.seed ^ 0x9e3779b9);
        let mut train: Vec<_> = grid(-3.0, 3.0, config.count.clamp(8, 200))
            .map(|x| Point {
                x,
                y: target(x) + config.noise * (2.0 * train_rng.sample() - 1.0),
            })
            .collect();
        let test = (0..200)
            .map(|_| {
                let x = 6.0 * test_rng.sample() - 3.0;
                Point {
                    x,
                    y: target(x) + config.noise * (2.0 * test_rng.sample() - 1.0),
                }
            })
            .collect();
        if config.outliers {
            let mut rng = Random(config.seed ^ 0x47a2b91);
            let mut order: Vec<_> = (0..train.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.index(i + 1));
            }
            for i in order
                .into_iter()
                .take(config.outlier_count.min(train.len() / 3).min(6))
            {
                let sign = if rng.sample() < 0.5 { -1.0 } else { 1.0 };
                train[i].y += sign * config.strength * (0.8 + 0.4 * rng.sample());
            }
        }
        Self { train, test }
    }
}

pub fn phase(x: f64) -> f64 {
    (x + 1.0).rem_euclid(2.0) - 1.0
}

pub fn fold(points: &[Point]) -> Vec<Point> {
    const SCALE: f64 = 1e10;
    let mut groups = BTreeMap::<i64, (f64, usize)>::new();
    for &Point { x, y } in points {
        let key = (phase(x) * SCALE).round() as i64;
        let key = if key == SCALE as i64 { -key } else { key };
        let group = groups.entry(key).or_default();
        group.0 += y;
        group.1 += 1;
    }
    groups
        .into_iter()
        .map(|(key, (sum, n))| Point {
            x: key as f64 / SCALE,
            y: sum / n as f64,
        })
        .collect()
}

#[derive(Clone, Copy)]
pub struct Metrics {
    pub rmse: f64,
    pub r_squared: Option<f64>,
}

pub fn metrics(points: &[Point], predict: impl Fn(f64) -> f64) -> Metrics {
    let count = points.len() as f64;
    let mean = points.iter().map(|p| p.y).sum::<f64>() / count;
    let error = points
        .iter()
        .map(|p| (p.y - predict(p.x)).powi(2))
        .sum::<f64>();
    let variance = points.iter().map(|p| (p.y - mean).powi(2)).sum::<f64>();
    Metrics {
        rmse: (error / count).sqrt(),
        r_squared: (variance > 0.0).then(|| 1.0 - error / variance),
    }
}
