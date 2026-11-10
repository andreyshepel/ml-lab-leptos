use crate::data::{Point, Random};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Start,
    Assigned,
    Recentered,
    Converged,
}

pub struct KMeans {
    pub points: Vec<Point>,
    pub centers: Vec<Point>,
    pub labels: Vec<usize>,
    pub step: Step,
    pub iterations: usize,
}

fn distance(a: Point, b: Point) -> f64 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2)
}

pub fn observations(seed: u32) -> Vec<Point> {
    let mut rng = Random(seed);
    let mut points = Vec::with_capacity(105);
    for (x, y) in [(-2.5, -1.3), (0.1, 1.5), (2.5, -0.9)] {
        for _ in 0..35 {
            let r = 0.95 * rng.sample().sqrt();
            let angle = std::f64::consts::TAU * rng.sample();
            points.push(Point {
                x: x + r * angle.cos(),
                y: y + r * angle.sin(),
            });
        }
    }
    points
}

impl KMeans {
    pub fn example(k: usize, seed: u32) -> Self {
        let points = observations(seed);
        let centers = [
            (-3.1, 1.7),
            (2.8, 1.6),
            (0.0, -2.2),
            (-3.1, -2.2),
            (3.2, -2.3),
        ]
        .into_iter()
        .take(k.clamp(2, 5))
        .map(|(x, y)| Point { x, y })
        .collect();
        Self {
            points,
            centers,
            labels: Vec::new(),
            step: Step::Start,
            iterations: 0,
        }
    }

    pub fn advance(&mut self) {
        match self.step {
            Step::Start | Step::Recentered => {
                let labels = self
                    .points
                    .iter()
                    .map(|&point| {
                        self.centers
                            .iter()
                            .enumerate()
                            .min_by(|(_, a), (_, b)| {
                                distance(point, **a).total_cmp(&distance(point, **b))
                            })
                            .map_or(0, |(index, _)| index)
                    })
                    .collect::<Vec<_>>();
                if self.labels == labels {
                    self.step = Step::Converged;
                } else {
                    self.labels = labels;
                    self.step = Step::Assigned;
                }
            }
            Step::Assigned => {
                let mut sums = vec![(0.0, 0.0, 0); self.centers.len()];
                for (&point, &label) in self.points.iter().zip(&self.labels) {
                    let (x, y, n) = &mut sums[label];
                    *x += point.x;
                    *y += point.y;
                    *n += 1;
                }
                for (center, (x, y, n)) in self.centers.iter_mut().zip(sums) {
                    // An empty cluster keeps its previous center.
                    if n > 0 {
                        *center = Point {
                            x: x / f64::from(n),
                            y: y / f64::from(n),
                        };
                    }
                }
                self.iterations += 1;
                self.step = Step::Recentered;
            }
            Step::Converged => {}
        }
    }

    pub fn squared_error(&self) -> Option<f64> {
        (!self.labels.is_empty()).then(|| {
            self.points
                .iter()
                .zip(&self.labels)
                .map(|(&p, &label)| distance(p, self.centers[label]))
                .sum()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demonstration_alternates_and_converges_without_increasing_error() {
        for k in 2..=5 {
            let mut model = KMeans::example(k, 72);
            let mut previous = f64::INFINITY;
            for _ in 0..100 {
                model.advance();
                let error = model.squared_error().unwrap();
                assert!(error <= previous + 1e-9);
                previous = error;
                if model.step == Step::Converged {
                    break;
                }
            }
            assert_eq!(model.step, Step::Converged);
        }
    }

    #[test]
    fn changing_k_keeps_the_same_unlabeled_points() {
        let a = KMeans::example(2, 72);
        let b = KMeans::example(3, 72);
        assert_eq!(a.points, b.points);
        assert!(a.labels.is_empty() && b.labels.is_empty());
        assert_eq!(a.centers.len(), 2);
        assert_eq!(b.centers.len(), 3);
    }
}

/// Single-link agglomeration. Edges sorted by length define the merge order.
pub struct SingleLink {
    labels: Vec<usize>,
    edges: Vec<(f64, usize, usize)>,
    next_edge: usize,
    target: usize,
    pub groups: usize,
    pub last_merge: Option<(usize, usize)>,
}

impl SingleLink {
    pub fn new(points: &[Point], target: usize) -> Self {
        let mut edges = Vec::new();
        for (i, &a) in points.iter().enumerate() {
            for (j, &b) in points.iter().enumerate().skip(i + 1) {
                edges.push((distance(a, b), i, j));
            }
        }
        edges.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self {
            labels: (0..points.len()).collect(),
            edges,
            next_edge: 0,
            target: target.clamp(1, points.len().max(1)),
            groups: points.len(),
            last_merge: None,
        }
    }

    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    pub fn done(&self) -> bool {
        self.groups <= self.target
    }

    pub fn advance(&mut self) {
        while !self.done() {
            let Some(&(_, a, b)) = self.edges.get(self.next_edge) else {
                return;
            };
            self.next_edge += 1;
            let (left, right) = (self.labels[a], self.labels[b]);
            if left != right {
                for label in &mut self.labels {
                    if *label == right {
                        *label = left;
                    }
                }
                self.groups -= 1;
                self.last_merge = Some((a, b));
                return;
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DensityLabel {
    Pending,
    Noise,
    Cluster(usize),
}

/// Each step examines a new seed point and grows its entire density-connected group.
pub struct Dbscan {
    pub labels: Vec<DensityLabel>,
    neighborhoods: Vec<Vec<usize>>,
    visited: Vec<bool>,
    min_samples: usize,
    next_seed: usize,
    pub groups: usize,
    pub last_seed: Option<usize>,
}

impl Dbscan {
    pub fn new(points: &[Point], epsilon: f64, min_samples: usize) -> Self {
        let neighborhoods = points
            .iter()
            .map(|&p| {
                points
                    .iter()
                    .enumerate()
                    .filter(|(_, q)| distance(p, **q) <= epsilon.powi(2))
                    .map(|(i, _)| i)
                    .collect()
            })
            .collect();
        Self {
            labels: vec![DensityLabel::Pending; points.len()],
            visited: vec![false; points.len()],
            neighborhoods,
            min_samples: min_samples.max(1),
            next_seed: 0,
            groups: 0,
            last_seed: None,
        }
    }

    pub fn done(&self) -> bool {
        self.next_seed == self.labels.len()
    }

    pub fn advance(&mut self) {
        if self.done() {
            return;
        }
        let seed = self.next_seed;
        self.last_seed = Some(seed);
        self.visited[seed] = true;
        if self.neighborhoods[seed].len() < self.min_samples {
            self.labels[seed] = DensityLabel::Noise;
        } else {
            let cluster = DensityLabel::Cluster(self.groups);
            self.groups += 1;
            self.labels[seed] = cluster;
            let mut queue = std::collections::VecDeque::new();
            let mut queued = vec![false; self.labels.len()];
            let enqueue =
                |i: usize, queue: &mut std::collections::VecDeque<usize>, queued: &mut [bool]| {
                    for &neighbor in &self.neighborhoods[i] {
                        if !std::mem::replace(&mut queued[neighbor], true) {
                            queue.push_back(neighbor);
                        }
                    }
                };
            enqueue(seed, &mut queue, &mut queued);
            while let Some(i) = queue.pop_front() {
                if !std::mem::replace(&mut self.visited[i], true)
                    && self.neighborhoods[i].len() >= self.min_samples
                {
                    enqueue(i, &mut queue, &mut queued);
                }
                if matches!(self.labels[i], DensityLabel::Pending | DensityLabel::Noise) {
                    self.labels[i] = cluster;
                }
            }
        }
        while self.next_seed < self.visited.len() && self.visited[self.next_seed] {
            self.next_seed += 1;
        }
    }
}

#[cfg(test)]
mod comparison_tests {
    use super::*;

    #[test]
    fn single_link_merges_the_closest_groups_first() {
        let points = [0.0, 0.1, 5.0, 5.2].map(|x| Point { x, y: 0.0 });
        let mut model = SingleLink::new(&points, 2);
        model.advance();
        assert_eq!(model.last_merge, Some((0, 1)));
        model.advance();
        assert!(model.done());
        assert_eq!(model.labels(), &[0, 0, 2, 2]);
    }

    #[test]
    fn dbscan_relabels_border_points_and_keeps_isolated_noise() {
        let points = [0.0, 0.5, 1.0, 5.0].map(|x| Point { x, y: 0.0 });
        let mut model = Dbscan::new(&points, 0.6, 3);
        model.advance();
        assert!(model.labels[0] == DensityLabel::Noise);
        while !model.done() {
            model.advance();
        }
        assert!(
            model.labels[..3]
                .iter()
                .all(|&l| l == DensityLabel::Cluster(0))
        );
        assert!(model.labels[3] == DensityLabel::Noise);
    }
}
