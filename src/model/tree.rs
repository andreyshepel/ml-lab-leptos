use crate::data::{Point, Random};

pub struct Tree {
    pub count: usize,
    pub value: f64,
    pub branch: Option<Branch>,
}

pub struct Branch {
    pub threshold: f64,
    pub left: Box<Tree>,
    pub right: Box<Tree>,
}

impl Tree {
    pub fn fit(mut points: Vec<Point>, depth: usize, min_leaf: usize) -> Self {
        points.sort_by(|a, b| a.x.total_cmp(&b.x));
        Self::grow(&points, depth, min_leaf)
    }

    fn grow(points: &[Point], depth: usize, min_leaf: usize) -> Self {
        let count = points.len();
        let sum = points.iter().map(|p| p.y).sum::<f64>();
        let value = sum / count as f64;
        let mut node = Self {
            count,
            value,
            branch: None,
        };
        if depth == 0 || count < 2 * min_leaf {
            return node;
        }
        let squares = points.iter().map(|p| p.y * p.y).sum::<f64>();
        let mut best = squares - sum * value;
        let mut split = None;
        let (mut left, mut left_sq) = (0.0, 0.0);
        for i in 1..count {
            left += points[i - 1].y;
            left_sq += points[i - 1].y.powi(2);
            if i < min_leaf || count - i < min_leaf || points[i - 1].x == points[i].x {
                continue;
            }
            let loss = left_sq - left.powi(2) / i as f64 + (squares - left_sq)
                - (sum - left).powi(2) / (count - i) as f64;
            if loss < best - 1e-12 {
                best = loss;
                split = Some(i);
            }
        }
        if let Some(i) = split {
            node.branch = Some(Branch {
                threshold: (points[i - 1].x + points[i].x) / 2.0,
                left: Box::new(Self::grow(&points[..i], depth - 1, min_leaf)),
                right: Box::new(Self::grow(&points[i..], depth - 1, min_leaf)),
            });
        }
        node
    }

    pub fn predict(&self, x: f64) -> f64 {
        match &self.branch {
            Some(branch) if x <= branch.threshold => branch.left.predict(x),
            Some(branch) => branch.right.predict(x),
            None => self.value,
        }
    }
}

pub fn forest(points: &[Point], count: usize, depth: usize, leaf: usize, seed: u32) -> Vec<Tree> {
    let mut rng = Random(seed ^ 7919);
    (0..count)
        .map(|_| {
            Tree::fit(
                (0..points.len())
                    .map(|_| points[rng.index(points.len())])
                    .collect(),
                depth,
                leaf,
            )
        })
        .collect()
}
