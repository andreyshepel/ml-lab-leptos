mod checkpoint;
mod neural;
mod tree;
pub use neural::Network;
pub use tree::Tree;

use crate::{
    data::{Point, phase},
    numerics::{backward, cholesky, dot, forward, least_squares},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Polynomial,
    Knn,
    Tree,
    Forest,
    Rbf,
    Neural,
    Gp,
}

impl Method {
    pub const ALL: [Self; 7] = [
        Self::Polynomial,
        Self::Knn,
        Self::Tree,
        Self::Forest,
        Self::Rbf,
        Self::Neural,
        Self::Gp,
    ];
    pub fn title(self) -> &'static str {
        match self {
            Self::Polynomial => "Полиномиальная регрессия",
            Self::Knn => "k ближайших соседей",
            Self::Tree => "Дерево решений",
            Self::Forest => "Случайный лес",
            Self::Rbf => "RBF-регрессия",
            Self::Neural => "Многослойная нейросеть",
            Self::Gp => "Гауссовский процесс",
        }
    }
    pub fn tab(self) -> &'static str {
        match self {
            Self::Polynomial => "Регрессия",
            Self::Knn => "kNN",
            Self::Tree => "Дерево",
            Self::Forest => "Лес",
            Self::Rbf => "RBF",
            Self::Neural => "Нейросеть",
            Self::Gp => "Гауссовский процесс",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Parameters {
    Polynomial {
        degree: usize,
        lambda: f64,
    },
    Knn {
        k: usize,
    },
    Tree {
        depth: usize,
        leaf: usize,
    },
    Forest {
        trees: usize,
        depth: usize,
        leaf: usize,
    },
    Rbf {
        centers: usize,
        width: f64,
        lambda: f64,
    },
    Neural {
        depth: usize,
        hidden: usize,
        relu: bool,
        rate: f64,
        lambda: f64,
    },
    Gp {
        length: f64,
        noise: f64,
    },
}

impl Parameters {
    pub fn preset(method: Method, index: usize, count: usize, periodic: bool) -> Self {
        let i = index.min(2);
        match method {
            Method::Polynomial => Self::Polynomial {
                degree: [1, if periodic { 5 } else { 11 }, 23][i],
                lambda: 0.0,
            },
            Method::Knn => Self::Knn { k: [18, 2, 1][i] },
            Method::Tree => Self::Tree {
                depth: [2, 5, 16][i],
                leaf: [3, 1, 1][i],
            },
            Method::Forest => Self::Forest {
                trees: [30, 30, 64][i],
                depth: [2, 8, 16][i],
                leaf: [3, 1, 1][i],
            },
            Method::Rbf => Self::Rbf {
                centers: [4, 24, count.clamp(2, 48)][i],
                width: [1.2, 0.5, 0.06][i],
                lambda: [0.1, 0.001, 1e-8][i],
            },
            Method::Neural => Self::Neural {
                depth: [1, 2, 3][i],
                hidden: [3, 32, 32][i],
                relu: false,
                rate: 0.01,
                lambda: [0.01, 0.0001, 0.0][i],
            },
            Method::Gp => Self::Gp {
                length: [2.0, 0.6, 0.06][i],
                noise: [0.3, 0.07, 0.001][i],
            },
        }
    }

    pub fn initial(method: Method) -> Self {
        match method {
            Method::Polynomial => Self::Polynomial {
                degree: 4,
                lambda: 0.0,
            },
            Method::Neural => Self::Neural {
                depth: 1,
                hidden: 32,
                relu: false,
                rate: 0.01,
                lambda: 0.0001,
            },
            _ => Self::preset(method, 1, 24, false),
        }
    }
}

pub struct Fit {
    pub model: Model,
    pub periodic: bool,
}

pub enum Model {
    Polynomial {
        coefficients: Vec<f64>,
        center: f64,
        scale: f64,
    },
    Knn {
        points: Vec<Point>,
        k: usize,
    },
    Tree(Tree),
    Forest(Vec<Tree>),
    Rbf {
        centers: Vec<f64>,
        coefficients: Vec<f64>,
        width: f64,
    },
    Neural(Network),
    Gp {
        centers: Vec<f64>,
        coefficients: Vec<f64>,
        length: f64,
        lower: Vec<Vec<f64>>,
    },
}

fn features(z: f64, count: usize) -> Vec<f64> {
    let mut result = vec![1.0; count];
    if count > 1 {
        result[1] = z;
    }
    for i in 2..count {
        result[i] = 2.0 * z * result[i - 1] - result[i - 2];
    }
    result
}

fn kernel(x: f64, center: f64, width: f64) -> f64 {
    (-0.5 * ((x - center) / width).powi(2)).exp()
}

fn ridge(mut rows: Vec<Vec<f64>>, points: &[Point], lambda: f64, skip: usize) -> Option<Vec<f64>> {
    let mut targets: Vec<_> = points.iter().map(|p| p.y).collect();
    let columns = rows[0].len();
    if lambda > 0.0 {
        for j in skip..columns {
            let mut row = vec![0.0; columns];
            row[j] = lambda.sqrt();
            rows.push(row);
            targets.push(0.0);
        }
    }
    least_squares(rows, targets)
}

impl Fit {
    pub fn train(
        points: &[Point],
        parameters: Parameters,
        periodic: bool,
        seed: u32,
    ) -> Result<Self, &'static str> {
        if points.len() < 2 || !points.iter().all(|p| p.x.is_finite() && p.y.is_finite()) {
            return Err("Для обучения нужны хотя бы две конечные точки.");
        }
        let lo = points.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let hi = points.iter().map(|p| p.x).fold(f64::NEG_INFINITY, f64::max);
        let model = match parameters {
            Parameters::Polynomial { degree, lambda } => {
                let count = degree.min(points.len() - 1).min(23) + 1;
                let center = (lo + hi) / 2.0;
                let scale = if hi > lo { (hi - lo) / 2.0 } else { 1.0 };
                let rows = points
                    .iter()
                    .map(|p| features((p.x - center) / scale, count))
                    .collect();
                let coefficients = ridge(rows, points, lambda, 1).ok_or(
                    "Полином не удалось оценить. Уменьшите степень или включите регуляризацию.",
                )?;
                Model::Polynomial {
                    coefficients,
                    center,
                    scale,
                }
            }
            Parameters::Knn { k } => {
                let mut sorted = points.to_vec();
                sorted.sort_by(|a, b| a.x.total_cmp(&b.x));
                Model::Knn {
                    points: sorted,
                    k: k.clamp(1, points.len()),
                }
            }
            Parameters::Tree { depth, leaf } => Model::Tree(Tree::fit(
                points.to_vec(),
                depth.clamp(1, 16),
                leaf.clamp(1, 8),
            )),
            Parameters::Forest { trees, depth, leaf } => Model::Forest(tree::forest(
                points,
                trees.clamp(1, 64),
                depth.clamp(1, 16),
                leaf.clamp(1, 8),
                seed,
            )),
            Parameters::Rbf {
                centers,
                width,
                lambda,
            } => {
                let centers: Vec<_> = crate::data::grid(lo, hi, centers.clamp(2, 48)).collect();
                let rows = points
                    .iter()
                    .map(|p| centers.iter().map(|&c| kernel(p.x, c, width)).collect())
                    .collect();
                let coefficients = ridge(rows, points, lambda.max(1e-8), 0)
                    .ok_or("Не удалось оценить коэффициенты RBF.")?;
                Model::Rbf {
                    centers,
                    coefficients,
                    width,
                }
            }
            Parameters::Neural {
                depth,
                hidden,
                relu,
                rate,
                lambda,
            } => Model::Neural(Network::new(
                depth.clamp(1, 4),
                hidden.clamp(2, 64),
                relu,
                rate,
                lambda,
                seed,
            )),
            Parameters::Gp { length, noise } => {
                let centers: Vec<_> = points.iter().map(|p| p.x).collect();
                let matrix = centers
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        centers
                            .iter()
                            .enumerate()
                            .map(|(j, &y)| {
                                kernel(x, y, length)
                                    + if i == j { noise.powi(2) + 1e-10 } else { 0.0 }
                            })
                            .collect()
                    })
                    .collect();
                let lower = cholesky(matrix)
                    .ok_or("Не удалось разложить матрицу ядра. Увеличьте предполагаемый шум.")?;
                let mut coefficients: Vec<_> = points.iter().map(|p| p.y).collect();
                forward(&lower, &mut coefficients);
                backward(&lower, &mut coefficients);
                Model::Gp {
                    centers,
                    coefficients,
                    length,
                    lower,
                }
            }
        };
        Ok(Self { model, periodic })
    }

    pub fn input(&self, x: f64) -> f64 {
        if self.periodic { phase(x) } else { x }
    }
    pub fn predict(&self, x: f64) -> f64 {
        self.model.predict(self.input(x))
    }

    pub fn uncertainty(&self, x: f64) -> Option<f64> {
        let Model::Gp {
            centers,
            length,
            lower,
            ..
        } = &self.model
        else {
            return None;
        };
        let mut k: Vec<_> = centers
            .iter()
            .map(|&c| kernel(self.input(x), c, *length))
            .collect();
        forward(lower, &mut k);
        Some((1.0 - dot(&k, &k)).max(0.0).sqrt())
    }
}

impl Model {
    pub fn predict(&self, x: f64) -> f64 {
        match self {
            Self::Knn { points, k } => nearest(points, *k, x).map(|p| p.y).sum::<f64>() / *k as f64,
            Self::Tree(tree) => tree.predict(x),
            Self::Forest(trees) => {
                trees.iter().map(|tree| tree.predict(x)).sum::<f64>() / trees.len() as f64
            }
            Self::Neural(net) => net.predict(x),
            _ => self.contributions(x).into_iter().sum(),
        }
    }

    pub fn contributions(&self, x: f64) -> Vec<f64> {
        match self {
            Self::Polynomial {
                coefficients,
                center,
                scale,
            } => features((x - center) / scale, coefficients.len())
                .into_iter()
                .zip(coefficients)
                .map(|(z, w)| z * w)
                .collect(),
            Self::Rbf {
                centers,
                coefficients,
                width,
            }
            | Self::Gp {
                centers,
                coefficients,
                length: width,
                ..
            } => centers
                .iter()
                .zip(coefficients)
                .map(|(&c, w)| w * kernel(x, c, *width))
                .collect(),
            Self::Forest(trees) => trees.iter().map(|t| t.predict(x)).collect(),
            Self::Neural(net) => {
                let layer = &net.layers[net.layers.len() - 1];
                net.forward(x)[net.layers.len() - 1]
                    .iter()
                    .zip(&layer.weights[..layer.input])
                    .map(|(v, w)| v * w)
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}

pub fn nearest(points: &[Point], k: usize, x: f64) -> impl Iterator<Item = &Point> {
    let mut right = points.partition_point(|p| p.x < x);
    let mut left = right;
    (0..k.min(points.len())).map(move |_| {
        if left > 0 && (right == points.len() || x - points[left - 1].x <= points[right].x - x) {
            left -= 1;
            &points[left]
        } else {
            let p = &points[right];
            right += 1;
            p
        }
    })
}
