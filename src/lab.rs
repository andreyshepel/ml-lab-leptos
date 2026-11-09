use std::collections::BTreeSet;

use crate::{
    data::{DataConfig, Dataset, Metrics, Point, fold, grid, metrics, target},
    model::{Fit, Method, Model, Parameters},
};

pub struct Lab {
    pub data_config: DataConfig,
    pub data: Dataset,
    pub periodic: bool,
    pub exclude: bool,
    pub marked: BTreeSet<usize>,
    pub parameters: [Parameters; 7],
    pub models: [Result<Fit, &'static str>; 7],
    pub scores: [Option<[Metrics; 3]>; 7],
    pub training: Vec<Point>,
    pub history: Vec<(f64, f64, f64)>,
    pub epochs: u32,
    pub warning: Option<&'static str>,
    used: Vec<Point>,
    clean: Vec<Point>,
}

pub fn index(method: Method) -> usize {
    method as usize
}

impl Default for Lab {
    fn default() -> Self {
        Self::new()
    }
}

impl Lab {
    pub fn new() -> Self {
        let data_config = DataConfig::default();
        let mut lab = Self {
            data_config,
            data: Dataset::generate(data_config),
            periodic: false,
            exclude: false,
            marked: BTreeSet::new(),
            parameters: Method::ALL.map(Parameters::initial),
            models: std::array::from_fn(|_| Err("Модель не рассчитана.")),
            scores: [None; 7],
            training: vec![],
            history: vec![],
            epochs: 6000,
            warning: None,
            used: vec![],
            clean: grid(-3.0, 3.0, 1001)
                .map(|x| Point { x, y: target(x) })
                .collect(),
        };
        lab.refit_all();
        lab
    }

    pub fn regenerate(&mut self) {
        self.data = Dataset::generate(self.data_config);
        self.marked.clear();
        self.refit_all();
    }

    fn selected_points(&self) -> Vec<Point> {
        self.data
            .train
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| !self.marked.contains(i))
            .map(|(_, p)| p)
            .collect()
    }

    pub fn mark(&mut self, i: usize) {
        if i >= self.data.train.len() {
            return;
        }
        self.warning = None;
        if !self.marked.remove(&i) {
            self.marked.insert(i);
            if fold(&self.selected_points()).len() < 2 {
                self.marked.remove(&i);
                self.warning = Some("Оставьте хотя бы две точки с разными фазами.");
                return;
            }
        }
        if self.exclude {
            self.refit_all();
        }
    }

    pub fn refit_all(&mut self) {
        self.warning = None;
        self.used = if self.exclude {
            self.selected_points()
        } else {
            self.data.train.clone()
        };
        self.training = if self.periodic {
            fold(&self.used)
        } else {
            self.used.clone()
        };
        for method in Method::ALL {
            self.refit(method);
        }
        let config = self.data_config;
        if self.periodic
            && !config.outliers
            && self.used.len() == 24
            && config.count == 24
            && config.seed == 42
            && config.noise == 0.12
            && self.epochs == 6000
            && let Ok(Fit {
                model: Model::Neural(net),
                ..
            }) = &mut self.models[index(Method::Neural)]
            && net.restore_reference()
        {
            self.score(Method::Neural);
            self.history.clear();
            self.record();
        }
    }

    pub fn refit(&mut self, method: Method) {
        let i = index(method);
        match &mut self.parameters[i] {
            Parameters::Polynomial { degree, .. } => {
                *degree = (*degree).min(self.training.len().saturating_sub(1))
            }
            Parameters::Knn { k } => *k = (*k).min(self.training.len()),
            _ => {}
        }
        self.models[i] = Fit::train(
            &self.training,
            self.parameters[i],
            self.periodic,
            self.data_config.seed,
        );
        self.score(method);
        if method == Method::Neural {
            self.history.clear();
            self.record();
        }
    }

    pub fn used_count(&self) -> usize {
        self.used.len()
    }

    fn score(&mut self, method: Method) {
        let i = index(method);
        self.scores[i] = self.models[i].as_ref().ok().map(|fit| {
            [&self.used, &self.data.test, &self.clean].map(|data| metrics(data, |x| fit.predict(x)))
        });
    }

    fn record(&mut self) {
        if let Some(scores) = self.scores[index(Method::Neural)] {
            self.history
                .push((f64::from(self.epoch()), scores[0].rmse, scores[1].rmse));
        }
    }

    pub fn epoch(&self) -> u32 {
        match &self.models[index(Method::Neural)] {
            Ok(Fit {
                model: Model::Neural(net),
                ..
            }) => net.epoch,
            _ => 0,
        }
    }

    pub fn train_batch(&mut self, max_epochs: u32, limit: u32) -> bool {
        let Ok(Fit {
            model: Model::Neural(net),
            ..
        }) = &mut self.models[index(Method::Neural)]
        else {
            return false;
        };
        for _ in 0..max_epochs.min(limit.saturating_sub(net.epoch)) {
            if !net.step(&self.training) {
                self.warning =
                    Some("Численная ошибка обучения. Уменьшите скорость и начните заново.");
                return false;
            }
        }
        let more = net.epoch < limit;
        self.score(Method::Neural);
        self.record();
        more
    }

    pub fn batch_size(&self) -> u32 {
        let Ok(Fit {
            model: Model::Neural(net),
            ..
        }) = &self.models[index(Method::Neural)]
        else {
            return 1;
        };
        let weights: usize = net.layers.iter().map(|l| l.weights.len()).sum();
        (90_000 / (self.training.len() * weights).max(1)).clamp(1, 40) as u32
    }

    pub fn diagnostic(&self, method: Method) -> (&'static str, Vec<(f64, f64, f64)>) {
        if method == Method::Neural {
            return ("Эпохи", self.history.clone());
        }
        let params = self.parameters[index(method)];
        let n = self.training.len();
        let (label, values): (_, Vec<_>) = match params {
            Parameters::Polynomial { lambda, .. } => (
                "Степень полинома",
                (0..n.min(24))
                    .map(|degree| (degree as f64, Parameters::Polynomial { degree, lambda }))
                    .collect(),
            ),
            Parameters::Knn { .. } => {
                let mut ks = vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, n];
                ks.retain(|k| *k <= n);
                ks.sort_unstable();
                ks.dedup();
                (
                    "Число соседей",
                    ks.into_iter()
                        .map(|k| (k as f64, Parameters::Knn { k }))
                        .collect(),
                )
            }
            Parameters::Tree { leaf, .. } => (
                "Глубина дерева",
                [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
                    .into_iter()
                    .map(|depth| (depth as f64, Parameters::Tree { depth, leaf }))
                    .collect(),
            ),
            Parameters::Forest { depth, leaf, .. } => (
                "Число деревьев",
                [1, 2, 4, 8, 16, 24, 32, 48, 64]
                    .into_iter()
                    .map(|trees| (trees as f64, Parameters::Forest { trees, depth, leaf }))
                    .collect(),
            ),
            Parameters::Rbf {
                centers, lambda, ..
            } => (
                "Ширина колокола",
                [0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0]
                    .into_iter()
                    .map(|width| {
                        (
                            width,
                            Parameters::Rbf {
                                centers,
                                width,
                                lambda,
                            },
                        )
                    })
                    .collect(),
            ),
            Parameters::Gp { noise, .. } => (
                "Масштаб связи",
                [0.06, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.2, 1.6, 2.0]
                    .into_iter()
                    .map(|length| (length, Parameters::Gp { length, noise }))
                    .collect(),
            ),
            Parameters::Neural { .. } => return ("Эпохи", vec![]),
        };
        let scores = values
            .into_iter()
            .filter_map(|(x, p)| {
                let fit =
                    Fit::train(&self.training, p, self.periodic, self.data_config.seed).ok()?;
                Some((
                    x,
                    metrics(&self.used, |v| fit.predict(v)).rmse,
                    metrics(&self.data.test, |v| fit.predict(v)).rmse,
                ))
            })
            .collect();
        (label, scores)
    }
}
