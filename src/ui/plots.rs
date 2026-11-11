use super::{Display, num, state::Lab};
use leptos::prelude::*;
use ml_lab_leptos::{
    data::{Point, grid, target},
    model::{Fit, Model},
};

pub const TRAIN: &str = "#8faaff";
pub const TEST: &str = "#de9aef";
const TARGET: &str = "#6ce0c3";
const MODEL: &str = "#ffbc7b";
const OUTLIER: &str = "#ff718a";

struct Axes {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    height: f64,
}

impl Axes {
    fn x(&self, x: f64) -> f64 {
        64.0 + 812.0 * (x - self.min_x) / (self.max_x - self.min_x)
    }
    fn y(&self, y: f64) -> f64 {
        22.0 + (self.height - 60.0) * (self.max_y - y) / (self.max_y - self.min_y)
    }
    fn path(&self, points: impl IntoIterator<Item = (f64, f64)>) -> String {
        points
            .into_iter()
            .enumerate()
            .map(|(i, (x, y))| {
                format!(
                    "{}{:.3},{:.3}",
                    if i == 0 { "M" } else { "L" },
                    self.x(x),
                    self.y(y)
                )
            })
            .collect()
    }
    fn grid(&self) -> impl IntoView + use<> {
        (0..=4).map(|i|{
            let t=f64::from(i)/4.0;
            let value=self.max_y-(self.max_y-self.min_y)*t;
            let y=self.y(value);
            let x=self.min_x+(self.max_x-self.min_x)*t;
            view! {
                <line x1="64" x2="876" y1=y y2=y stroke="#2b3b52"/>
                <text x="58" y=y+4.0 text-anchor="end" fill="#aabbd1" font-size="11">{num(value)}</text>
                <text x=self.x(x) y=self.height-12.0 text-anchor="middle" fill="#aabbd1" font-size="11">{num(x)}</text>
            }
        }).collect_view()
    }
}

fn observation_bound(points: &[Point]) -> f64 {
    points.iter().map(|p| p.y.abs() * 1.08).fold(1.6, f64::max)
}

pub(super) fn observations_plot(points: &[Point]) -> impl IntoView + use<> {
    let bound = observation_bound(points);
    let axes = Axes {
        min_x: -3.0,
        max_x: 3.0,
        min_y: -bound,
        max_y: bound,
        height: 370.0,
    };
    let dots = points
        .iter()
        .map(|p| view! { <circle cx=axes.x(p.x) cy=axes.y(p.y) r="4.4" fill=TRAIN stroke=TRAIN/> })
        .collect_view();
    view! {
        <svg class="plot" viewBox="0 0 900 370" role="img" aria-label="График исходных наблюдений">
            <defs><clipPath id="regression-observations-area"><rect x="64" y="22" width="812" height="310"/></clipPath></defs>
            {axes.grid()}
            <g clip-path="url(#regression-observations-area)">{dots}</g>
        </svg>
    }
}

pub(super) fn regression_plot(
    lab: &Lab,
    fit: Option<&Fit>,
    display: Display,
    probe: f64,
    mark: Callback<usize>,
) -> impl IntoView + use<> {
    let (lo, hi) = if display.extrapolate {
        (-9.0, 9.0)
    } else {
        (-3.0, 3.0)
    };
    let xs: Vec<_> = grid(lo, hi, 501).collect();
    let samples: Vec<_> = fit
        .filter(|_| display.model)
        .map(|fit| {
            xs.iter()
                .map(|&x| (x, fit.predict(x), fit.uncertainty(x)))
                .collect()
        })
        .unwrap_or_default();
    let mut bound = 1.6_f64;
    if display.autoscale {
        bound = observation_bound(&lab.data.train);
        for &(_, y, sd) in &samples {
            bound = bound.max((y.abs() + 1.96 * sd.unwrap_or(0.0)) * 1.08);
        }
    }
    let a = Axes {
        min_x: lo,
        max_x: hi,
        min_y: -bound,
        max_y: bound,
        height: 370.0,
    };
    let target_path = display
        .target
        .then(|| a.path(xs.iter().map(|&x| (x, target(x)))));
    let prediction = (!samples.is_empty()).then(|| a.path(samples.iter().map(|&(x, y, _)| (x, y))));
    let band = samples.first().and_then(|s| s.2).map(|_| {
        let upper = samples
            .iter()
            .map(|&(x, y, sd)| (x, y + 1.96 * sd.unwrap_or(0.0)));
        let lower = samples
            .iter()
            .rev()
            .map(|&(x, y, sd)| (x, y - 1.96 * sd.unwrap_or(0.0)));
        format!("{}Z", a.path(upper.chain(lower)))
    });
    let parts = if display.model && display.parts {
        fit.map(|fit| {
            if let Model::Tree(tree) = &fit.model {
                fn thresholds(tree: &ml_lab_leptos::model::Tree, out: &mut Vec<f64>) {
                    if let Some(b) = &tree.branch {
                        out.push(b.threshold);
                        thresholds(&b.left, out);
                        thresholds(&b.right, out);
                    }
                }
                let mut splits = vec![];
                thresholds(tree, &mut splits);
                let shifts = if fit.periodic { -4..=4 } else { 0..=0 };
                splits
                    .into_iter()
                    .flat_map(|x| shifts.clone().map(move |k| x + 2.0 * f64::from(k)))
                    .filter(|&x| x >= lo && x <= hi)
                    .map(|x| a.path([(x, -bound), (x, bound)]))
                    .collect::<Vec<_>>()
            } else {
                let values: Vec<_> = xs
                    .iter()
                    .map(|&x| fit.model.contributions(fit.input(x)))
                    .collect();
                let count = values.first().map_or(0, Vec::len).min(12);
                (0..count)
                    .map(|i| a.path(xs.iter().zip(&values).map(|(&x, v)| (x, v[i]))))
                    .collect()
            }
        })
        .unwrap_or_default()
    } else {
        vec![]
    };
    let tests =
        display.test_points.then(|| {
            lab.data.test.iter().map(|p|view!{
        <circle cx=a.x(p.x) cy=a.y(p.y) r="2.6" fill="none" stroke=TEST opacity="0.6"/>
    }).collect_view()
        });
    let train=lab.data.train.iter().enumerate().map(|(i,p)|{
        let marked=lab.marked.contains(&i);
        let color=if marked {OUTLIER}else{TRAIN};
        let (x,y)=(a.x(p.x),a.y(p.y));
        let toggle=move || mark.run(i);
        view! {
            <g class="plot-point" role="button" tabindex="0" aria-label=format!("Обучающая точка {}",i+1)
                aria-pressed=marked.to_string() on:click=move |_|toggle()
                on:keydown=move |ev|{if ev.key()=="Enter"||ev.key()==" "{ev.prevent_default();toggle();}}>
                <circle cx=x cy=y r="9" fill="transparent"/>
                <circle cx=x cy=y r=if marked{7.0}else{4.4} stroke=color
                    fill=if marked&&lab.exclude{"none"}else{color}/>
                {marked.then(||view!{<path d=format!("M{},{}l8,8m-8,0l8,-8",x-4.0,y-4.0) stroke=if lab.exclude{OUTLIER}else{"#0b1321"} stroke-width="2"/>})}
            </g>
        }
    }).collect_view();
    let probe=fit.filter(|_|display.model).map(|fit|view!{
        <line x1=a.x(probe) x2=a.x(probe) y1="22" y2="332" stroke="#aabbd1" stroke-dasharray="4 5"/>
        <circle cx=a.x(probe) cy=a.y(fit.predict(probe)) r="4" fill=MODEL/>
    });
    let shades = display.extrapolate.then(|| {
        [(-9.0, -3.0), (3.0, 9.0)]
            .into_iter()
            .map(|(x, end)| {
                view! {
                    <rect x=a.x(x) width=a.x(end)-a.x(x) y="22" height="310" fill="#25354a"/>
                }
            })
            .collect_view()
    });
    let clipped = lab.data.train.iter().any(|p| p.y.abs() > bound)
        || samples
            .iter()
            .any(|(_, y, sd)| y.abs() + 1.96 * sd.unwrap_or(0.0) > bound);
    view! {
        <svg class="plot" viewBox="0 0 900 370" role="group" aria-label="Целевая функция, модель и обучающие точки">
            <defs><clipPath id="regression-model-area"><rect x="64" y="22" width="812" height="310"/></clipPath></defs>
            {shades}{a.grid()}
            <g clip-path="url(#regression-model-area)">
                {band.map(|d|view!{<path d=d fill="#72b7d4" opacity=".18"/>})}
                {parts.into_iter().map(|d|view!{<path d=d fill="none" stroke=TRAIN opacity=".32"/>}).collect_view()}
                {target_path.map(|d|view!{<path d=d fill="none" stroke=TARGET stroke-width="2.4"/>})}
                {prediction.map(|d|view!{<path d=d fill="none" stroke=MODEL stroke-width="2.4"/>})}
                {tests}{probe}{train}
            </g>
        </svg>
        <div class="legend">
            {display.target.then(||view!{<span class="truth">"━ Целевая функция"</span>})}
            {display.model.then(||view!{<span class="model">"━ Модель"</span>})}
            <span class="train">"● Обучение"</span>
            {display.test_points.then(||view!{<span class="test">"○ Тест"</span>})}
            {(!lab.marked.is_empty()).then(||view!{<span style:color=OUTLIER>"× Отмеченные выбросы"</span>})}
            {samples.first().is_some_and(|s|s.2.is_some()).then(||view!{<span>"Интервал 95%"</span>})}
        </div>
        {clipped.then(||view!{<p class="warning">"Часть данных выходит за пределы графика. Включите автомасштаб Y."</p>})}
    }
}

pub fn diagnostic(
    label: &'static str,
    points: &[(f64, f64, f64)],
    test: bool,
) -> impl IntoView + use<> {
    let log = |v: f64| v.max(1e-8).log10();
    let range = points.iter().flat_map(|&(_, train, eval)| {
        [Some(log(train)), test.then(|| log(eval))]
            .into_iter()
            .flatten()
    });
    let min = range.clone().fold(-2.0, f64::min).floor();
    let max = range.fold(0.0, f64::max).ceil();
    let lo = points.first().map_or(0.0, |p| p.0);
    let hi = points.last().map_or(1.0, |p| p.0).max(lo + 1.0);
    let a = Axes {
        min_x: lo,
        max_x: hi,
        min_y: min,
        max_y: max,
        height: 230.0,
    };
    let series = |test| {
        a.path(
            points
                .iter()
                .map(|&(x, y, z)| (x, log(if test { z } else { y }))),
        )
    };
    let train = series(false);
    let test_path = test.then(|| series(true));
    let single = if let [(x, y, z)] = points {
        Some((a.x(*x), a.y(log(*y)), a.y(log(*z))))
    } else {
        None
    };
    view! {
        <p class="small">{label}" · по вертикали log₁₀(RMSE)"</p>
        <svg viewBox="0 0 900 230" role="img" aria-label="График ошибки">
            {a.grid()}
            <path d=train fill="none" stroke=TRAIN stroke-width="2"/>
            {test_path.map(|d|view!{<path d=d fill="none" stroke=TEST stroke-width="2"/>})}
            {single.map(|(x,y,z)|view!{<circle cx=x cy=y r="4" fill=TRAIN/>{test.then(||view!{<circle cx=x cy=z r="4" fill=TEST/>})}})}
        </svg>
    }
}
