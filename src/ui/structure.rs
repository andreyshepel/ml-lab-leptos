use super::num;
use leptos::prelude::*;
use ml_lab_leptos::model::{Fit, Method, Model, Tree, nearest};

const EMPHASIS: &str = "#ffbc7b";
const QUIET: &str = "#42546f";

struct NodeView {
    x: f64,
    y: f64,
    width: f64,
    title: String,
    count: usize,
    active: bool,
}

fn tree_view(tree: &Tree, x: f64) -> impl IntoView + use<> {
    fn visit(
        tree: &Tree,
        x: f64,
        interval: (f64, f64),
        depth: usize,
        active: bool,
        nodes: &mut Vec<NodeView>,
        edges: &mut Vec<(f64, f64, f64, f64, bool)>,
    ) {
        let (lo, hi) = interval;
        let cx = (lo + hi) / 2.0;
        let y = 36.0 + depth as f64 * 84.0;
        let title = match &tree.branch {
            Some(_) if depth == 3 => "Поддерево…".to_owned(),
            Some(b) => format!("x ≤ {}?", num(b.threshold)),
            None => format!("ŷ = {}", num(tree.value)),
        };
        nodes.push(NodeView {
            x: cx,
            y,
            width: (hi - lo - 6.0).min(180.0),
            title,
            count: tree.count,
            active,
        });
        if depth < 3
            && let Some(b) = &tree.branch
        {
            for (child, (a, z), selected) in [
                (&b.left, (lo, cx), x <= b.threshold),
                (&b.right, (cx, hi), x > b.threshold),
            ] {
                let selected = selected && active;
                edges.push((cx, y + 22.0, (a + z) / 2.0, y + 62.0, selected));
                visit(child, x, (a, z), depth + 1, selected, nodes, edges);
            }
        }
    }
    let mut nodes = vec![];
    let mut edges = vec![];
    visit(tree, x, (10.0, 990.0), 0, true, &mut nodes, &mut edges);
    let mut path = vec![];
    let mut node = tree;
    while let Some(b) = &node.branch {
        let left = x <= b.threshold;
        path.push(format!(
            "x {} {}",
            if left { "≤" } else { ">" },
            num(b.threshold)
        ));
        node = if left { &b.left } else { &b.right };
    }
    view! {
        <div class="structure"><svg viewBox="0 0 1000 320" role="img" aria-label="Первые четыре уровня дерева решений">
            {edges.into_iter().map(|(x1,y1,x2,y2,hit)|view!{
                <line x1=x1 y1=y1 x2=x2 y2=y2 stroke=if hit{EMPHASIS}else{QUIET} stroke-width=if hit{2.4}else{1.0}/>
            }).collect_view()}
            {nodes.into_iter().map(|n|view!{
                <rect x=n.x-n.width/2.0 y=n.y-22.0 width=n.width height="44" rx="8" fill="#0b1321" stroke=if n.active{EMPHASIS}else{QUIET}/>
                <text x=n.x y=n.y-2.0 text-anchor="middle" fill="#edf3fa" font-size="12">{n.title}</text>
                <text x=n.x y=n.y+14.0 text-anchor="middle" fill="#aabbd1" font-size="10">{format!("{} наблюдений",n.count)}</text>
            }).collect_view()}
        </svg></div>
        <p class="small">{format!("Путь точки: {} → ŷ = {}.",if path.is_empty(){"корневой лист".to_owned()}else{path.join(" → ")},num(node.value))}</p>
    }
}

fn network_view(
    net: &ml_lab_leptos::model::Network,
    x: f64,
    stage: Option<usize>,
) -> impl IntoView + use<> {
    let values = net.forward(x);
    let layers = net.layers.len();
    let position = |layer: usize, j: usize, count: usize| {
        let x = 60.0 + 880.0 * layer as f64 / layers as f64;
        let y = if count == 1 {
            190.0
        } else {
            58.0 + 270.0 * j as f64 / (count.min(8) - 1) as f64
        };
        (x, y)
    };
    let mut edges = vec![];
    for (i, layer) in net.layers.iter().enumerate() {
        let max = layer.weights[..layer.input * layer.output]
            .iter()
            .map(|w| w.abs())
            .fold(1e-8, f64::max);
        for j in 0..layer.output.min(8) {
            for k in 0..layer.input.min(8) {
                let (x1, y1) = position(i, k, layer.input);
                let (x2, y2) = position(i + 1, j, layer.output);
                let w = layer.weights[j * layer.input + k];
                let color = if w >= 0.0 { "#6ce0c3" } else { "#de9aef" };
                let opacity = if stage.is_none_or(|s| i < s) {
                    0.18 + 0.5 * w.abs() / max
                } else {
                    0.06
                };
                edges.push(view!{<line x1=x1 y1=y1 x2=x2 y2=y2 stroke=color opacity=opacity><title>{num(w)}</title></line>});
            }
        }
    }
    view! {
        <div class="structure"><svg viewBox="0 0 1000 385" role="img" aria-label="Слои нейросети и текущие активации">
            {edges}
            {values.into_iter().enumerate().map(|(i,values)|{
                let count=values.len();
                let visible=stage.is_none_or(|s|i<=s);
                let (cx,_)=position(i,0,count);
                view!{
                    <text x=cx y="24" fill="#aabbd1" text-anchor="middle" font-size="13">
                        {if i==0{"Вход x/3".to_owned()}else if i==layers{"Выход ŷ".to_owned()}else{format!("Слой {i}")}}
                    </text>
                    {values.into_iter().take(8).enumerate().map(|(j,value)|{
                        let (cx,cy)=position(i,j,count);
                        view!{
                            <circle cx=cx cy=cy r="17" fill="#0b1321" stroke=if visible{EMPHASIS}else{QUIET}/>
                            <text x=cx y=cy+4.0 fill="#edf3fa" text-anchor="middle" font-size="10">{if visible{num(value)}else{"·".to_owned()}}</text>
                        }
                    }).collect_view()}
                    {(count>8).then(||view!{<text x=cx y="365" fill="#aabbd1" text-anchor="middle" font-size="12">{format!("+ {} нейронов",count-8)}</text>})}
                }
            }).collect_view()}
        </svg></div>
        <p class="small">"Цвет связей показывает знак веса. Схема показывает до восьми нейронов в каждом слое."</p>
    }
}

pub fn formula(method: Method) -> &'static str {
    match method {
        Method::Polynomial => "ŷ = Σ wᵢ Tᵢ(z), где z — нормализованный вход",
        Method::Knn => "ŷ = среднее y среди k ближайших наблюдений",
        Method::Tree => "ŷ = среднее y в выбранном листе",
        Method::Forest => "ŷ = среднее предсказаний всех деревьев",
        Method::Rbf | Method::Gp => "ŷ = Σ wᵢ exp(−(x−cᵢ)² / (2ℓ²))",
        Method::Neural => "ŷ = линейный выход после последовательности аффинных слоёв и активаций",
    }
}

pub fn parameters(fit: &Fit) -> Vec<(String, String)> {
    match &fit.model {
        Model::Polynomial {
            coefficients,
            center,
            scale,
        } => std::iter::once((
            "z".to_owned(),
            format!("(x − {}) / {}", num(*center), num(*scale)),
        ))
        .chain(
            coefficients
                .iter()
                .enumerate()
                .map(|(i, w)| (format!("w{i}"), num(*w))),
        )
        .collect(),
        Model::Knn { points, k } => std::iter::once(("k".to_owned(), k.to_string()))
            .chain(points.iter().map(|p| (num(p.x), num(p.y))))
            .collect(),
        Model::Tree(tree) => {
            fn rules(tree: &Tree, lo: f64, hi: f64, out: &mut Vec<(String, String)>) {
                if let Some(branch) = &tree.branch {
                    rules(&branch.left, lo, branch.threshold, out);
                    rules(&branch.right, branch.threshold, hi, out);
                } else {
                    out.push((format!("{} < x ≤ {}", num(lo), num(hi)), num(tree.value)));
                }
            }
            let mut rows = vec![];
            rules(tree, f64::NEG_INFINITY, f64::INFINITY, &mut rows);
            rows
        }
        Model::Forest(trees) => trees
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    format!("Дерево {}", i + 1),
                    format!("{} наблюдений в bootstrap-выборке", t.count),
                )
            })
            .collect(),
        Model::Rbf {
            centers,
            coefficients,
            ..
        }
        | Model::Gp {
            centers,
            coefficients,
            ..
        } => centers
            .iter()
            .zip(coefficients)
            .map(|(&c, &w)| (num(c), num(w)))
            .collect(),
        Model::Neural(net) => {
            net.layers
                .iter()
                .enumerate()
                .flat_map(|(i, layer)| {
                    layer.weights.iter().enumerate().map(move |(j, &w)| {
                        (format!("Слой {} · параметр {}", i + 1, j + 1), num(w))
                    })
                })
                .collect()
        }
    }
}

fn neighbor_view(
    points: &[ml_lab_leptos::data::Point],
    k: usize,
    x: f64,
    prediction: f64,
) -> impl IntoView + use<> {
    let lo = points.first().map_or(x, |p| p.x.min(x)) - 0.1;
    let hi = points.last().map_or(x, |p| p.x.max(x)) + 0.1;
    let bound = points.iter().map(|p| p.y.abs()).fold(1.0, f64::max);
    let px = |x: f64| 44.0 + 912.0 * (x - lo) / (hi - lo);
    let py = |y: f64| 108.0 - 84.0 * y / bound;
    let selected: Vec<_> = nearest(points, k, x).collect();
    view! {
        <svg viewBox="0 0 1000 220" role="img" aria-label="Ближайшие соседи проверочной точки">
            <line x1=px(x) x2=px(x) y1="16" y2="200" stroke=QUIET stroke-dasharray="4 4"/>
            {points.iter().map(|point| view!{
                <circle cx=px(point.x) cy=py(point.y) r="4" fill=QUIET/>
            }).collect_view()}
            {selected.iter().map(|point| view!{
                <line x1=px(point.x) y1=py(point.y) x2=px(x) y2=py(prediction) stroke=EMPHASIS opacity=".35"/>
                <circle cx=px(point.x) cy=py(point.y) r="5" fill=EMPHASIS/>
            }).collect_view()}
            <circle cx=px(x) cy=py(prediction) r="6" fill="#6ce0c3"/>
        </svg>
        <p class="small">{format!("Оранжевым выделены {} ближайших наблюдений. Зелёная точка — среднее их значений.",selected.len())}</p>
        <div class="table-wrap"><table>
            <thead><tr><th>"Координата соседа"</th><th>"Наблюдение"</th><th>"Расстояние"</th></tr></thead>
            <tbody>{selected.into_iter().map(|p|view!{<tr><td>{num(p.x)}</td><td>{num(p.y)}</td><td>{num((p.x-x).abs())}</td></tr>}).collect_view()}</tbody>
        </table></div>
    }
}

fn component_view(values: Vec<f64>, ensemble: bool, prediction: f64) -> impl IntoView {
    let mut items: Vec<_> = values
        .iter()
        .take(7)
        .enumerate()
        .map(|(i, &v)| {
            (
                format!(
                    "{} {}",
                    if ensemble {
                        "Дерево"
                    } else {
                        "Компонент"
                    },
                    i + 1
                ),
                v,
            )
        })
        .collect();
    if values.len() > 7 {
        let remaining = &values[7..];
        let sum = remaining.iter().sum::<f64>();
        let label = if ensemble {
            "Среднее остальных деревьев"
        } else {
            "Сумма остальных вкладов"
        };
        items.push((
            label.to_owned(),
            if ensemble {
                sum / remaining.len() as f64
            } else {
                sum
            },
        ));
    }
    let scale = items
        .iter()
        .map(|(_, value)| value.abs())
        .fold(1e-8, f64::max);
    let height = items.len() * 30 + 34;
    view! {
        <svg viewBox=format!("0 0 1000 {height}") role="img" aria-label="Вклады компонентов в предсказание">
            <line x1="560" x2="560" y1="8" y2=height-22 stroke=QUIET/>
            {items.into_iter().enumerate().map(|(i,(label,value))|{
                let y=i*30+20;
                let width=260.0*value.abs()/scale;
                view!{
                    <text x="10" y=y+4 fill="#aabbd1" font-size="13">{label}</text>
                    <rect x=if value<0.0{560.0-width}else{560.0} y=y-8 width=width height="16"
                        rx="3" fill=if value<0.0{"#de9aef"}else{"#6ce0c3"}/>
                    <text x="990" y=y+4 fill="#edf3fa" font-size="13" text-anchor="end">{num(value)}</text>
                }
            }).collect_view()}
            <text x="990" y=height-6 fill=EMPHASIS font-size="14" text-anchor="end">{format!("Итог: ŷ = {}",num(prediction))}</text>
        </svg>
    }
}

pub fn structure(fit: &Fit, probe: f64, stage: Option<usize>) -> impl IntoView + use<> {
    let x = fit.input(probe);
    match &fit.model {
        Model::Tree(tree) => tree_view(tree, x).into_any(),
        Model::Neural(net) => network_view(net, x, stage).into_any(),
        Model::Knn { points, k } => neighbor_view(points, *k, x, fit.predict(probe)).into_any(),
        model => component_view(
            model.contributions(x),
            matches!(model, Model::Forest(_)),
            fit.predict(probe),
        )
        .into_any(),
    }
}
