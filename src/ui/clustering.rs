use leptos::prelude::*;
use ml_lab_leptos::{
    clustering::{Dbscan, DensityLabel, KMeans, SingleLink, Step, observations},
    data::Point,
};

use super::num;

const COLORS: [&str; 5] = ["#8faaff", "#ffbc7b", "#6ce0c3", "#de9aef", "#fa8197"];

#[component]
pub fn Clustering() -> impl IntoView {
    let method = RwSignal::new(0_u8);
    let seed = RwSignal::new(72_u32);
    view! {
        <nav class="experiment-nav" aria-label="Метод кластеризации">
            {["k-means", "Иерархическая", "DBSCAN"].into_iter().enumerate().map(|(i,label)| view! {
                <button aria-pressed=move || (method.get() == i as u8).to_string()
                    on:click=move |_| method.set(i as u8)>{label}</button>
            }).collect_view()}
        </nav>
        {move || match method.get() {
            0 => view! { <Centroid seed=seed/> }.into_any(),
            method => {
                let density = method == 2;
                view! { <OtherMethod density=density seed=seed/> }.into_any()
            },
        }}
    }
}

#[component]
fn Centroid(seed: RwSignal<u32>) -> impl IntoView {
    let model = RwSignal::new(KMeans::example(3, 72));
    Effect::new(move |_| {
        let next_seed = seed.get();
        model.set(KMeans::example(
            model.with_untracked(|m| m.centers.len()),
            next_seed,
        ));
    });
    let reset = move |k| model.set(KMeans::example(k, seed.get_untracked()));
    let x = |v: f64| 400.0 + v * 87.0;
    let y = |v: f64| 225.0 - v * 70.0;
    view! {
        <section class="panel cluster-panel">
            <div class="eyebrow">"Обучение без учителя"</div>
            <h2>"Кластеризация · k-means"</h2>
            <div class="neural-actions">
                <label class="cluster-count">"Число кластеров k"
                    <select prop:value=move || model.with(|m| m.centers.len().to_string())
                        on:change=move |event| {
                            if let Ok(k) = event_target_value(&event).parse() { reset(k); }
                        }>
                        {[2, 3, 4, 5].into_iter().map(|k| view! { <option value=k>{k}</option> }).collect_view()}
                    </select>
                </label>
                <button on:click=move |_| model.update(KMeans::advance)
                    disabled=move || model.with(|m| m.step == Step::Converged)>"Следующий шаг"</button>
                <button on:click=move |_| reset(model.with_untracked(|m| m.centers.len()))>"Начать заново"</button>
                <button on:click=move |_| {
                    seed.update(|s| *s = s.wrapping_add(1));
                }>"Новые точки"</button>
            </div>
            <p role="status">{move || model.with(|m| match m.step {
                Step::Start => "Начальные центры заданы. Точки пока не распределены.",
                Step::Assigned => "Каждая точка назначена ближайшему центру.",
                Step::Recentered => "Центры переместились в среднее положение точек своих групп.",
                Step::Converged => "Группы больше не меняются. Алгоритм остановился.",
            })}</p>
            <svg viewBox="0 0 800 420" role="img" aria-label="Точки и центры кластеров">
                <rect x="1" y="1" width="798" height="418" rx="12" fill="#0b1321" stroke="#2b3b52"/>
                {move || model.with(|m| {
                    m.points.iter().enumerate().map(|(i, p)| {
                        let color = m.labels.get(i).map_or("#aabbd1", |&label| COLORS[label]);
                        view! { <circle cx=x(p.x) cy=y(p.y) r="4.5" fill=color opacity=".86"/> }
                    }).collect_view()
                })}
                {move || model.with(|m| {
                    m.centers.iter().enumerate().map(|(i, c)| {
                        let (cx, cy) = (x(c.x), y(c.y));
                        view! {
                            <path d=format!("M{} {}h20 M{} {}v20", cx-10.0, cy, cx, cy-10.0)
                                stroke=COLORS[i] stroke-width="4" stroke-linecap="round"/>
                            <text x=cx+14.0 y=cy-9.0 fill=COLORS[i] font-size="15">{i+1}</text>
                        }
                    }).collect_view()
                })}
            </svg>
            <div class="legend"><span>"● Наблюдение"</span><span>"+ Центр кластера"</span>
                <span>{move || model.with(|m| format!("Итераций: {}", m.iterations))}</span>
                {move || model.with(|m| m.squared_error().map(|v| format!("Сумма квадратов расстояний: {}",num(v))))}
            </div>
        </section>
    }
}

enum Grouping {
    Link(SingleLink),
    Density(Dbscan),
}

impl Grouping {
    fn done(&self) -> bool {
        match self {
            Self::Link(m) => m.done(),
            Self::Density(m) => m.done(),
        }
    }
    fn advance(&mut self) {
        match self {
            Self::Link(m) => m.advance(),
            Self::Density(m) => m.advance(),
        }
    }
}

#[component]
fn OtherMethod(density: bool, seed: RwSignal<u32>) -> impl IntoView {
    let k = RwSignal::new(3_usize);
    let epsilon = RwSignal::new(0.42_f64);
    let min_samples = RwSignal::new(5_usize);
    let sparse = RwSignal::new(false);
    let create = move || {
        let mut points = observations(seed.get());
        if sparse.get() {
            points.extend([(-3.8, 1.1), (0.0, -2.4), (3.8, 1.5)].map(|(x, y)| Point { x, y }));
        }
        let grouping = if density {
            Grouping::Density(Dbscan::new(&points, epsilon.get(), min_samples.get()))
        } else {
            Grouping::Link(SingleLink::new(&points, k.get()))
        };
        (points, grouping)
    };
    let state = RwSignal::new(create());
    Effect::new(move |_| state.set(create()));
    let x = |v: f64| 400.0 + v * 87.0;
    let y = |v: f64| 225.0 - v * 70.0;
    view! {
        <section class="panel cluster-panel">
            <div class="eyebrow">"Обучение без учителя"</div>
            <h2>{if density {"Кластеризация · DBSCAN"} else {"Иерархическая кластеризация · single-link"}}</h2>
            <p>{if density {
                "DBSCAN начинает группу там, где в заданном радиусе достаточно точек, и расширяет её через соседние плотные области."
            } else {
                "Сначала каждая точка образует свою группу. На каждом шаге сливаются две группы с ближайшей парой точек."
            }}</p>
            <div class="neural-actions">
                {if density {view!{
                    <label class="cluster-count">"Радиус ε " {move || num(epsilon.get())}
                        <input type="range" min="0.10" max="1.50" step="0.02" prop:value=move || epsilon.get()
                            on:input=move |ev|{if let Ok(v)=event_target_value(&ev).parse(){epsilon.set(v);}}/>
                    </label>
                    <label class="cluster-count">"Точек для плотной области " {move || min_samples.get()}
                        <input type="range" min="2" max="15" step="1" prop:value=move || min_samples.get()
                            on:input=move |ev|{if let Ok(v)=event_target_value(&ev).parse(){min_samples.set(v);}}/>
                    </label>
                    <label class="check"><input type="checkbox" prop:checked=move || sparse.get()
                        on:change=move |ev|sparse.set(event_target_checked(&ev))/>"Добавить редкие точки"</label>
                }.into_any()} else {view!{
                    <label class="cluster-count">"Остановиться на k группах"
                        <select prop:value=move || k.get().to_string() on:change=move |ev|{
                            if let Ok(v)=event_target_value(&ev).parse(){k.set(v);}
                        }>
                            {[2,3,4,5].into_iter().map(|v|view!{<option value=v>{v}</option>}).collect_view()}
                        </select>
                    </label>
                }.into_any()}}
                <button disabled=move ||state.with(|(_,m)|m.done())
                    on:click=move |_|state.update(|(_,m)|m.advance())>"Следующий шаг"</button>
                <button disabled=move ||state.with(|(_,m)|m.done())
                    on:click=move |_|state.update(|(_,m)|while !m.done(){m.advance();})>"До результата"</button>
                <button on:click=move |_|state.set(create())>"Начать заново"</button>
                <button on:click=move |_|seed.update(|s|*s=s.wrapping_add(1))>"Новые точки"</button>
            </div>
            <p role="status">{move ||state.with(|(_,m)|match m {
                Grouping::Link(m)=>if m.done(){format!("Готово. Групп: {}.",m.groups)}
                    else {format!("Групп: {}. Следующий шаг объединит ближайшую пару групп.",m.groups)},
                Grouping::Density(m)=> {
                    let noise=m.labels.iter().filter(|&&v|v==DensityLabel::Noise).count();
                    format!("{}Групп: {}. Помечено как шум: {}.",if m.done(){"Готово. "}else{""},m.groups,noise)
                }
            })}</p>
            <svg viewBox="0 0 800 420" role="img" aria-label="Группы наблюдений">
                <rect x="1" y="1" width="798" height="418" rx="12" fill="#0b1321" stroke="#2b3b52"/>
                {move || state.with(|(points,m)|{
                    let link=if let Grouping::Link(m)=m {m.last_merge}else{None};
                    let region=if let Grouping::Density(m)=m {m.last_seed}else{None};
                    let colors=if let Grouping::Link(m)=m {
                        let mut counts=std::collections::BTreeMap::new();
                        for &label in m.labels(){*counts.entry(label).or_insert(0)+=1;}
                        counts.into_iter().filter(|(_,n)|*n>1).enumerate()
                            .map(|(i,(label,_))|(label,COLORS[i%COLORS.len()])).collect()
                    }else{std::collections::BTreeMap::new()};
                    view!{
                        {link.map(|(a,b)|view!{<line x1=x(points[a].x) y1=y(points[a].y) x2=x(points[b].x) y2=y(points[b].y) stroke="#edf3fa" stroke-width="2"/>})}
                        {region.map(|i|view!{<ellipse cx=x(points[i].x) cy=y(points[i].y) rx=epsilon.get()*87.0 ry=epsilon.get()*70.0
                            fill="none" stroke="#edf3fa" stroke-dasharray="4 4"/>})}
                        {points.iter().enumerate().map(|(i,p)|{
                            let (color,noise)=match m{
                                Grouping::Link(m)=>(colors.get(&m.labels()[i]).copied().unwrap_or("#aabbd1"),false),
                                Grouping::Density(m)=>match m.labels[i]{
                                    DensityLabel::Pending=>("#aabbd1",false),
                                    DensityLabel::Noise=>("#64748b",true),
                                    DensityLabel::Cluster(k)=>(COLORS[k%COLORS.len()],false),
                                },
                            };
                            view!{<circle cx=x(p.x) cy=y(p.y) r="4.5" fill=if noise{"none"}else{color} stroke=color/>}
                        }).collect_view()}
                    }
                })}
            </svg>
            <div class="legend">{if density {"○ Шум · пунктир показывает радиус последней проверенной точки. Минимум соседей включает саму точку."}
                else {"Линия показывает пару точек, по которой произошло последнее слияние."}}</div>
        </section>
    }
}
