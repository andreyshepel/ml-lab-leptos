use std::time::Duration;

use leptos::{leptos_dom::helpers::set_interval_with_handle, prelude::*};
use ml_lab_leptos::reinforcement::{
    ARROWS, Episode, Experiment, GOAL, HEIGHT, LIMIT, Method, NetworkView, Outcome, START, TRAP,
    WALLS, WIDTH,
};

const SEED: u32 = 72;
const CELL: usize = 64;

fn fresh() -> [Experiment; 3] {
    Method::ALL.map(|method| Experiment::new(method, SEED))
}

#[component]
pub fn Reinforcement() -> impl IntoView {
    let experiments = RwSignal::new(fresh());
    let selected = RwSignal::new(0_usize);
    let epsilon = RwSignal::new(0.45_f64);
    let step_cost = RwSignal::new(0.04_f64);
    let preview = RwSignal::new(false);
    let inspected = RwSignal::new(None::<usize>);
    let run_to = RwSignal::new(None::<usize>);
    let timer = set_interval_with_handle(
        move || {
            let Some(goal) = run_to.get_untracked() else {
                return;
            };
            let index = selected.get_untracked();
            let (epsilon, cost) = (epsilon.get_untracked(), step_cost.get_untracked());
            let mut more = false;
            experiments.update(|models| {
                more = models[index].train_batch(goal, LIMIT, epsilon, cost);
            });
            if !more {
                run_to.set(None);
            }
        },
        Duration::from_millis(40),
    );
    let timer_available = timer.is_ok();
    if let Ok(handle) = timer {
        on_cleanup(move || handle.clear());
    }
    let train = move |episodes| {
        preview.set(false);
        let index = selected.get_untracked();
        let goal = experiments.with_untracked(|models| models[index].completed + episodes);
        run_to.set(Some(goal));
    };
    view! {
        <nav class="experiment-nav" aria-label="Метод обучения с подкреплением">
            {Method::ALL.into_iter().enumerate().map(|(i,method)| view! {
                <button aria-pressed=move || (selected.get()==i).to_string()
                    on:click=move |_| { run_to.set(None); selected.set(i); preview.set(false); }>{method.name()}</button>
            }).collect_view()}
        </nav>
        <section class="panel rl-panel">
            <div class="eyebrow">"Обучение с подкреплением"</div>
            <h2>{move || Method::ALL[selected.get()].name()}</h2>
            <p>{move || match Method::ALL[selected.get()] {
                Method::QLearning => "Таблица Q хранит оценки ходов. После действия агент обновляет оценку с учётом лучшего хода из следующей клетки.",
                Method::Sarsa => "Таблица Q обновляется по ходу, который агент действительно выбрал следующим. Этот ход может быть случайным.",
                Method::Dqn => "Нейросеть оценивает четыре хода. Она учится на переходах из буфера опыта и периодически обновляет сеть для расчёта целей обучения.",
            }}</p>
            <div class="rl-layout">
                <div>
                    {move || experiments.with(|models| board(&models[selected.get()], preview.get(), inspected))}
                    <div class="legend">
                        <span>"S Старт"</span><span class="truth">"+1 Цель"</span>
                        <span style="color:#fa8197">"−1 Ловушка"</span><span class="model">"● Агент и маршрут"</span>
                        <span>{format!("{} × {} клеток · до {} ходов", WIDTH, HEIGHT, LIMIT)}</span>
                    </div>
                </div>
                <div>
                    <label class="cluster-count">"Вероятность случайного хода ε "
                        {move || format!("{:.0}%",100.0*epsilon.get())}
                        <input type="range" min="0" max="1" step="0.05" prop:value=move || epsilon.get()
                            on:input=move |ev| {if let Ok(value)=event_target_value(&ev).parse(){run_to.set(None); epsilon.set(value);}}/>
                    </label>
                    <label class="cluster-count">"Штраф за обычный шаг"
                        <select prop:value=move || step_cost.get().to_string() on:change=move |ev|{
                            if let Ok(value)=event_target_value(&ev).parse::<f64>() && value!=step_cost.get_untracked() {
                                run_to.set(None);step_cost.set(value);experiments.set(fresh());preview.set(false);
                            }
                        }>
                            <option value="0">"0"</option>
                            <option value="0.04">"−0,04"</option>
                            <option value="0.15">"−0,15"</option>
                        </select>
                    </label>
                    <p class="small">"Смена штрафа начинает обучение заново для всех трёх методов."</p>
                    <div class="neural-actions">
                        <button disabled=move || run_to.get().is_some() on:click=move |_| {
                            preview.set(false);
                            let index = selected.get_untracked();
                            experiments.update(|models| {
                                models[index].step(epsilon.get_untracked(), step_cost.get_untracked());
                            });
                        }>"Следующий ход"</button>
                        <button disabled=move || !timer_available || run_to.get().is_some()
                            on:click=move |_|train(1)>"До конца эпизода"</button>
                        <button disabled=move || !timer_available || run_to.get().is_some()
                            on:click=move |_|train(100)>"Ещё 100 эпизодов"</button>
                        <button disabled=move || run_to.get().is_none()
                            on:click=move |_|run_to.set(None)>"Остановить обучение"</button>
                        <button aria-pressed=move ||preview.get().to_string()
                            on:click=move |_|{
                                run_to.set(None);
                                preview.update(|v|*v = !*v);
                            }>"Маршрут без случайных ходов"</button>
                        <button on:click=move |_|{
                            run_to.set(None);
                            let i=selected.get_untracked();
                            experiments.update(|models|models[i]=Experiment::new(Method::ALL[i],SEED));
                            preview.set(false);
                        }>"Сбросить метод"</button>
                    </div>
                    <p class="small" hidden=timer_available>
                        "Браузер не запустил таймер обучения. Доступны ручные ходы."
                    </p>
                    <p role="status" class="rl-status">{move || experiments.with(|models|
                        status(&models[selected.get()],preview.get())
                    )}</p>
                    {move || experiments.with(|models| {
                        let model=&models[selected.get()];
                        let cell=if preview.get(){START}else{model.position};
                        (!matches!(cell,GOAL|TRAP)).then(||view!{
                            <div class="legend">
                                <span>{if preview.get(){"Оценки Q на старте"}else{"Оценки Q в текущей клетке"}}</span>
                                {model.values(cell).into_iter().zip(ARROWS).map(|(v,arrow)|view!{
                                    <span>{format!("{arrow} {v:.2}")}</span>
                                }).collect_view()}
                            </div>
                        })
                    })}
                </div>
            </div>
            <section class="rl-inspector">
                <div class="method-header">
                    <h3>"Внутри агента"</h3>
                    <button disabled=move || inspected.get().is_none()
                        on:click=move |_| inspected.set(None)>"Следовать за агентом"</button>
                </div>
                <p class="small">"Выбери свободную клетку на поле, чтобы посмотреть оценки ходов из неё."</p>
                {move || experiments.with(|models| {
                    let model = &models[selected.get()];
                    let cell = inspected.get().unwrap_or_else(|| {
                        if preview.get() || matches!(model.position, GOAL | TRAP) { START } else { model.position }
                    });
                    let label = if inspected.get().is_some() {
                        "Выбранная клетка"
                    } else if cell == START {
                        "Стартовая клетка"
                    } else {
                        "Текущая клетка"
                    };
                    view! {
                        <p class="small">{format!("{label}: {}", cell_name(cell))}</p>
                        {if let Some(network) = model.network_view(cell) {
                            network_diagram(network, model.values(cell), model.best_action(cell)).into_any()
                        } else {
                            q_table(model, cell, inspected).into_any()
                        }}
                    }
                })}
            </section>
            <h3>"Награда за эпизод"</h3>
            {move || experiments.with(|models| history(&models[selected.get()].history))}
        </section>
    }
}

fn status(model: &Experiment, preview: bool) -> String {
    if preview {
        let route = model.greedy_path();
        return match route.last().copied() {
            Some(GOAL) => format!("Цель достигнута за {} ходов.", route.len() - 1),
            Some(TRAP) => "Маршрут ведёт в ловушку.".to_owned(),
            _ => "Маршрут остановился на повторной клетке или исчерпал лимит ходов.".to_owned(),
        };
    }
    let result = match model.outcome {
        Some(Outcome::Goal) => " Цель достигнута.",
        Some(Outcome::Trap) => " Агент попал в ловушку.",
        Some(Outcome::Limit) => " Достигнут лимит ходов.",
        None => "",
    };
    let action = model
        .last_action
        .zip(model.last_reward)
        .map_or_else(String::new, |(a, r)| {
            format!(" Последний ход {} ({r:+.2}).", ARROWS[a])
        });
    format!(
        "Эпизодов завершено: {}. Ходов: {}. Награда: {:+.2}.{action}{result}",
        model.completed,
        model.path.len() - 1,
        model.reward
    )
}

fn coordinates(cell: usize) -> (usize, usize) {
    (cell % WIDTH * CELL, cell / WIDTH * CELL)
}

fn board(
    model: &Experiment,
    preview: bool,
    inspected: RwSignal<Option<usize>>,
) -> impl IntoView + use<> {
    let selected = inspected.get();
    let path = if preview {
        model.greedy_path()
    } else {
        model.path.clone()
    };
    let points = path
        .iter()
        .map(|&cell| {
            let (x, y) = coordinates(cell);
            format!("{},{}", x + CELL / 2, y + CELL / 2)
        })
        .collect::<Vec<_>>()
        .join(" ");
    let (x, y) = coordinates(path.last().copied().unwrap_or(START));
    view! {
        <svg viewBox=format!("0 0 {} {}",WIDTH*CELL,HEIGHT*CELL) role="group" aria-label="Лабиринт агента">
            {(0..WIDTH*HEIGHT).map(|i|{
                let (x,y)=coordinates(i);
                let (fill,label)=match i {
                    START=>("#233c59","S"),
                    GOAL=>("#24493e","+1"),
                    TRAP=>("#572d40","−1"),
                    _ if WALLS.contains(&i)=>("#536278",""),
                    _=>("#0b1321",""),
                };
                let selectable = !matches!(i, GOAL | TRAP) && !WALLS.contains(&i);
                let active = selected == Some(i);
                view!{
                    <rect x=x+2 y=y+2 width=CELL-4 height=CELL-4 rx="6" fill=fill
                        stroke=if active {"#edf3fa"} else {"#3a4c67"} stroke-width=if active {3} else {1}
                        class:rl-cell=selectable role=if selectable {Some("button")} else {None}
                        tabindex=if selectable {Some(0)} else {None}
                        aria-label=cell_name(i)
                        on:click=move |_| if selectable { inspected.set(Some(i)); }
                        on:keydown=move |ev| if selectable && matches!(ev.key().as_str(),"Enter"|" ") {
                            ev.prevent_default(); inspected.set(Some(i));
                        }/>
                    <text pointer-events="none" x=x+CELL/2 y=y+20 text-anchor="middle" fill="#edf3fa" font-size="14">{label}</text>
                }
            }).collect_view()}
            <polyline pointer-events="none" points=points fill="none" stroke="#ffbc7b" stroke-width="4"
                stroke-linecap="round" stroke-linejoin="round"/>
            <circle pointer-events="none" cx=x+CELL/2 cy=y+CELL/2 r="10" fill="#ffbc7b" stroke="#0b1321" stroke-width="2"/>
        </svg>
    }
}

fn history(episodes: &[Episode]) -> impl IntoView + use<> {
    if episodes.is_empty() {
        return view! { <p class="small">"График появится после первого завершённого эпизода."</p> }.into_any();
    }
    let low = episodes.iter().map(|e| e.reward).fold(-1.0, f64::min);
    let position = |i: usize, reward: f64| {
        (
            46.0 + 720.0 * i as f64 / (episodes.len() - 1).max(1) as f64,
            18.0 + 100.0 * (1.0 - reward) / (1.0 - low),
        )
    };
    let points = episodes
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let (x, y) = position(i, e.reward);
            format!("{x:.2},{y:.2}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let (_, zero) = position(0, 0.0);
    let reached = episodes
        .iter()
        .filter(|e| e.outcome == Outcome::Goal)
        .count();
    view! {
        <svg viewBox="0 0 800 140" role="img" aria-label="Награда за последние эпизоды">
            <line x1="46" x2="766" y1=zero y2=zero stroke="#64748b" stroke-dasharray="4 4"/>
            <text x="36" y=zero+4.0 text-anchor="end" font-size="12" fill="#aabbd1">"0"</text>
            <polyline points=points stroke="#8faaff" stroke-width="2" fill="none"/>
            {episodes.iter().enumerate().map(|(i,e)|{
                let (x,y)=position(i,e.reward);
                let color=if e.outcome==Outcome::Goal{"#6ce0c3"}else{"#fa8197"};
                view!{<circle cx=x cy=y r="3" fill=color><title>{format!("Награда {:+.2}, {} ходов",e.reward,e.steps)}</title></circle>}
            }).collect_view()}
        </svg>
        <p class="small">{format!("Цель достигнута в {reached} из последних {} эпизодов.", episodes.len())}</p>
    }.into_any()
}

fn cell_name(cell: usize) -> String {
    format!("строка {}, столбец {}", cell / WIDTH + 1, cell % WIDTH + 1)
}

fn q_table(
    model: &Experiment,
    cell: usize,
    inspected: RwSignal<Option<usize>>,
) -> impl IntoView + use<> {
    view! {
        <div class="table-wrap rl-q-table">
            <table aria-label="Таблица оценок Q">
                <thead><tr><th scope="col">"Состояние"</th>
                    {ARROWS.into_iter().map(|arrow|view!{<th scope="col">{arrow}</th>}).collect_view()}
                    <th scope="col">"Лучший ход"</th>
                </tr></thead>
                <tbody>
                    {(0..WIDTH * HEIGHT).filter(|i| !WALLS.contains(i)).map(|i| {
                        let terminal = matches!(i, GOAL | TRAP);
                        let best = model.best_action(i);
                        let scores = model.values(i);
                        let tied = scores.iter().filter(|&&value| value == scores[best]).count() > 1;
                        view! {
                            <tr class:selected=i==cell>
                                <th scope="row">
                                    {if terminal {
                                        cell_name(i).into_any()
                                    } else {
                                        view! { <button on:click=move |_|inspected.set(Some(i))>{cell_name(i)}</button> }.into_any()
                                    }}
                                    {match i { START=>" · старт", GOAL=>" · цель", TRAP=>" · ловушка", _=>"" }}
                                </th>
                                {scores.into_iter().enumerate().map(|(a,value)|view!{
                                    <td class:truth=!terminal&&!tied&&a==best>
                                        {if terminal {"—".to_owned()} else {format!("{value:+.3}")}}
                                    </td>
                                }).collect_view()}
                                <td>{if terminal {"Конец эпизода"} else if tied {"Ничья"} else {ARROWS[best]}}</td>
                            </tr>
                        }
                    }).collect_view()}
                </tbody>
            </table>
        </div>
    }
}

fn network_diagram(net: NetworkView, values: [f64; 4], best: usize) -> impl IntoView {
    let count = net.activation.len();
    let y_hidden = |i| 56 + i * 19;
    let y_output = |i| 117 + i * 101;
    let edges = net
        .output_weights
        .into_iter()
        .enumerate()
        .flat_map(|(a, weights)| {
            weights.into_iter().enumerate().map(move |(i, w)| {
                let color = if w >= 0.0 { "#6ce0c3" } else { "#fa8197" };
                let active = net.activation[i] > 0.0;
                view! {
                    <line x1="315" y1=y_hidden(i) x2="550" y2=y_output(a) stroke=color
                        opacity=if active {0.24 + w.abs().min(1.0)*0.55} else {0.06}
                        stroke-width=if a == best && active {2} else {1}>
                        <title>{format!("Нейрон {} → {}: {w:+.3}", i+1, ARROWS[a])}</title>
                    </line>
                }
            })
        })
        .collect_view();
    view! {
        <p class="small">{format!("Вход one-hot: {} значений. Скрытый слой: {} нейрона ReLU. Выход: 4 оценки Q.", WIDTH * HEIGHT, count)}</p>
        <svg class="rl-network" viewBox="0 0 780 525" role="img" aria-label="Схема обучаемой сети DQN">
            <text x="32" y="26" fill="#edf3fa" font-size="15">"Активный вход"</text>
            <text x="263" y="26" fill="#edf3fa" font-size="15">"Слой ReLU"</text>
            <text x="555" y="26" fill="#edf3fa" font-size="15">"Оценки ходов"</text>
            <rect x="32" y="242" width="115" height="38" rx="6" fill="#233c59"/>
            <text x="46" y="266" fill="#edf3fa" font-size="14">"Клетка = 1"</text>
            {(0..count).map(|i|view!{
                <line x1="147" y1="261" x2="300" y2=y_hidden(i)
                    stroke={if net.input_weights[i] >= 0.0 {"#8faaff"} else {"#fa8197"}}
                    opacity={if net.activation[i] > 0.0 {0.45} else {0.13}}>
                    <title>{format!("Вход → нейрон {}: {:+.3}", i+1, net.input_weights[i])}</title>
                </line>
            }).collect_view()}
            {edges}
            {net.activation.into_iter().enumerate().map(|(i,value)|view!{
                <circle cx="307" cy=y_hidden(i) r="7" fill={if value>0.0 {"#8faaff"} else {"#44536c"}}>
                    <title>{format!("Нейрон {}: активация {value:.3}",i+1)}</title>
                </circle>
                <text x="260" y=y_hidden(i)+4 fill="#b7c5da" font-size="11">{format!("{value:.2}")}</text>
            }).collect_view()}
            {values.into_iter().enumerate().map(|(a,value)|view!{
                <rect x="550" y=y_output(a)-18 width="165" height="36" rx="6"
                    fill=if a==best {"#24493e"} else {"#233c59"}/>
                <text x="563" y=y_output(a)+5 fill="#edf3fa" font-size="15">{format!("{} {value:+.3}",ARROWS[a])}</text>
            }).collect_view()}
        </svg>
        <div class="legend">
            <span>{format!("Буфер опыта: {} переходов", net.replay_size)}</span>
            <span>{format!("Обновлений сети: {}", net.updates)}</span>
            <span>"Выходные связи: зелёные +, розовые −"</span>
        </div>
        <details>
            <summary>"Веса и активации выбранного состояния"</summary>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>"Нейрон"</th><th>"Вес входа"</th><th>"ReLU"</th>
                        {ARROWS.into_iter().map(|a|view!{<th>{a}</th>}).collect_view()}
                    </tr></thead>
                    <tbody>
                        {(0..count).map(|i|view!{
                            <tr><th scope="row">{i+1}</th>
                                <td>{format!("{:+.3}",net.input_weights[i])}</td>
                                <td>{format!("{:.3}",net.activation[i])}</td>
                                {net.output_weights.iter().map(|w|view!{<td>{format!("{:+.3}",w[i])}</td>}).collect_view()}
                            </tr>
                        }).collect_view()}
                        <tr><th scope="row" colspan="3">"Смещение выхода"</th>
                            {net.output_bias.into_iter().map(|v|view!{<td>{format!("{v:+.3}")}</td>}).collect_view()}
                        </tr>
                    </tbody>
                </table>
            </div>
        </details>
    }
}
