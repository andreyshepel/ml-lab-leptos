use std::time::Duration;

use leptos::{
    leptos_dom::helpers::set_interval_with_handle, prelude::*, wasm_bindgen::JsCast,
    web_sys::HtmlElement,
};
use ml_lab_leptos::reinforcement::{
    ARROWS, Collected, EPISODE_LIMIT, Episode, Experiment, GOAL_COUNT, HEIGHT, Method, NetworkView,
    START, Tile, WIDTH, tile,
};

const SEED: u32 = 72;
const CELL: usize = 64;
const BATCH_STEPS: usize = 96;

fn fresh() -> [Experiment; 3] {
    Method::ALL.map(|method| Experiment::new(method, SEED))
}

#[component]
pub fn Reinforcement(active: Signal<bool>) -> impl IntoView {
    let cartpole = RwSignal::new(false);
    view! {
        <nav class="experiment-nav" aria-label="Задача обучения с подкреплением">
            <button aria-pressed=move || (!cartpole.get()).to_string()
                on:click=move |_| cartpole.set(false)>"Лабиринт"</button>
            <button aria-pressed=move || cartpole.get().to_string()
                on:click=move |_| cartpole.set(true)>"CartPole"</button>
        </nav>
        <div hidden=move || cartpole.get()><Maze active=Signal::derive(move || active.get() && !cartpole.get())/></div>
        <div hidden=move || !cartpole.get()>
            <super::cartpole::CartPole active=Signal::derive(move || active.get() && cartpole.get())/>
        </div>
    }
}

#[component]
fn Maze(active: Signal<bool>) -> impl IntoView {
    let experiments = RwSignal::new(fresh());
    let count = |kind| (0..WIDTH * HEIGHT).filter(|&i| tile(i) == kind).count();
    let goals = count(Tile::Goal);
    let traps = count(Tile::Trap);
    let selected = RwSignal::new(0_usize);
    let epsilon = RwSignal::new(0.45_f64);
    let step_cost = RwSignal::new(0.01_f64);
    let preview = RwSignal::new(false);
    let inspected = RwSignal::new(None::<usize>);
    let table_scroll = StoredValue::new(0);
    let run_to = RwSignal::new(None::<usize>);
    let timer = set_interval_with_handle(
        move || {
            if !active.get_untracked() {
                return;
            }
            let Some(goal) = run_to.get_untracked() else {
                return;
            };
            let index = selected.get_untracked();
            let (epsilon, cost) = (epsilon.get_untracked(), step_cost.get_untracked());
            let mut more = false;
            experiments.update(|models| {
                let budget = if models[index].method == Method::Dqn {
                    16
                } else {
                    BATCH_STEPS
                };
                more = models[index].train_batch(goal, budget, epsilon, cost);
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
    Effect::new(move |_| {
        if !active.get() {
            run_to.set(None);
        }
    });
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
            <p class="small">"Собери все +1 за один эпизод. Повторный заход не приносит награду. Ловушки дают −1, и маршрут продолжается."</p>
            <div class="rl-layout">
                <div>
                    {move || experiments.with(|models| board(&models[selected.get()], preview.get(), inspected))}
                    <div class="legend">
                        <span>"S Старт"</span><span class="truth">{format!("Награды +1 · {goals}")}</span>
                        <span style="color:#fa8197">{format!("Ловушки −1 · {traps}")}</span><span class="model">"● Агент и маршрут"</span>
                        <span>{format!("{} × {} клеток · до 2 000 ходов", WIDTH, HEIGHT)}</span>
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
                            <option value="0.01">"−0,01"</option>
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
                        let collected = if preview.get() { Collected::default() } else { model.collected };
                        (preview.get() || !model.finished()).then(||view!{
                            <div class="legend">
                                <span>{if preview.get(){"Оценки Q на старте"}else{"Оценки Q в текущей клетке"}}</span>
                                {model.values(cell, collected).into_iter().zip(ARROWS).map(|(v,arrow)|view!{
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
                    let start = preview.get() || model.finished();
                    let collected = if start { Collected::default() } else { model.collected };
                    let chosen = inspected.get().filter(|&cell| tile(cell) != Tile::Goal || collected.contains(cell));
                    let cell = chosen.unwrap_or(if start { START } else { model.position });
                    let label = if chosen.is_some() {
                        "Выбранная клетка"
                    } else if cell == START {
                        "Стартовая клетка"
                    } else {
                        "Текущая клетка"
                    };
                    view! {
                        <p class="small">{format!("{label}: {} · собрано {}/{} · оценки для текущего набора собранных наград", cell_name(cell), collected.count(), GOAL_COUNT)}</p>
                        {if let Some(network) = model.network_view(cell, collected) {
                            network_diagram(network, model.values(cell, collected), model.best_action(cell, collected)).into_any()
                        } else {
                            q_table(model, cell, collected, inspected, table_scroll, active).into_any()
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
        return format!(
            "Маршрут: собрано {}/{} за {} ходов. {}",
            route.collected.count(),
            GOAL_COUNT,
            route.path.len() - 1,
            if route.collected.count() == GOAL_COUNT {
                "Все награды собраны."
            } else {
                "Жадный маршрут зациклился."
            }
        );
    }
    let result = if model.collected.count() == GOAL_COUNT {
        " Все награды собраны."
    } else if model.steps >= EPISODE_LIMIT {
        " Достигнут лимит ходов."
    } else {
        ""
    };
    let action = model
        .last_action
        .zip(model.last_reward)
        .map_or_else(String::new, |(a, r)| {
            format!(" Последний ход {} ({r:+.2}).", ARROWS[a])
        });
    format!(
        "Эпизодов завершено: {}. Ходов: {}. Собрано {}/{}. Награда: {:+.2}.{action}{result}",
        model.completed,
        model.steps,
        model.collected.count(),
        GOAL_COUNT,
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
    let route = preview.then(|| model.greedy_path());
    let (path, collected) = route
        .as_ref()
        .map_or((model.path.as_slice(), model.collected), |route| {
            (route.path.as_slice(), route.collected)
        });
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
                let cell = tile(i);
                let (fill,label)=match cell {
                    Tile::Open if i == START=>("#233c59","S"),
                    Tile::Open=>("#0b1321",""),
                    Tile::Goal if collected.contains(i)=>("#0b1321",""),
                    Tile::Goal=>("#24493e","+1"),
                    Tile::Trap=>("#572d40","−1"),
                    Tile::Wall=>("#536278",""),
                };
                let selectable = match cell {
                    Tile::Open | Tile::Trap => true,
                    Tile::Goal => !preview && collected.contains(i),
                    Tile::Wall => false,
                };
                let active = selected == Some(i);
                view!{
                    <rect x=x+2 y=y+2 width=CELL-4 height=CELL-4 rx="6" fill=fill
                        stroke=if active {"#edf3fa"} else {"#3a4c67"} stroke-width=if active {3} else {1}
                        class:rl-cell=selectable role=if selectable {Some("button")} else {None}
                        tabindex=if selectable {Some(0)} else {None}
                        aria-label=match cell {
                            Tile::Goal if !collected.contains(i) => format!("Награда +1, {}", cell_name(i)),
                            Tile::Trap => format!("Ловушка −1, {}", cell_name(i)),
                            _ => cell_name(i),
                        }
                        on:click=move |_| if selectable { inspected.set(Some(i)); }
                        on:keydown=move |ev| if selectable && matches!(ev.key().as_str(),"Enter"|" ") {
                            ev.prevent_default(); inspected.set(Some(i));
                        }/>
                    <text pointer-events="none" x=x+CELL/2 y=y+23 text-anchor="middle" fill="#edf3fa" font-size="23">{label}</text>
                }
            }).collect_view()}
            <polyline pointer-events="none" points=points fill="none" stroke="#ffbc7b" stroke-width="4"
                stroke-linecap="round" stroke-linejoin="round"/>
            <circle pointer-events="none" cx=x+CELL/2 cy=y+CELL/2 r="13" fill="#ffbc7b" stroke="#0b1321" stroke-width="2"/>
        </svg>
    }
}

fn history(episodes: &[Episode]) -> impl IntoView + use<> {
    if episodes.is_empty() {
        return view! { <p class="small">"График появится после первого завершённого эпизода."</p> }.into_any();
    }
    let low = episodes.iter().map(|e| e.reward).fold(-1.0, f64::min);
    let high = GOAL_COUNT as f64;
    let position = |i: usize, reward: f64| {
        (
            46.0 + 720.0 * i as f64 / (episodes.len() - 1).max(1) as f64,
            18.0 + 100.0 * (high - reward) / (high - low),
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
    view! {
        <svg viewBox="0 0 800 140" role="img" aria-label="Награда за последние эпизоды">
            <line x1="46" x2="766" y1=zero y2=zero stroke="#64748b" stroke-dasharray="4 4"/>
            <text x="36" y=zero+4.0 text-anchor="end" font-size="12" fill="#aabbd1">"0"</text>
            <polyline points=points stroke="#8faaff" stroke-width="2" fill="none"/>
            {episodes.iter().enumerate().map(|(i,e)|{
                let (x,y)=position(i,e.reward);
                let color = if e.collected == GOAL_COUNT { "#6ce0c3" } else { "#fa8197" };
                view!{<circle cx=x cy=y r="3" fill=color><title>{format!("Награда {:+.2}, {} ходов, собрано {}/{}",e.reward,e.steps,e.collected,GOAL_COUNT)}</title></circle>}
            }).collect_view()}
        </svg>
        <p class="small">{format!("Последние {} завершённых эпизодов.", episodes.len())}</p>
    }.into_any()
}

fn cell_name(cell: usize) -> String {
    format!("строка {}, столбец {}", cell / WIDTH + 1, cell % WIDTH + 1)
}

fn q_table(
    model: &Experiment,
    cell: usize,
    collected: Collected,
    inspected: RwSignal<Option<usize>>,
    scroll: StoredValue<i32>,
    active: Signal<bool>,
) -> impl IntoView + use<> {
    let container = NodeRef::<leptos::html::Div>::new();
    let header = NodeRef::<leptos::html::Thead>::new();
    Effect::new(move |_| {
        if !active.get() {
            return;
        }
        let (Some(container), Some(header)) = (container.get(), header.get()) else {
            return;
        };
        let top = if inspected.get().is_none() {
            let Some(row) = container
                .query_selector("tr.selected")
                .ok()
                .flatten()
                .and_then(|row| row.dyn_into::<HtmlElement>().ok())
            else {
                return;
            };
            let visible_height = container.client_height() - header.offset_height();
            row.offset_top() - header.offset_height() - (visible_height - row.offset_height()) / 2
        } else {
            scroll.get_value()
        };
        container.set_scroll_top(top);
        scroll.set_value(container.scroll_top());
    });
    view! {
        <div class="table-wrap rl-q-table" node_ref=container
            on:scroll=move |_| {
                if let Some(container) = container.get_untracked() {
                    scroll.set_value(container.scroll_top());
                }
            }>
            <table aria-label="Таблица оценок Q">
                <thead node_ref=header><tr><th scope="col">"Состояние"</th>
                    {ARROWS.into_iter().map(|arrow|view!{<th scope="col">{arrow}</th>}).collect_view()}
                    <th scope="col">"Лучший ход"</th>
                </tr></thead>
                <tbody>
                    {(0..WIDTH * HEIGHT).filter(|&i| tile(i) != Tile::Wall).map(|i| {
                        let pending = tile(i) == Tile::Goal && !collected.contains(i);
                        let best = model.best_action(i, collected);
                        let scores = model.values(i, collected);
                        let tied = scores.iter().filter(|&&value| value == scores[best]).count() > 1;
                        view! {
                            <tr class:selected=i==cell>
                                <th scope="row">
                                    {if pending {
                                        cell_name(i).into_any()
                                    } else {
                                        view! { <button on:click=move |_|inspected.set(Some(i))>{cell_name(i)}</button> }.into_any()
                                    }}
                                    {match tile(i) { Tile::Open if i == START=>" · старт", Tile::Goal=>" · награда", Tile::Trap=>" · ловушка", _=>"" }}
                                </th>
                                {scores.into_iter().enumerate().map(|(a,value)|view!{
                                    <td class:truth=!pending&&!tied&&a==best>
                                        {if pending {"—".to_owned()} else {format!("{value:+.3}")}}
                                    </td>
                                }).collect_view()}
                                <td>{if pending {"Подбор меняет состояние"} else if tied {"Ничья"} else {ARROWS[best]}}</td>
                            </tr>
                        }
                    }).collect_view()}
                </tbody>
            </table>
        </div>
    }
}

fn network_diagram(net: NetworkView, values: [f64; 4], best: usize) -> impl IntoView {
    let count = net.first.len();
    view! {
        <p class="small">{format!("Вход: {} значений one-hot для клетки и {} флагов наград. Два скрытых слоя ReLU по {} нейрона. Выход: четыре линейные оценки Q.", WIDTH * HEIGHT, GOAL_COUNT, count)}</p>
        <svg class="rl-network" viewBox="0 0 780 125" role="img" aria-label="Схема двухслойной сети DQN">
            <path d="M 180 61 H 214 M 360 61 H 394 M 540 61 H 574" stroke="#8faaff" stroke-width="2"/>
            <g fill="#233c59">
                <rect x="10" y="15" width="170" height="92" rx="6"/>
                <rect x="214" y="15" width="146" height="92" rx="6"/>
                <rect x="394" y="15" width="146" height="92" rx="6"/>
            </g>
            <g fill="#edf3fa" font-size="15" text-anchor="middle">
                <text x="95" y="57">"Клетка и награды"</text>
                <text x="95" y="79">{WIDTH * HEIGHT + GOAL_COUNT}</text>
                <text x="287" y="57">"Слой 1 · ReLU"</text>
                <text x="287" y="79">{count}</text>
                <text x="467" y="57">"Слой 2 · ReLU"</text>
                <text x="467" y="79">{count}</text>
            </g>
            {values.into_iter().enumerate().map(|(a,value)|view!{
                <rect x="574" y=8+a*28 width="180" height="25" rx="5"
                    fill=if a==best {"#24493e"} else {"#233c59"}/>
                <text x="592" y=26+a*28 fill="#edf3fa" font-size="15">{format!("{} {value:+.3}",ARROWS[a])}</text>
            }).collect_view()}
        </svg>
        <p class="small">"Схема показывает слои целиком. Отдельные связи между слоями скрыты, но все участвуют в расчёте."</p>
        <div class="legend">
            <span>{format!("Буфер опыта: {} переходов", net.replay_size)}</span>
            <span>{format!("Обновлений сети: {}", net.updates)}</span>
        </div>
        <details>
            <summary>"Активации обоих слоёв и веса выхода"</summary>
            <div class="table-wrap">
                <table>
                    <thead><tr><th>"Номер нейрона"</th><th>"ReLU 1"</th><th>"ReLU 2"</th>
                        {ARROWS.into_iter().map(|a|view!{<th>{format!("Вес 2 → {a}")}</th>}).collect_view()}
                    </tr></thead>
                    <tbody>
                        {(0..count).map(|i|view!{
                            <tr><th scope="row">{i+1}</th>
                                <td>{format!("{:.3}",net.first[i])}</td>
                                <td>{format!("{:.3}",net.second[i])}</td>
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
