use std::{cell::Cell, time::Duration};

use leptos::{leptos_dom::helpers::set_interval_with_handle, prelude::*};
use ml_lab_leptos::reinforcement::cartpole::{
    Action, DECISION_MS, Frame, NetworkConfig, Reward, TRACK_EDGE, Training,
};

use super::controls::{Slider, Toggle};

const DEMO_SEED: u32 = 9173;
const TIMER_MS: u64 = 10;
const FAST_MULTIPLIER: usize = 1000;
const FAST_BATCH: usize = FAST_MULTIPLIER * TIMER_MS as usize / DECISION_MS as usize;

#[component]
pub(super) fn CartPole(active: Signal<bool>) -> impl IntoView {
    let reward = RwSignal::new(Reward::AboveOnly);
    let run = RwSignal::new(Training::new(DEMO_SEED, true, Reward::AboveOnly));
    let running = RwSignal::new(false);
    let fast = RwSignal::new(false);
    let learning = RwSignal::new(true);
    let failed = RwSignal::new(false);
    let applied = RwSignal::new(NetworkConfig::default());
    let draft = RwSignal::new(NetworkConfig::default());
    let config_error = RwSignal::new(None::<&'static str>);
    let frame = Signal::derive(move || run.with(Training::frame));

    let advance = move |budget| {
        let mut ok = true;
        run.update(|run| ok = run.advance(budget, learning.get_untracked()));
        if !ok {
            running.set(false);
            failed.set(true);
        }
    };
    let elapsed_ms = Cell::new(0);
    let timer = set_interval_with_handle(
        move || {
            if !active.get_untracked() || !running.get_untracked() {
                elapsed_ms.set(0);
                return;
            }
            if fast.get_untracked() {
                elapsed_ms.set(0);
                advance(FAST_BATCH);
                return;
            }
            let elapsed = elapsed_ms.get() + TIMER_MS;
            if elapsed >= DECISION_MS {
                elapsed_ms.set(0);
                advance(1);
            } else {
                elapsed_ms.set(elapsed);
            }
        },
        Duration::from_millis(TIMER_MS),
    );
    let timer_available = timer.is_ok();
    if let Ok(handle) = timer {
        on_cleanup(move || handle.clear());
    }
    Effect::new(move |_| {
        if !active.get() {
            running.set(false);
        }
    });
    let rebuild =
        move |config| match Training::with_network(DEMO_SEED, true, reward.get_untracked(), config)
        {
            Ok(new_run) => {
                running.set(false);
                failed.set(false);
                config_error.set(None);
                applied.set(config);
                run.set(new_run);
            }
            Err(error) => config_error.set(Some(error)),
        };
    let reset = move || rebuild(applied.get_untracked());

    view! {
        <section class="panel rl-panel">
            <div class="eyebrow">"Обучение с подкреплением · DQN"</div>
            <h2>"CartPole · поднять и удержать шест"</h2>
            <p>"Шест начинает внизу. Агент пробует раскачать его толчками тележки, поднять и удержать. "
                "На краях тележка упирается в ограничители, а шест продолжает вращаться."</p>
            <div class="neural-actions">
                <button disabled=move || !timer_available || failed.get() || running.get()
                    on:click=move |_| running.set(true)>"Запустить"</button>
                <button disabled=move || !running.get() on:click=move |_| running.set(false)>"Пауза"</button>
                <button disabled=move || failed.get() || running.get()
                    on:click=move |_| advance(1)>"Один шаг"</button>
                <button disabled=move || failed.get() on:click=move |_| run.update(Training::drop_pole)>
                    "Опустить шест"
                </button>
                <button on:click=move |_| reset()>"Начать с нуля"</button>
            </div>
            <div class="rl-layout">
                <Cart frame=frame/>
                <div>
                    <label class="cluster-count">"Награда за высоту шеста"
                        <select prop:value=move || if reward.get() == Reward::Signed { "signed" } else { "above" }
                            on:change=move |ev| {
                                let mode = if event_target_value(&ev) == "signed" { Reward::Signed } else { Reward::AboveOnly };
                                if mode != reward.get_untracked() {
                                    reward.set(mode);
                                    reset();
                                }
                            }>
                            <option value="above">"0 внизу, до +1 наверху"</option>
                            <option value="signed">"От −1 внизу до +1 наверху"</option>
                        </select>
                    </label>
                    <p class="small">"В горизонтальном положении награда равна нулю. Чем ближе шест к вертикали сверху, "
                        "тем она больше. Смена награды сбрасывает веса и буфер опыта."</p>
                    <label class="check">
                        <input type="checkbox" prop:checked=move || learning.get()
                            on:change=move |ev| learning.set(event_target_checked(&ev))/>
                        "Обучать во время движения"
                    </label>
                    <label class="check">
                        <input type="checkbox" prop:checked=move || fast.get()
                            on:change=move |ev| fast.set(event_target_checked(&ev))/>
                        {format!("Ускорить ×{FAST_MULTIPLIER}")}
                    </label>
                    <p class="small">"При выключенном обучении агент выбирает действие с большей Q-оценкой. "
                        "«Опустить шест» позволяет проверить восстановление с текущими весами."</p>
                    <p class="small rl-status">{move || {
                        let f = frame.get();
                        let mean = f.mean_reward.map_or_else(|| "—".to_owned(), |x| format!("{x:+.2}"));
                        let above = f.above_fraction.map_or_else(|| "—".to_owned(), |x| format!("{:.0}%", x * 100.0));
                        let epsilon = if learning.get() { run.with(Training::epsilon) } else { 0.0 };
                        format!("Последние 10 с симуляции · средняя награда {mean} · выше горизонтали {above} · случайный ход {:.0}%", epsilon * 100.0)
                    }}</p>
                    <p role="status" class="small" hidden=move || !failed.get()>
                        "Симуляция остановлена из-за численного сбоя. Можно начать с нуля."
                    </p>
                    <p class="small" hidden=timer_available>"Таймер недоступен. Можно выполнять шаги вручную."</p>
                </div>
            </div>
            <details class="rl-inspector">
                <summary>"Настройки нейросети DQN"</summary>
                <p class="small">{move || format!("Текущая сеть: {} · Adam, шаг {} · L2 {}",
                    architecture(applied.get()), applied.get().rate, applied.get().l2)}</p>
                <Slider label="Скрытые слои" min=1.0 max=Signal::derive(|| NetworkConfig::MAX_DEPTH as f64)
                    step=1.0 value=Signal::derive(move || draft.get().depth as f64)
                    change=Callback::new(move |value| draft.update(|config| config.depth = value as usize))/>
                <Slider label="Нейроны в скрытом слое" min=NetworkConfig::MIN_WIDTH as f64
                    max=Signal::derive(|| NetworkConfig::MAX_WIDTH as f64)
                    step=1.0 value=Signal::derive(move || draft.get().width as f64)
                    change=Callback::new(move |value| draft.update(|config| config.width = value as usize))/>
                <Toggle label="Активация ReLU (иначе tanh)"
                    checked=Signal::derive(move || draft.get().relu)
                    change=Callback::new(move |value| draft.update(|config| config.relu = value))/>
                <label>"Скорость обучения"
                    <select prop:value=move || draft.get().rate.to_string()
                        on:change=move |ev| {
                            if let Ok(rate) = event_target_value(&ev).parse() {
                                draft.update(|config| config.rate = rate);
                            }
                        }>
                        {NetworkConfig::RATES.into_iter().map(|value| view! {
                            <option value=value.to_string()>{value.to_string()}</option>
                        }).collect_view()}
                    </select>
                </label>
                <label>"Регуляризация L2"
                    <select prop:value=move || draft.get().l2.to_string()
                        on:change=move |ev| {
                            if let Ok(l2) = event_target_value(&ev).parse() {
                                draft.update(|config| config.l2 = l2);
                            }
                        }>
                        {NetworkConfig::L2_OPTIONS.into_iter().map(|value| view! {
                            <option value=value.to_string()>{value.to_string()}</option>
                        }).collect_view()}
                    </select>
                </label>
                <p class="small">"Применение настроек ставит симуляцию на паузу и сбрасывает веса, опыт и счётчик шагов."</p>
                <div class="neural-actions">
                    <button disabled=move || draft.get() == applied.get()
                        on:click=move |_| rebuild(draft.get_untracked())>"Применить и начать с нуля"</button>
                    <button disabled=move || draft.get() == NetworkConfig::default()
                        on:click=move |_| draft.set(NetworkConfig::default())>"Подставить настройки по умолчанию"</button>
                </div>
                <p role="status" class="small" hidden=move || config_error.get().is_none()>
                    {move || config_error.get().unwrap_or_default()}
                </p>
            </details>
            <details>
                <summary>"Как DQN связан с Q-таблицей"</summary>
                <p class="small">"В лабиринте каждой клетке соответствует строка таблицы с оценками ходов. "
                    "В CartPole положение, скорости и угол непрерывны. Сеть вычисляет аналог строки Q-таблицы "
                    "для поданного состояния: Q(состояние, влево) и Q(состояние, вправо)."</p>
                <p class="small">"На входе положение и скорость тележки, синус и косинус угла, угловая скорость шеста. "
                    "На выходе две оценки будущей награды с дисконтом. "
                    "Под тележкой показаны оценки перед последним толчком и фактически выбранное действие. "
                    "Обучение меняет общие веса сети, поэтому один переход может изменить оценки и для других состояний."</p>
            </details>
        </section>
    }
}

fn architecture(config: NetworkConfig) -> String {
    let hidden = format!(
        "{} {}",
        config.width,
        if config.relu { "ReLU" } else { "tanh" }
    );
    std::iter::once("5".to_owned())
        .chain(std::iter::repeat_n(hidden, config.depth))
        .chain(std::iter::once("2".to_owned()))
        .collect::<Vec<_>>()
        .join(" → ")
}

#[component]
fn Cart(frame: Signal<Frame>) -> impl IntoView {
    view! {
        <article class="cartpole-card">
            <svg viewBox="0 0 400 220" role="img" aria-label="Шест на тележке">
                <path d="M 44 146 H 356 M 44 137 V 154 M 356 137 V 154" stroke="#506482" stroke-width="2" fill="none"/>
                <path d="M 44 126 H 356" stroke="#2b3b52" stroke-dasharray="4 6"/>
                <g transform=move || format!("translate({:.2} 0)", 200.0 + 156.0 * frame.get().state.position / TRACK_EDGE)>
                    <rect x="-20" y="124" width="40" height="15" rx="4" fill="var(--blue)"/>
                    <circle cx="-12" cy="142" r="4" fill="var(--text)"/>
                    <circle cx="12" cy="142" r="4" fill="var(--text)"/>
                    <line x1="0" y1="126" x2="0" y2="61" stroke="var(--orange)" stroke-width="6"
                        stroke-linecap="round"
                        transform=move || format!("rotate({:.3} 0 126)", frame.get().state.angle.to_degrees())/>
                    <circle cx="0" cy="126" r="4" fill="var(--text)"/>
                </g>
            </svg>
            <p class="cartpole-score">{move || {
                let f = frame.get();
                if f.steps == 0 {
                    "Шест внизу · первый шаг ещё впереди".to_owned()
                } else if f.decision.is_none() {
                    format!("Шаг {} · шест опущен вручную", f.steps)
                } else {
                    format!("Шаг {} · награда {:+.2}", f.steps, f.reward)
                }
            }}</p>
            <dl class="cartpole-observation">
                <div><dt>"Положение"</dt><dd>{move || format!("{:+.2} м", frame.get().state.position)}</dd></div>
                <div><dt>"Скорость"</dt><dd>{move || format!("{:+.2} м/с", frame.get().state.velocity)}</dd></div>
                <div><dt>"Угол"</dt><dd>{move || format!("{:+.2}°", frame.get().state.angle.to_degrees())}</dd></div>
                <div><dt>"Угловая скорость"</dt><dd>{move || format!("{:+.1} °/с", frame.get().state.angular_velocity.to_degrees())}</dd></div>
            </dl>
            {move || frame.get().decision.map(|decision| view! {
                <div class="cartpole-actions" aria-label="Оценки Q и действие на последнем шаге">
                    <span class:chosen=decision.action == Action::Left>{format!("← Q {:.2}", decision.values[0])}</span>
                    <span class:chosen=decision.action == Action::Right>{format!("Q {:.2} →", decision.values[1])}</span>
                    <span>{if decision.exploratory { "Случайный толчок" } else { "Выбор по Q" }}</span>
                </div>
            })}
        </article>
    }
}
