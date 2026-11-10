mod clustering;
mod controls;
mod plots;
mod reinforcement;
use ml_lab_leptos::lab as state;
mod structure;

use std::time::Duration;

use controls::{DataControls, DisplayControls, ModelControls, Slider};
use leptos::prelude::*;
use ml_lab_leptos::{
    data::{Metrics, Point, grid, metrics, target},
    model::{Method, Parameters},
};
use state::{Lab, index};

#[derive(Clone, Copy, PartialEq)]
pub struct Display {
    target: bool,
    model: bool,
    test_points: bool,
    compare_test: bool,
    compare_target: bool,
    autoscale: bool,
    extrapolate: bool,
    parts: bool,
}

impl Default for Display {
    fn default() -> Self {
        Self {
            target: false,
            model: false,
            test_points: false,
            compare_test: false,
            compare_target: false,
            autoscale: true,
            extrapolate: false,
            parts: false,
        }
    }
}

pub fn num(x: f64) -> String {
    if !x.is_finite() {
        return "—".to_owned();
    }
    if x.abs() < 1e-9 {
        return "≈ 0".to_owned();
    }
    if x.abs() >= 1e4 || x.abs() < 1e-4 {
        return format!("{x:.4e}");
    }
    let decimals = (4 - x.abs().log10().floor() as i32).max(0) as usize;
    let value = format!("{x:.decimals$}");
    if decimals == 0 {
        value
    } else {
        value.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

#[component]
pub fn App() -> impl IntoView {
    let experiment = RwSignal::new(0);
    view! {
        <nav class="experiment-nav" aria-label="Эксперимент">
            {["Регрессия", "Кластеризация", "Обучение с подкреплением"].into_iter().enumerate().map(|(i, name)|view! {
                <button aria-pressed=move || (experiment.get()==i).to_string()
                    on:click=move |_|experiment.set(i)>{name}</button>
            }).collect_view()}
        </nav>
        <div hidden=move || experiment.get()!=0><Regression/></div>
        <div hidden=move || experiment.get()!=1><clustering::Clustering/></div>
        <div hidden=move || experiment.get()!=2><reinforcement::Reinforcement/></div>
    }
}

#[component]
fn Regression() -> impl IntoView {
    let lab = RwSignal::new(Lab::new());
    let selected = RwSignal::new(Method::Polynomial);
    let active = Memo::new(move |_| selected.get());
    let display = RwSignal::new(Display::default());
    let probe = RwSignal::new(0.5_f64);
    let stage = RwSignal::new(None::<usize>);
    let run_to = RwSignal::new(None::<u32>);
    let select = move |method| {
        run_to.set(None);
        selected.set(method);
        stage.set(None);
    };
    let mark = Callback::new(move |i| {
        if lab.with_untracked(|s| s.exclude) {
            run_to.set(None);
        }
        lab.update(|s| s.mark(i));
    });
    let timer = set_interval_with_handle(
        move || {
            let Some(goal) = run_to.get_untracked() else {
                return;
            };
            let mut more = false;
            lab.update(|s| {
                more = s.train_batch(s.batch_size(), goal);
            });
            if !more {
                run_to.set(None);
            }
        },
        Duration::from_millis(40),
    );
    if let Ok(handle) = timer {
        on_cleanup(move || handle.clear());
    } else {
        lab.update(|s| {
            s.warning = Some("Браузер не запустил таймер обучения. Доступны ручные шаги.")
        });
    }

    let diagnostic_key = Memo::new(move |_| {
        let method = active.get();
        lab.with(|s| {
            (
                method,
                s.parameters[index(method)],
                s.data_config.seed,
                s.training.clone(),
                s.data_config,
                if method == Method::Neural {
                    (s.history.len(), s.epoch())
                } else {
                    (0, 0)
                },
            )
        })
    });
    let diagnostic = Memo::new(move |_| {
        diagnostic_key.track();
        lab.with_untracked(|s| s.diagnostic(active.get_untracked()))
    });
    let coefficients = NodeRef::<leptos::html::Details>::new();
    let coefficients_open = RwSignal::new(false);
    let rows = Memo::new(move |_| {
        if !coefficients_open.get() || run_to.get().is_some() {
            return None;
        }
        let method = active.get();
        lab.with(|s| {
            s.models[index(method)]
                .as_ref()
                .ok()
                .map(structure::parameters)
        })
    });
    let outside_points: Vec<_> = grid(-9.0, -3.0, 200)
        .chain(grid(3.0, 9.0, 200))
        .map(|x| Point { x, y: target(x) })
        .collect();
    view! {
        <header><div class="eyebrow">"ML LAB / 01—07"</div><h1>"Одна целевая функция."<br/><span>"Семь способов её приблизить."</span></h1></header>
        <nav aria-label="Методы обучения">
            {Method::ALL.into_iter().map(|method|view!{
                <button aria-pressed=move||(active.get()==method).to_string() on:click=move|_|select(method)>{method.tab()}</button>
            }).collect_view()}
        </nav>
        <main>
            <aside>
                <DataControls lab=lab run_to=run_to/>
                <ModelControls lab=lab active=active run_to=run_to/>
                <DisplayControls display=display/>
            </aside>
            <div class="workspace">
                <section class="panel">
                    <div class="method-header"><div><div class="eyebrow">"Модель"</div><h2>{move||active.get().title()}</h2></div></div>
                    <p>{move||description(active.get())}</p>
                    <div class="presets">
                        {["Простая модель","Рабочий пример","Высокая гибкость"].into_iter().enumerate().map(|(i,label)|view!{
                            <button on:click=move|_|{
                                run_to.set(None);
                                stage.set(None);
                                let method=active.get_untracked();
                                lab.update(|s|{
                                    s.parameters[index(method)]=Parameters::preset(method,i,s.training.len(),s.periodic);
                                    if method==Method::Neural{s.epochs=[3000,6000,12000][i];}
                                    s.refit(method);
                                });
                                if method==Method::Neural {run_to.set(Some(lab.with_untracked(|s|s.epochs)));}
                            }>{label}</button>
                        }).collect_view()}
                    </div>
                    {move||{
                        let mode=display.get();
                        let method=active.get();
                        let x=probe.get();
                        lab.with(|s|plots::main_plot(s,s.models[index(method)].as_ref().ok(),mode,x,mark))
                    }}
                    <p class="warning" role="status" hidden=move||lab.with(|s|s.warning.is_none()&&s.models[index(active.get())].is_ok())>
                        {move||lab.with(|s|s.warning.or_else(||s.models[index(active.get())].as_ref().err().copied()))}
                    </p>
                    <div class="metric-list" aria-label="Ошибки модели">
                        {move||{
                            let mode=display.get();
                            let method=active.get();
                            lab.with(|s|s.scores[index(method)].map(|scores|view!{
                                {metric(format!("Обучение · {} точек",s.used_count()),scores[0])}
                                {mode.compare_test.then(||metric("Тест · 200 точек".to_owned(),scores[1]))}
                                {mode.compare_target.then(||metric("Целевая функция".to_owned(),scores[2]))}
                            }))
                        }}
                    </div>
                    {move||display.with(|d|d.compare_target).then(||lab.with(|s|{
                        s.models[index(active.get())].as_ref().ok().map(|fit|{
                            let value=metrics(&outside_points,|x|fit.predict(x)).rmse;
                            view!{<p class="note">{format!("RMSE вне обучения: {} · целевая функция на [−9, −3] и [3, 9].",num(value))}</p>}
                        })
                    }))}
                </section>
                <section class="panel">
                    <div class="eyebrow">"Внутри модели"</div><h2>"Как получается предсказание"</h2>
                    <Slider label="Проверочная точка x" min=-5.0 max=Signal::derive(||5.0) step=0.05
                        value=probe.into() change=Callback::new(move|x|{probe.set(x);stage.set(None);})/>
                    <p class="small">{move||lab.with(|s|s.models[index(active.get())].as_ref().ok().map(|fit|{
                        let x=probe.get();
                        let mut text=format!("Для x = {} модель предсказывает {}",num(x),num(fit.predict(x)));
                        if display.with(|d|d.compare_target){text.push_str(&format!(", целевое значение = {}",num(target(x))));}
                        if fit.periodic{text.push_str(&format!(". Фаза u = {}",num(fit.input(x))));}
                        text.push('.');
                        text
                    }))}</p>
                    <div class="neural-actions" hidden=move||active.get()!=Method::Neural>
                        <button on:click=move|_|{
                            let max=lab.with_untracked(|s|match s.parameters[index(Method::Neural)]{Parameters::Neural{depth,..}=>depth+1,_=>1});
                            stage.update(|s|*s=Some(s.map_or(0,|v|(v+1)%(max+1))));
                        }>"Следующий слой →"</button>
                        <button on:click=move|_|stage.set(None)>"Показать все слои"</button>
                    </div>
                    {move||{
                        let x=probe.get();let stage=stage.get();let method=active.get();
                        lab.with(|s|s.models[index(method)].as_ref().ok().map(|fit|structure::structure(fit,x,stage)))
                    }}
                </section>
                <section class="panel">
                    <h2>{move||if active.get()==Method::Neural{"Как меняется ошибка во время обучения"}else{"Как настройка влияет на ошибку"}}</h2>
                    {move||diagnostic.with(|(label,points)|plots::diagnostic(label,points,display.with(|d|d.compare_test)))}
                </section>
                <section class="panel">
                    <div class="eyebrow">"Как получается предсказание"</div><h2>"Формула и параметры"</h2>
                    <p class="formula">{move||structure::formula(active.get())}</p>
                    <details node_ref=coefficients on:toggle=move|_|{
                        coefficients_open.set(coefficients.get_untracked().is_some_and(|element|element.open()));
                    }><summary>"Показать получившиеся коэффициенты / правила"</summary>
                        <p class="small" hidden=move||run_to.get().is_none()>"Коэффициенты доступны после паузы или завершения обучения."</p>
                        {move||rows.with(|result|result.as_ref().map(|rows|view!{
                            <div class="table-wrap"><table><thead><tr><th>"Параметр / правило"</th><th>"Значение"</th></tr></thead>
                                <tbody>{rows.iter().map(|(k,v)|view!{<tr><td>{k.clone()}</td><td>{v.clone()}</td></tr>}).collect_view()}</tbody>
                            </table></div>
                        }))}
                    </details>
                </section>
                <section class="panel">
                    <h2>"Сравнение на общих данных"</h2>
                    <div class="table-wrap"><table>
                        <thead><tr><th>"Метод"</th><th>"RMSE обучение"</th>
                            <th hidden=move||!display.with(|d|d.compare_test)>"RMSE тест"</th>
                            <th hidden=move||!display.with(|d|d.compare_target)>"RMSE целевая функция"</th>
                            <th>"Состояние"</th>
                        </tr></thead>
                        <tbody>{Method::ALL.into_iter().map(|method|view!{
                            <tr class:selected=move||active.get()==method>
                                <td><button on:click=move|_|select(method)>{method.tab()}</button></td>
                                <td>{move||score(lab,method,0)}</td>
                                <td hidden=move||!display.with(|d|d.compare_test)>{move||score(lab,method,1)}</td>
                                <td hidden=move||!display.with(|d|d.compare_target)>{move||score(lab,method,2)}</td>
                                <td>{move||lab.with(|s|if method==Method::Neural {format!("{} эпох",s.epoch())}else if s.models[index(method)].is_ok(){"рассчитано".to_owned()}else{"ошибка".to_owned()})}</td>
                            </tr>
                        }).collect_view()}</tbody>
                    </table></div>
                </section>
            </div>
        </main>
        <footer>"RMSE измеряет среднюю величину ошибки. Тестовые точки используются только для оценки."</footer>
    }
}

fn metric(title: String, metric: Metrics) -> impl IntoView {
    view! {<div class="metric">{title}" · RMSE"<strong>{num(metric.rmse)}</strong>{format!("R² = {}",metric.r_squared.map_or_else(||"—".to_owned(),num))}</div>}
}

fn score(lab: RwSignal<Lab>, method: Method, column: usize) -> String {
    lab.with(|s| {
        s.scores[index(method)].map_or_else(|| "—".to_owned(), |scores| num(scores[column].rmse))
    })
}

fn description(method: Method) -> &'static str {
    match method {
        Method::Polynomial => {
            "Подбираем коэффициенты полинома по обучающим наблюдениям в базисе Чебышёва."
        }
        Method::Knn => "Предсказание равно среднему значению k ближайших наблюдений.",
        Method::Tree => {
            "Дерево делит ось x по порогам и возвращает среднее наблюдений выбранного листа."
        }
        Method::Forest => {
            "Лес усредняет предсказания деревьев, обученных на случайных выборках с возвращением."
        }
        Method::Rbf => "Модель суммирует локальные гауссовы функции с обученными весами.",
        Method::Neural => {
            "Сеть последовательно преобразует вход. Adam обновляет веса по градиенту ошибки обучения."
        }
        Method::Gp => {
            "Гауссовское ядро связывает соседние значения. Полоса показывает условный интервал скрытой функции."
        }
    }
}
