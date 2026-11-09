use super::{
    Display, num,
    state::{Lab, index},
};
use leptos::prelude::*;
use ml_lab_leptos::model::{Method, Parameters};

#[component]
pub fn Toggle(label: &'static str, checked: Signal<bool>, change: Callback<bool>) -> impl IntoView {
    view! {
        <label class="check"><input type="checkbox" prop:checked=move||checked.get()
            on:change=move|ev|change.run(event_target_checked(&ev))/> {label}</label>
    }
}

#[component]
pub fn Slider(
    label: &'static str,
    min: f64,
    max: Signal<f64>,
    step: f64,
    value: Signal<f64>,
    change: Callback<f64>,
) -> impl IntoView {
    view! {
        <label>{label}<output>{move||num(value.get())}</output>
            <input type="range" min=min max=move||max.get() step=step prop:value=move||value.get()
                on:input=move|ev|{if let Ok(v)=event_target_value(&ev).parse::<f64>(){change.run(v);}}/>
        </label>
    }
}

#[component]
pub fn DataControls(lab: RwSignal<Lab>, run_to: RwSignal<Option<u32>>) -> impl IntoView {
    let regenerate = move |update: fn(&mut ml_lab_leptos::data::DataConfig, f64), value: f64| {
        run_to.set(None);
        lab.update(|state| {
            update(&mut state.data_config, value);
            state.regenerate();
        });
    };
    view! {
        <details class="controls" open><summary class="eyebrow">"Общие данные"</summary>
            <Toggle label="Учитывать известный период T = 2"
                checked=Signal::derive(move||lab.with(|s|s.periodic))
                change=Callback::new(move|checked|{run_to.set(None);lab.update(|s|{s.periodic=checked;s.refit_all();});})/>
            <p class="small">{move||lab.with(|s|if s.periodic{
                format!("Обучение по фазе: {} уникальных фаз из {} наблюдений.",s.training.len(),s.used_count())
            }else{"Обычный режим. Модель получает исходную координату x.".to_owned()})}</p>
            <Slider label="Обучающие точки" min=8.0 max=Signal::derive(||200.0) step=1.0
                value=Signal::derive(move||lab.with(|s|s.data_config.count as f64))
                change=Callback::new(move|v|regenerate(|s,v|{s.count=v as usize;s.outlier_count=s.outlier_count.min(s.count/3);},v))/>
            <Slider label="Шум ±" min=0.0 max=Signal::derive(||0.5) step=0.01
                value=Signal::derive(move||lab.with(|s|s.data_config.noise))
                change=Callback::new(move|v|regenerate(|s,v|s.noise=v,v))/>
            <Toggle label="Добавить выбросы" checked=Signal::derive(move||lab.with(|s|s.data_config.outliers))
                change=Callback::new(move|v|regenerate(|s,v|s.outliers=v!=0.0,if v{1.0}else{0.0}))/>
            <div hidden=move||lab.with(|s|!s.data_config.outliers)>
                <Slider label="Количество выбросов" min=1.0 max=Signal::derive(move||lab.with(|s|(s.data_config.count/3).min(6) as f64)) step=1.0
                    value=Signal::derive(move||lab.with(|s|s.data_config.outlier_count as f64))
                    change=Callback::new(move|v|regenerate(|s,v|s.outlier_count=v as usize,v))/>
                <Slider label="Сила выбросов" min=2.0 max=Signal::derive(||5.0) step=0.5
                    value=Signal::derive(move||lab.with(|s|s.data_config.strength))
                    change=Callback::new(move|v|regenerate(|s,v|s.strength=v,v))/>
            </div>
            <Toggle label="Исключить выбросы из обучения" checked=Signal::derive(move||lab.with(|s|s.exclude))
                change=Callback::new(move|v|{run_to.set(None);lab.update(|s|{s.exclude=v;s.refit_all();});})/>
            <p class="small">"Клик по обучающей точке ставит или снимает метку выброса."</p>
            <p class="small" role="status">{move||lab.with(|s|format!("Помечено: {}. В обучении: {} из {}.",s.marked.len(),s.used_count(),s.data.train.len()))}</p>
            <label for="data-seed">"Seed набора данных"</label>
            <div class="inline">
                <input id="data-seed" type="number" min="0" max="4294967295" step="1"
                    prop:value=move||lab.with(|s|s.data_config.seed)
                    on:change=move|ev|{if let Ok(seed)=event_target_value(&ev).parse::<u32>(){regenerate(|s,v|s.seed=v as u32,f64::from(seed));}}/>
                <button on:click=move|_|{run_to.set(None);lab.update(|s|{s.data_config.seed=s.data_config.seed.wrapping_add(1);s.regenerate();});}>"Новые точки"</button>
            </div>
            <p class="small">"Обучающие точки получены из целевой функции на [−3, 3] с добавлением шума."</p>
        </details>
    }
}

#[component]
pub fn DisplayControls(display: RwSignal<Display>) -> impl IntoView {
    macro_rules! toggle {
        ($label:literal,$field:ident) => {
            view! {
                <Toggle label=$label checked=Signal::derive(move||display.with(|d|d.$field))
                    change=Callback::new(move|v|display.update(|d|d.$field=v))/>
            }
        };
    }
    view! {
        <details class="controls" open><summary class="eyebrow">"Отображение"</summary>
            {toggle!("Показывать целевую функцию",target)}
            {toggle!("Показывать обученную модель",model)}
            {toggle!("Тестовые точки (200)",test_points)}
            {toggle!("Сравнение с тестом",compare_test)}
            {toggle!("Сравнение с целевой функцией",compare_target)}
            {toggle!("Автомасштаб Y",autoscale)}
            {toggle!("За пределами обучения",extrapolate)}
            {toggle!("Внутреннее устройство",parts)}
        </details>
    }
}

#[component]
pub fn ModelControls(
    lab: RwSignal<Lab>,
    active: Memo<Method>,
    run_to: RwSignal<Option<u32>>,
) -> impl IntoView {
    view! {
        <details class="controls" open><summary class="eyebrow">"Настройки модели"</summary>
            {move||model_fields(lab,active.get(),run_to)}
        </details>
    }
}

fn model_fields(
    lab: RwSignal<Lab>,
    method: Method,
    run_to: RwSignal<Option<u32>>,
) -> impl IntoView {
    let read = move || lab.with(|s| s.parameters[index(method)]);
    let change = move |update: fn(&mut Parameters, f64), value: f64| {
        run_to.set(None);
        lab.update(|s| {
            update(&mut s.parameters[index(method)], value);
            s.refit(method);
        });
    };
    macro_rules! range {
        ($variant:ident,$field:ident,$label:literal,$min:expr,$max:expr,$step:expr,$ty:ty)=>{
            view!{<Slider label=$label min=$min max=Signal::derive($max) step=$step
                value=Signal::derive(move||if let Parameters::$variant{$field,..}=read(){$field as f64}else{$min})
                change=Callback::new(move|v|change(|params,v|{if let Parameters::$variant{$field,..}=params{*$field=v as $ty;}},v))/>}
        }
    }
    macro_rules! select {
        ($variant:ident,$field:ident,$label:literal,$options:expr)=>{
            view!{
                <label>{$label}
                    <select prop:value=move||if let Parameters::$variant{$field,..}=read(){$field.to_string()}else{String::new()}
                        on:change=move|ev|{if let Ok(v)=event_target_value(&ev).parse(){change(|p,v|{if let Parameters::$variant{$field,..}=p{*$field=v;}},v);}}>
                        {$options.iter().map(|&v|view!{<option value=v.to_string()>{v.to_string()}</option>}).collect_view()}
                    </select>
                </label>
            }
        }
    }
    let regularization = [0.0_f64, 0.0001, 0.01, 0.1, 1.0];
    match method {
        Method::Polynomial=>view!{
            {range!(Polynomial,degree,"Степень полинома",0.0,move||lab.with(|s|(s.training.len()-1).min(23) as f64),1.0,usize)}
            {select!(Polynomial,lambda,"Регуляризация λ",regularization)}
        }.into_any(),
        Method::Knn=>range!(Knn,k,"Число соседей",1.0,move||lab.with(|s|s.training.len() as f64),1.0,usize).into_any(),
        Method::Tree=>view!{
            {range!(Tree,depth,"Максимальная глубина",1.0,||16.0,1.0,usize)}
            {range!(Tree,leaf,"Минимум точек в листе",1.0,||8.0,1.0,usize)}
        }.into_any(),
        Method::Forest=>view!{
            {range!(Forest,trees,"Число деревьев",1.0,||64.0,1.0,usize)}
            {range!(Forest,depth,"Глубина деревьев",1.0,||16.0,1.0,usize)}
            {range!(Forest,leaf,"Минимум точек в листе",1.0,||8.0,1.0,usize)}
        }.into_any(),
        Method::Rbf=>view!{
            {range!(Rbf,centers,"Число колоколов",2.0,||48.0,1.0,usize)}
            {range!(Rbf,width,"Ширина σ",0.06,||2.0,0.02,f64)}
            {select!(Rbf,lambda,"Регуляризация λ",[1e-8_f64,0.0001,0.001,0.01,0.1,1.0])}
        }.into_any(),
        Method::Gp=>view!{
            {range!(Gp,length,"Масштаб связи ℓ",0.06,||2.0,0.02,f64)}
            {select!(Gp,noise,"Предполагаемый шум σₙ",[0.001_f64,0.03,0.07,0.15,0.3,0.6])}
        }.into_any(),
        Method::Neural=>view!{
            {range!(Neural,depth,"Скрытые слои",1.0,||4.0,1.0,usize)}
            {range!(Neural,hidden,"Нейроны в скрытом слое",2.0,||64.0,1.0,usize)}
            <Toggle label="Активация ReLU (иначе tanh)" checked=Signal::derive(move||matches!(read(),Parameters::Neural{relu:true,..}))
                change=Callback::new(move|v|change(|p,v|{if let Parameters::Neural{relu,..}=p{*relu=v!=0.0;}},if v{1.0}else{0.0}))/>
            {select!(Neural,rate,"Скорость обучения",[0.001_f64,0.003,0.01,0.03])}
            {select!(Neural,lambda,"Регуляризация L2",[0.0_f64,0.0001,0.001,0.01])}
            <label>"Цель обучения, эпох"
                <select prop:value=move||lab.with(|s|s.epochs.to_string()) on:change=move|ev|{
                    if let Ok(epochs)=event_target_value(&ev).parse(){lab.update(|s|s.epochs=epochs);}
                }>
                    {[1000,3000,6000,12000].into_iter().map(|v|view!{<option value=v>{v}</option>}).collect_view()}
                </select>
            </label>
            <div class="neural-actions">
                <button on:click=move|_|{
                    if run_to.get_untracked().is_some() {
                        run_to.set(None);
                    } else {
                        lab.update(|s|{if s.epoch()>=s.epochs{s.refit(Method::Neural);}});
                        run_to.set(Some(lab.with_untracked(|s|s.epochs)));
                    }
                }>{move||if run_to.get().is_some(){"Пауза"}else{"Обучать"}}</button>
                <button on:click=move|_|{run_to.set(Some(lab.with_untracked(|s|s.epoch().saturating_add(100))));}>"+100 эпох"</button>
                <button on:click=move|_|{run_to.set(None);lab.update(|s|s.refit(Method::Neural));}>"Начать заново"</button>
            </div>
            <p class="small">{move||lab.with(|s|format!("Эпоха {} / {}",s.epoch(),s.epochs))}</p>
        }.into_any(),
    }
}
