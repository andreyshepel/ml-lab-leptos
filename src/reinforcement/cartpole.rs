mod dqn;

use std::f64::consts::{PI, TAU};

use crate::data::Random;
use dqn::Agent;

pub const TRACK_EDGE: f64 = 2.4;
pub const DECISION_MS: u64 = 100;
const SEED: u32 = 72;
const PHYSICS_MS: u64 = 10;
const PHYSICS_STEPS: u64 = DECISION_MS / PHYSICS_MS;

/// Наблюдение тележки и свободно вращающегося шеста. Нулевой угол направлен вверх.
#[derive(Clone, Copy)]
pub struct State {
    pub position: f64,
    pub velocity: f64,
    pub angle: f64,
    pub angular_velocity: f64,
}

impl State {
    fn new(seed: u32, from_below: bool) -> Self {
        let mut rng = Random(seed);
        let mut state = Self {
            position: 0.1 * rng.sample() - 0.05,
            velocity: 0.0,
            angle: 0.1 * rng.sample() - 0.05,
            angular_velocity: 0.0,
        };
        if from_below {
            state.angle = wrap(state.angle + PI);
        }
        state
    }

    /// Высота конца шеста относительно шарнира, в долях длины шеста.
    pub fn height(self) -> f64 {
        self.angle.cos()
    }

    fn inputs(self) -> [f64; 5] {
        let (sin, cos) = self.angle.sin_cos();
        [
            self.position / TRACK_EDGE,
            self.velocity / 4.0,
            sin,
            cos,
            self.angular_velocity / 8.0,
        ]
    }

    fn advance(self, action: Action) -> Self {
        // Полунеявный Эйлер: сначала скорости, затем координаты.
        const DT: f64 = PHYSICS_MS as f64 / 1000.0;
        const MASS: f64 = 1.1;
        const HALF_LENGTH: f64 = 0.5;
        const POLE_MOMENT: f64 = 0.1 * HALF_LENGTH;
        let force = match action {
            Action::Left => -10.0,
            Action::Right => 10.0,
        };
        let (sin, cos) = self.angle.sin_cos();
        let pole_inertia = HALF_LENGTH * (4.0 / 3.0);
        let temp = (force - 0.08 * self.velocity
            + POLE_MOMENT * self.angular_velocity.powi(2) * sin)
            / MASS;
        let angular_acceleration =
            (9.8 * sin - cos * temp) / (pole_inertia - POLE_MOMENT * cos.powi(2) / MASS);
        let acceleration = temp - POLE_MOMENT * angular_acceleration * cos / MASS;
        let velocity = self.velocity + DT * acceleration;
        let position = self.position + DT * velocity;
        let at_wall = position.abs() >= TRACK_EDGE;
        // При упоре поворот продолжается вокруг неподвижного шарнира.
        // Удар гасит скорость тележки, сохраняя момент шеста вокруг шарнира.
        let angular_velocity = self.angular_velocity
            + DT * (if at_wall {
                (9.8 * sin + self.velocity * cos / DT) / pole_inertia
            } else {
                angular_acceleration
            } - 0.08 * self.angular_velocity);
        Self {
            position: position.clamp(-TRACK_EDGE, TRACK_EDGE),
            velocity: if at_wall { 0.0 } else { velocity },
            angle: wrap(self.angle + DT * angular_velocity),
            angular_velocity,
        }
    }
}

fn wrap(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

/// Две силы фиксированной величины, приложенные к тележке.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Left,
    Right,
}

impl Action {
    const ALL: [Self; 2] = [Self::Left, Self::Right];

    fn best(values: [f64; 2]) -> Self {
        if values[1] > values[0] {
            Self::Right
        } else {
            Self::Left
        }
    }
}

/// Награда по высоте конца шеста на каждом физическом шаге.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reward {
    AboveOnly,
    Signed,
}

impl Reward {
    fn at(self, state: State) -> f64 {
        match self {
            Self::AboveOnly => state.height().max(0.0),
            Self::Signed => state.height(),
        }
    }
}

/// Кадр симуляции и накопленные показатели непрерывного запуска.
#[derive(Clone, Copy)]
pub struct Frame {
    pub state: State,
    pub steps: usize,
    pub reward: f64,
    pub mean_reward: Option<f64>,
    pub above_fraction: Option<f64>,
    pub decision: Option<Decision>,
}

/// Оценки сети перед применением силы и фактически выбранное действие.
#[derive(Clone, Copy)]
pub struct Decision {
    pub values: [f64; 2],
    pub action: Action,
    pub exploratory: bool,
}

/// Архитектура скрытых слоёв и настройки Adam для CartPole.
#[derive(Clone, Copy, PartialEq)]
pub struct NetworkConfig {
    pub depth: usize,
    pub width: usize,
    pub relu: bool,
    pub rate: f64,
    pub l2: f64,
}

impl NetworkConfig {
    pub const MAX_DEPTH: usize = 4;
    pub const MIN_WIDTH: usize = 2;
    pub const MAX_WIDTH: usize = 64;
    pub const RATES: [f64; 5] = [0.0001, 0.0003, 0.001, 0.003, 0.01];
    pub const L2_OPTIONS: [f64; 4] = [0.0, 0.0001, 0.001, 0.01];

    fn validate(self) -> Result<(), &'static str> {
        if !(1..=Self::MAX_DEPTH).contains(&self.depth)
            || !(Self::MIN_WIDTH..=Self::MAX_WIDTH).contains(&self.width)
        {
            return Err("Выбери 1–4 скрытых слоя и 2–64 нейрона на слой.");
        }
        if !Self::RATES.contains(&self.rate) || !Self::L2_OPTIONS.contains(&self.l2) {
            return Err("Выбери скорость обучения и L2 из доступных значений.");
        }
        Ok(())
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            depth: 1,
            width: 48,
            relu: true,
            rate: 0.001,
            l2: 0.0,
        }
    }
}

const WINDOW: usize = 100;

/// Онлайн-обучение на той же траектории, которую показывает интерфейс.
pub struct Training {
    agent: Agent,
    rng: Random,
    frame: Frame,
    reward: Reward,
    learning_steps: usize,
    recent: [(f64, f64); WINDOW],
    sums: (f64, f64),
}

impl Training {
    /// Создаёт новую сеть и воспроизводимое начальное положение.
    pub fn new(seed: u32, from_below: bool, reward: Reward) -> Self {
        Self::build(seed, from_below, reward, NetworkConfig::default())
    }

    /// Создаёт новую сессию с заданной сетью или возвращает причину отказа.
    pub fn with_network(
        seed: u32,
        from_below: bool,
        reward: Reward,
        config: NetworkConfig,
    ) -> Result<Self, &'static str> {
        config.validate()?;
        Ok(Self::build(seed, from_below, reward, config))
    }

    fn build(seed: u32, from_below: bool, reward: Reward, config: NetworkConfig) -> Self {
        Self {
            agent: Agent::new(SEED, config),
            rng: Random(seed ^ 0x61ab),
            frame: Frame {
                state: State::new(seed, from_below),
                steps: 0,
                reward: 0.0,
                mean_reward: None,
                above_fraction: None,
                decision: None,
            },
            reward,
            learning_steps: 0,
            recent: [(0.0, 0.0); WINDOW],
            sums: (0.0, 0.0),
        }
    }

    /// Вероятность случайного действия на следующем такте управления.
    pub fn epsilon(&self) -> f64 {
        (0.6 - self.learning_steps as f64 / 12_000.0).max(0.05)
    }

    pub fn frame(&self) -> Frame {
        self.frame
    }

    /// Меняет положение шеста в текущем запуске, сохраняя опыт и веса.
    pub fn drop_pole(&mut self) {
        self.frame.state.angle = PI - 0.02;
        self.frame.state.angular_velocity = 0.0;
        self.frame.decision = None;
        self.agent.interrupt();
    }

    /// Выполняет до budget тактов управления. Возвращает false при численном сбое.
    pub fn advance(&mut self, budget: usize, learning: bool) -> bool {
        if !learning {
            self.agent.interrupt();
        }
        for _ in 0..budget {
            let from = self.frame.state.inputs();
            let values = self.agent.values(from);
            let exploratory = learning && self.rng.sample() < self.epsilon();
            let action = if exploratory {
                Action::ALL[self.rng.index(Action::ALL.len())]
            } else {
                Action::best(values)
            };
            let mut state = self.frame.state;
            let mut reward = 0.0;
            let mut above = 0;
            // Сила держится 100 мс; награда усредняется по десяти шагам физики.
            for _ in 0..PHYSICS_STEPS {
                state = state.advance(action);
                reward += self.reward.at(state);
                above += usize::from(state.height() > 0.0);
            }
            reward /= PHYSICS_STEPS as f64;
            let above = above as f64 / PHYSICS_STEPS as f64;
            if !reward.is_finite()
                || values
                    .iter()
                    .chain(&state.inputs())
                    .any(|value| !value.is_finite())
                || (learning
                    && !self
                        .agent
                        .learn(from, action, state.inputs(), reward, &mut self.rng))
            {
                return false;
            }
            let i = self.frame.steps % WINDOW;
            self.sums.0 += reward - self.recent[i].0;
            self.sums.1 += above - self.recent[i].1;
            self.recent[i] = (reward, above);
            self.learning_steps += usize::from(learning);
            self.frame.steps += 1;
            let count = self.frame.steps.min(WINDOW) as f64;
            self.frame = Frame {
                state,
                steps: self.frame.steps,
                reward,
                mean_reward: Some(self.sums.0 / count),
                above_fraction: Some((self.sums.1 / count).clamp(0.0, 1.0)),
                decision: Some(Decision {
                    values,
                    action,
                    exploratory,
                }),
            };
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_stops_cart_but_leaves_pole_free_and_allows_reversing() {
        let initial = State {
            position: TRACK_EDGE,
            velocity: 0.0,
            angle: PI / 2.0,
            angular_velocity: 0.0,
        };
        let blocked = initial.advance(Action::Right);
        assert_eq!(blocked.position, TRACK_EDGE);
        assert_eq!(blocked.velocity, 0.0);
        assert!(blocked.angle > PI / 2.0);
        let reversed = blocked.advance(Action::Left);
        assert!(reversed.position < TRACK_EDGE);
        assert!(reversed.velocity < 0.0);
    }

    #[test]
    fn reward_distinguishes_both_sides_of_horizontal() {
        let mut state = State::new(SEED, false);
        for (angle, expected) in [(0.0, 1.0), (PI / 2.0, 0.0), (PI, -1.0)] {
            state.angle = angle;
            assert!((Reward::Signed.at(state) - expected).abs() < 1e-12);
            assert!((Reward::AboveOnly.at(state) - expected.max(0.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn network_settings_validate_and_train_at_the_supported_limit() {
        let config = NetworkConfig {
            depth: NetworkConfig::MAX_DEPTH,
            width: NetworkConfig::MAX_WIDTH,
            relu: false,
            rate: 0.0003,
            l2: 0.001,
        };
        assert!(
            Training::with_network(
                9173,
                true,
                Reward::AboveOnly,
                NetworkConfig {
                    depth: config.depth + 1,
                    ..config
                }
            )
            .is_err()
        );
        let mut run = Training::with_network(9173, true, Reward::AboveOnly, config)
            .expect("поддерживаемая конфигурация");
        let probe = State::new(SEED, false).inputs();
        let initial_values = run.agent.values(probe);
        assert!(run.advance(265, true));
        assert_ne!(run.agent.values(probe), initial_values);
    }

    #[test]
    fn learning_and_manual_drop_continue_the_same_run() {
        let mut run = Training::new(9173, true, Reward::AboveOnly);
        assert!(run.frame().state.height() < 0.0);
        let probe = State::new(SEED, false).inputs();
        let initial_values = run.agent.values(probe);
        assert!(run.advance(520, true));
        let learned_values = run.agent.values(probe);
        assert_ne!(learned_values, initial_values);
        run.drop_pole();
        assert!(run.frame().state.height() < 0.0);
        assert!(run.advance(1, false));
        assert_eq!(run.frame().steps, 521);
        assert_eq!(run.agent.values(probe), learned_values);
    }
}
