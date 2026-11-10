mod dqn;

use crate::data::Random;
use dqn::DeepQ;
pub use dqn::NetworkView;

pub const WIDTH: usize = 9;
pub const HEIGHT: usize = 7;
pub const START: usize = 54;
pub const GOAL: usize = 8;
pub const TRAP: usize = 25;
pub const WALLS: [usize; 19] = [
    10, 11, 12, 15, 16, 21, 24, 28, 30, 32, 33, 37, 39, 41, 46, 50, 52, 57, 59,
];
const STATES: usize = WIDTH * HEIGHT;
const ACTIONS: usize = 4;
pub const LIMIT: usize = 96;
const GAMMA: f64 = 0.95;
pub const ARROWS: [&str; ACTIONS] = ["↑", "→", "↓", "←"];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Method {
    QLearning,
    Sarsa,
    Dqn,
}

impl Method {
    pub const ALL: [Self; 3] = [Self::QLearning, Self::Sarsa, Self::Dqn];

    pub fn name(self) -> &'static str {
        match self {
            Self::QLearning => "Q-learning",
            Self::Sarsa => "SARSA",
            Self::Dqn => "DQN",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Goal,
    Trap,
    Limit,
}

#[derive(Clone, Copy)]
pub struct Episode {
    pub reward: f64,
    pub steps: usize,
    pub outcome: Outcome,
}

#[derive(Clone, Copy)]
struct Transition {
    from: usize,
    action: usize,
    reward: f64,
    to: usize,
    terminal: bool,
}

fn move_to(state: usize, action: usize) -> usize {
    let (x, y) = (state % WIDTH, state / WIDTH);
    let next = match action {
        0 if y > 0 => state - WIDTH,
        1 if x + 1 < WIDTH => state + 1,
        2 if y + 1 < HEIGHT => state + WIDTH,
        3 if x > 0 => state - 1,
        _ => state,
    };
    if WALLS.contains(&next) { state } else { next }
}

fn best(values: [f64; ACTIONS]) -> usize {
    (1..ACTIONS).fold(0, |a, b| if values[b] > values[a] { b } else { a })
}

enum Learner {
    Table(Box<[[f64; ACTIONS]; STATES]>),
    Network(Box<DeepQ>),
}

pub struct Experiment {
    pub method: Method,
    learner: Learner,
    rng: Random,
    replay_rng: Random,
    next_action: Option<usize>,
    pub position: usize,
    pub path: Vec<usize>,
    pub reward: f64,
    pub last_action: Option<usize>,
    pub last_reward: Option<f64>,
    pub outcome: Option<Outcome>,
    pub completed: usize,
    pub history: Vec<Episode>,
}

impl Experiment {
    pub fn new(method: Method, seed: u32) -> Self {
        let learner = match method {
            Method::Dqn => Learner::Network(Box::new(DeepQ::new(seed))),
            _ => Learner::Table(Box::new([[0.0; ACTIONS]; STATES])),
        };
        Self {
            method,
            learner,
            rng: Random(seed),
            replay_rng: Random(seed ^ 0xbade),
            next_action: None,
            position: START,
            path: vec![START],
            reward: 0.0,
            last_action: None,
            last_reward: None,
            outcome: None,
            completed: 0,
            history: Vec::new(),
        }
    }

    pub fn network_view(&self, state: usize) -> Option<NetworkView> {
        let Learner::Network(net) = &self.learner else {
            return None;
        };
        (state < STATES).then(|| net.view(state))
    }

    pub fn best_action(&self, state: usize) -> usize {
        best(self.values(state))
    }

    pub fn values(&self, state: usize) -> [f64; ACTIONS] {
        match &self.learner {
            Learner::Table(table) => table[state],
            Learner::Network(net) => net.values(state),
        }
    }

    fn action(&mut self, state: usize, epsilon: f64) -> usize {
        if self.rng.sample() < epsilon {
            return self.rng.index(ACTIONS);
        }
        let values = self.values(state);
        let top = values[best(values)];
        let mut ties = [0; ACTIONS];
        let mut n = 0;
        for (i, value) in values.into_iter().enumerate() {
            if value == top {
                ties[n] = i;
                n += 1;
            }
        }
        ties[self.rng.index(n)]
    }

    pub fn step(&mut self, epsilon: f64, step_cost: f64) {
        if self.outcome.take().is_some() {
            self.position = START;
            self.path.clear();
            self.path.push(START);
            self.reward = 0.0;
        }
        let from = self.position;
        let action = self
            .next_action
            .take()
            .unwrap_or_else(|| self.action(from, epsilon));
        let to = move_to(from, action);
        let outcome = match to {
            GOAL => Some(Outcome::Goal),
            TRAP => Some(Outcome::Trap),
            _ if self.path.len() == LIMIT => Some(Outcome::Limit),
            _ => None,
        };
        let reward = match outcome {
            Some(Outcome::Goal) => 1.0,
            Some(Outcome::Trap) => -1.0,
            _ => -step_cost,
        };
        let terminal = outcome.is_some();
        let next = if !terminal && self.method == Method::Sarsa {
            Some(self.action(to, epsilon))
        } else {
            None
        };
        let transition = Transition {
            from,
            action,
            reward,
            to,
            terminal,
        };
        match &mut self.learner {
            Learner::Table(table) => {
                let future = if terminal {
                    0.0
                } else {
                    table[to][next.unwrap_or_else(|| best(table[to]))]
                };
                table[from][action] += 0.25 * (reward + GAMMA * future - table[from][action]);
            }
            Learner::Network(net) => net.learn(transition, &mut self.replay_rng),
        }
        self.next_action = next;
        self.position = to;
        self.path.push(to);
        self.reward += reward;
        self.last_action = Some(action);
        self.last_reward = Some(reward);
        self.outcome = outcome;
        if let Some(outcome) = outcome {
            self.completed += 1;
            if self.history.len() == 100 {
                self.history.remove(0);
            }
            self.history.push(Episode {
                reward: self.reward,
                steps: self.path.len() - 1,
                outcome,
            });
        }
    }

    /// Завершает заданное число эпизодов, включая текущий незавершённый эпизод.
    pub fn train(&mut self, episodes: usize, epsilon: f64, step_cost: f64) {
        let target = self.completed.saturating_add(episodes);
        while self.train_batch(target, LIMIT, epsilon, step_cost) {}
    }

    /// Выполняет до `max_steps` ходов до абсолютной цели `target` по числу эпизодов.
    /// Возвращает `true`, если до цели нужны дополнительные ходы.
    pub fn train_batch(
        &mut self,
        target: usize,
        max_steps: usize,
        epsilon: f64,
        step_cost: f64,
    ) -> bool {
        for _ in 0..max_steps {
            if self.completed >= target {
                return false;
            }
            self.step(epsilon, step_cost);
        }
        self.completed < target
    }

    pub fn greedy_path(&self) -> Vec<usize> {
        let mut path = vec![START];
        let mut state = START;
        for _ in 0..LIMIT {
            let next = move_to(state, best(self.values(state)));
            if path.contains(&next) {
                break;
            }
            path.push(next);
            if matches!(next, GOAL | TRAP) {
                break;
            }
            state = next;
        }
        path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_training_respects_the_step_budget_and_stops_at_the_target() {
        let mut batched = Experiment::new(Method::Dqn, 72);
        assert!(batched.train_batch(2, 1, 0.45, 0.04));
        assert_eq!(batched.path.len(), 2);
        while batched.train_batch(2, 7, 0.45, 0.04) {}

        let mut manual = Experiment::new(Method::Dqn, 72);
        while manual.completed < 2 {
            manual.step(0.45, 0.04);
        }

        assert_eq!(batched.completed, 2);
        assert_eq!(batched.path, manual.path);
        for state in 0..STATES {
            assert_eq!(batched.values(state), manual.values(state));
        }
        assert!(!batched.train_batch(2, LIMIT, 0.45, 0.04));
        assert_eq!(batched.path, manual.path);
    }

    #[test]
    fn walls_and_boundaries_keep_the_agent_in_place() {
        assert_eq!(move_to(START, 1), START + 1);
        assert_eq!(move_to(START + 2, 1), START + 2);
        assert_eq!(move_to(START, 3), START);
        assert_eq!(move_to(START, 0), START - WIDTH);
    }

    #[test]
    fn all_methods_learn_to_reach_the_goal_on_the_demo_seed() {
        for method in Method::ALL {
            let mut agent = Experiment::new(method, 72);
            agent.train(300, 0.45, 0.04);
            assert_eq!(agent.completed, 300);
            let path = agent.greedy_path();
            assert_eq!(path.last(), Some(&GOAL), "{}: {path:?}", method.name());
            assert!(path.len() <= 19, "{}: {path:?}", method.name());
        }
    }

    #[test]
    fn inspected_network_reconstructs_the_agents_action_values() {
        let mut agent = Experiment::new(Method::Dqn, 72);
        agent.train(2, 0.45, 0.04);
        for cell in [START, START - WIDTH] {
            let net = agent.network_view(cell).unwrap();
            assert!(net.updates > 0);
            for (action, q) in agent.values(cell).into_iter().enumerate() {
                let reconstructed = net.output_bias[action]
                    + net
                        .activation
                        .iter()
                        .zip(net.output_weights[action])
                        .map(|(h, w)| h * w)
                        .sum::<f64>();
                assert!((q - reconstructed).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn preview_does_not_train_or_change_episode() {
        let mut agent = Experiment::new(Method::Dqn, 72);
        agent.train(2, 0.25, 0.04);
        let before = agent.values(START);
        let _ = agent.greedy_path();
        assert_eq!(agent.completed, 2);
        assert_eq!(agent.values(START), before);
        assert!(agent.outcome.is_some());
    }
}
