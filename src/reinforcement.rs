pub mod cartpole;
mod dqn;

use crate::data::Random;
use dqn::DeepQ;
pub use dqn::NetworkView;

pub const WIDTH: usize = 15;
pub const HEIGHT: usize = 15;
pub const START: usize = (HEIGHT - 1) * WIDTH;
const MAP: [&[u8; WIDTH]; HEIGHT] = [
    b"...#.....#....G",
    b".#.#.###.#.##..",
    b"T#...#.....#...",
    b".###.#.###.#.#.",
    b".....#T....#.#.",
    b".###...###...#T",
    b".#G..#...#.#...",
    b".#.#.#.#...#.#.",
    b"...#...#.###...",
    b".###.T.#.....#G",
    b"...#...###.#.#.",
    b".#...#.....#...",
    b".#.G.#.###.###.",
    b".#...#...T.....",
    b"S..#...T......G",
];
const STATES: usize = WIDTH * HEIGHT;
const ACTIONS: usize = 4;
pub const EPISODE_LIMIT: usize = 2_000;
const PATH_CAPACITY: usize = 512;

/// Тип клетки на общей карте всех трёх методов.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    Open,
    Wall,
    Goal,
    Trap,
}

/// Возвращает тип клетки по её индексу на поле.
pub fn tile(index: usize) -> Tile {
    match MAP[index / WIDTH][index % WIDTH] {
        b'#' => Tile::Wall,
        b'G' => Tile::Goal,
        b'T' => Tile::Trap,
        _ => Tile::Open,
    }
}

pub const GOAL_COUNT: usize = 5;
const MASKS: usize = 1 << GOAL_COUNT;

const fn goal_bits() -> [usize; STATES] {
    let mut bits = [0; STATES];
    let mut cell = 0;
    let mut goal = 0;
    while cell < STATES {
        if MAP[cell / WIDTH][cell % WIDTH] == b'G' {
            bits[cell] = 1 << goal;
            goal += 1;
        }
        cell += 1;
    }
    assert!(goal == GOAL_COUNT);
    bits
}
const GOAL_BITS: [usize; STATES] = goal_bits();

/// Собранные в текущем эпизоде награды.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Collected(usize);

impl Collected {
    pub fn contains(self, cell: usize) -> bool {
        self.0 & GOAL_BITS[cell] != 0
    }

    pub fn count(self) -> usize {
        self.0.count_ones() as usize
    }

    fn complete(self) -> bool {
        self.0 == MASKS - 1
    }
}

#[derive(Clone, Copy)]
struct State {
    cell: usize,
    collected: Collected,
}

impl State {
    const START: Self = Self {
        cell: START,
        collected: Collected(0),
    };

    fn index(self) -> usize {
        self.collected.0 * STATES + self.cell
    }

    fn advance(self, action: usize) -> Self {
        let cell = move_to(self.cell, action);
        Self {
            cell,
            collected: Collected(self.collected.0 | GOAL_BITS[cell]),
        }
    }
}

/// Жадный маршрут от старта и собранные на нём награды.
pub struct Preview {
    pub path: Vec<usize>,
    pub collected: Collected,
}

const GAMMA: f64 = 0.98;
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

/// Итог завершённого или ограниченного по длине эпизода.
#[derive(Clone, Copy)]
pub struct Episode {
    pub reward: f64,
    pub steps: usize,
    pub collected: usize,
}

#[derive(Clone, Copy)]
struct Transition {
    from: State,
    action: usize,
    reward: f64,
    to: State,
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
    if tile(next) == Tile::Wall {
        state
    } else {
        next
    }
}

fn best(values: [f64; ACTIONS]) -> usize {
    (1..ACTIONS).fold(0, |a, b| if values[b] > values[a] { b } else { a })
}

enum Learner {
    Table(Vec<[f64; ACTIONS]>),
    Network(Box<DeepQ>),
}

pub struct Experiment {
    pub method: Method,
    learner: Learner,
    rng: Random,
    replay_rng: Random,
    next_action: Option<usize>,
    pub position: usize,
    pub collected: Collected,
    pub path: Vec<usize>,
    pub steps: usize,
    pub reward: f64,
    pub last_action: Option<usize>,
    pub last_reward: Option<f64>,
    pub completed: usize,
    pub history: Vec<Episode>,
}

impl Experiment {
    pub fn new(method: Method, seed: u32) -> Self {
        let learner = match method {
            Method::Dqn => Learner::Network(Box::new(DeepQ::new(seed))),
            _ => Learner::Table(vec![[0.0; ACTIONS]; STATES * MASKS]),
        };
        Self {
            method,
            learner,
            rng: Random(seed),
            replay_rng: Random(seed ^ 0xbade),
            next_action: None,
            position: START,
            collected: Collected::default(),
            path: vec![START],
            steps: 0,
            reward: 0.0,
            last_action: None,
            last_reward: None,
            completed: 0,
            history: Vec::new(),
        }
    }

    pub fn network_view(&self, cell: usize, collected: Collected) -> Option<NetworkView> {
        let Learner::Network(net) = &self.learner else {
            return None;
        };
        (cell < STATES).then(|| net.view(State { cell, collected }))
    }

    pub fn best_action(&self, cell: usize, collected: Collected) -> usize {
        best(self.values(cell, collected))
    }

    pub fn values(&self, cell: usize, collected: Collected) -> [f64; ACTIONS] {
        self.state_values(State { cell, collected })
    }

    fn state_values(&self, state: State) -> [f64; ACTIONS] {
        match &self.learner {
            Learner::Table(table) => table[state.index()],
            Learner::Network(net) => net.values(state),
        }
    }

    fn action(&mut self, state: State, epsilon: f64) -> usize {
        if self.rng.sample() < epsilon {
            return self.rng.index(ACTIONS);
        }
        let values = self.state_values(state);
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

    /// Эпизод заканчивается после сбора всех наград или по лимиту ходов.
    pub fn finished(&self) -> bool {
        self.collected.complete() || self.steps >= EPISODE_LIMIT
    }

    pub fn step(&mut self, epsilon: f64, step_cost: f64) {
        if self.finished() {
            self.position = START;
            self.collected = Collected::default();
            self.path.clear();
            self.path.push(START);
            self.steps = 0;
            self.reward = 0.0;
        }
        let from = State {
            cell: self.position,
            collected: self.collected,
        };
        let action = self
            .next_action
            .take()
            .unwrap_or_else(|| self.action(from, epsilon));
        let to = from.advance(action);
        let reward = if to.collected != from.collected {
            1.0
        } else if to.cell != from.cell && tile(to.cell) == Tile::Trap {
            -1.0
        } else {
            -step_cost
        };
        let terminal = to.collected.complete();
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
                    table[to.index()][next.unwrap_or_else(|| best(table[to.index()]))]
                };
                table[from.index()][action] +=
                    0.25 * (reward + GAMMA * future - table[from.index()][action]);
            }
            Learner::Network(net) => net.learn(transition, &mut self.replay_rng),
        }
        self.next_action = if self.steps + 1 < EPISODE_LIMIT {
            next
        } else {
            None
        };
        self.position = to.cell;
        self.collected = to.collected;
        // Храним только видимый хвост маршрута. Счётчик ходов от него не зависит.
        if self.path.len() == PATH_CAPACITY {
            self.path.drain(..PATH_CAPACITY / 2);
        }
        self.path.push(to.cell);
        self.steps += 1;
        self.reward += reward;
        self.last_action = Some(action);
        self.last_reward = Some(reward);
        if self.finished() {
            self.completed += 1;
            if self.history.len() == 100 {
                self.history.remove(0);
            }
            self.history.push(Episode {
                reward: self.reward,
                steps: self.steps,
                collected: self.collected.count(),
            });
        }
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

    pub fn greedy_path(&self) -> Preview {
        let mut path = vec![START];
        let mut state = State::START;
        let mut seen = [false; STATES * MASKS];
        seen[state.index()] = true;
        loop {
            let next = state.advance(best(self.state_values(state)));
            if seen[next.index()] {
                break;
            }
            seen[next.index()] = true;
            path.push(next.cell);
            state = next;
            if state.collected.complete() {
                break;
            }
        }
        Preview {
            path,
            collected: state.collected,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_training_respects_the_step_budget_and_stops_at_the_target() {
        let mut batched = Experiment::new(Method::QLearning, 72);
        assert!(batched.train_batch(2, 1, 0.45, 0.01));
        assert_eq!(batched.path.len(), 2);
        for _ in 0..1000 {
            if !batched.train_batch(2, 7, 0.45, 0.01) {
                break;
            }
        }
        assert_eq!(batched.completed, 2);

        let mut manual = Experiment::new(Method::QLearning, 72);
        while manual.completed < 2 && manual.steps < 7000 {
            manual.step(0.45, 0.01);
        }

        assert_eq!(manual.completed, 2);
        assert_eq!(batched.path, manual.path);
        for state in 0..STATES {
            assert_eq!(
                batched.values(state, batched.collected),
                manual.values(state, manual.collected)
            );
        }
        assert!(!batched.train_batch(2, 7, 0.45, 0.01));
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
    fn all_rewards_and_traps_are_reachable() {
        use std::collections::VecDeque;

        let mut distances = [None; STATES];
        distances[START] = Some(0);
        let mut queue = VecDeque::from([(START, 0)]);
        while let Some((cell, distance)) = queue.pop_front() {
            for action in 0..ACTIONS {
                let next = move_to(cell, action);
                if distances[next].is_none() {
                    distances[next] = Some(distance + 1);
                    queue.push_back((next, distance + 1));
                }
            }
        }
        for (cell, distance) in distances.into_iter().enumerate() {
            if tile(cell) != Tile::Wall {
                assert!(distance.is_some(), "клетка {cell}");
            }
        }
    }

    #[test]
    fn rewards_are_one_time_and_only_last_pickup_finishes_the_episode() {
        let mut agent = Experiment::new(Method::QLearning, 72);
        for (cell, bit) in GOAL_BITS
            .into_iter()
            .enumerate()
            .filter(|(_, bit)| *bit != 0)
        {
            let (neighbor, action) = (0..STATES)
                .filter(|&other| tile(other) != Tile::Wall && other != cell)
                .find_map(|other| {
                    (0..ACTIONS)
                        .find(|&a| move_to(other, a) == cell)
                        .map(|a| (other, a))
                })
                .expect("награда достижима");
            agent.position = neighbor;
            agent.next_action = Some(action);
            agent.step(0.0, 0.01);
            assert_eq!(agent.last_reward, Some(1.0));
            assert!(agent.collected.contains(cell));
            if agent.collected.complete() {
                assert!(agent.finished());
                break;
            }
            assert!(!agent.finished());
            agent.position = neighbor;
            agent.next_action = Some(action);
            agent.step(0.0, 0.01);
            assert_eq!(agent.last_reward, Some(-0.01), "маска {bit}");
        }
        agent.step(0.0, 0.01);
        assert_eq!(agent.collected.count(), 0);
    }

    #[test]
    fn greedy_q_learning_collects_all_rewards_on_the_demo_seed() {
        let mut agent = Experiment::new(Method::QLearning, 72);
        assert!(!agent.train_batch(200, 200_000, 0.45, 0.01));
        let preview = agent.greedy_path();
        assert_eq!(preview.collected.count(), GOAL_COUNT);
    }

    #[test]
    fn unfinished_episode_reaches_the_new_limit_and_keeps_a_bounded_path() {
        let mut agent = Experiment::new(Method::QLearning, 72);
        for _ in 1..EPISODE_LIMIT {
            agent.next_action = Some(3);
            agent.step(0.0, 0.01);
        }
        assert!(!agent.finished());
        assert_eq!(agent.completed, 0);
        agent.next_action = Some(3);
        agent.step(0.0, 0.01);
        assert!(agent.finished());
        assert_eq!(agent.completed, 1);
        assert_eq!(agent.history[0].steps, EPISODE_LIMIT);
        assert_eq!(agent.history[0].collected, 0);
        assert!(agent.path.len() <= PATH_CAPACITY);
        agent.step(0.0, 0.01);
        assert_eq!(agent.steps, 1);
    }

    #[test]
    fn entering_a_trap_costs_one_without_ending_the_episode() {
        let mut agent = Experiment::new(Method::QLearning, 72);
        agent.position = 15;
        agent.next_action = Some(2);
        agent.step(0.0, 0.01);
        assert_eq!(tile(agent.position), Tile::Trap);
        assert_eq!(agent.last_reward, Some(-1.0));
        assert!(!agent.finished());
    }

    #[test]
    fn inspected_network_reconstructs_the_agents_action_values() {
        let mut agent = Experiment::new(Method::Dqn, 72);
        assert!(agent.train_batch(2, 32, 0.45, 0.04));
        for cell in [START, START - WIDTH] {
            let net = agent.network_view(cell, agent.collected).unwrap();
            assert!(net.updates > 0);
            for (action, q) in agent.values(cell, agent.collected).into_iter().enumerate() {
                let reconstructed = net.output_bias[action]
                    + net
                        .second
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
        assert!(agent.train_batch(2, 32, 0.25, 0.04));
        let before = agent.values(START, agent.collected);
        let steps = agent.steps;
        let collected = agent.collected;
        let _ = agent.greedy_path();
        assert_eq!(agent.completed, 0);
        assert_eq!(agent.steps, steps);
        assert_eq!(agent.collected, collected);
        assert_eq!(agent.values(START, agent.collected), before);
    }
}
