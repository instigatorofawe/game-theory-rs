use ndarray::*;
use rayon::prelude::*;
use std::fmt::{Debug, Display};

pub trait Node: Debug + Sync + Send + Display {
    fn name(&self) -> String;
    fn state_probabilities(&self) -> Array<f64, Ix1>;
    fn payouts(&self) -> Array<f64, Ix1>;
    fn strategy(&self) -> Option<Array<f64, Ix2>>;
    fn avg_strategy(&self) -> Option<Array<f64, Ix2>>;
    fn children(&self) -> Option<&Vec<Box<dyn Node>>>;

    fn set_state_probabilities(&mut self, p: Array<f64, Ix1>);
    fn update_probabilities(&mut self);
    fn update_ev(&mut self);
    fn update_strategy(&mut self);
}

#[derive(Debug)]
pub struct ActionNode {
    pub name: String,
    pub state_probabilities: Array<f64, Ix1>, // Indexed by state
    pub total_probabilities: Array<f64, Ix1>, // Indexed by infoset
    pub evs: Array<f64, Ix1>,                 // Indexed by state
    pub infosets: Vec<Vec<usize>>,            // Indexed by infoset, member(state)
    pub strategy: Array<f64, Ix2>,            // Indexed by action, infoset
    pub avg_strategy: Array<f64, Ix2>,        // Indexed by action, infoset
    pub regrets: Array<f64, Ix2>,             // Indexed by action, infoset
    pub children: Vec<Box<dyn Node>>,
    pub sign: i8,        // 1 for positive payout, -1 for negative payout
    pub iter_count: u64, // CFR iteration count
}

impl ActionNode {
    fn expand_strategy(&self) -> Array<f64, Ix2> {
        let mut result: Array<f64, Ix2> =
            Array::zeros((self.children.len(), self.state_probabilities.len()));

        self.infosets
            .iter()
            .enumerate()
            .map(|(infoset_index, infoset_contents)| {
                infoset_contents
                    .iter()
                    .map(|state_index| {
                        result
                            .slice_mut(s![.., *state_index])
                            .assign(&self.strategy.slice(s![.., infoset_index]))
                    })
                    .for_each(drop);
            })
            .for_each(drop);
        result
    }

    fn infoset_probabilities(&self, state_probabilities: &Array<f64, Ix1>) -> Array<f64, Ix1> {
        let result: Array<f64, Ix1> = self
            .infosets
            .iter()
            .map(|x| x.iter().map(|i| state_probabilities[*i]).sum())
            .collect();
        result
    }

    fn infoset_evs(
        &self,
        evs: &Array<f64, Ix1>,
        state_probabilities: &Array<f64, Ix1>,
    ) -> Array<f64, Ix1> {
        let result: Array<f64, Ix1> = self
            .infosets
            .iter()
            .map(|x| {
                x.iter()
                    .map(|state_index| evs[*state_index] * state_probabilities[*state_index])
                    .sum()
            })
            .collect::<Array<f64, Ix1>>()
            / &self
                .infoset_probabilities(state_probabilities)
                .iter()
                .map(|x| match x {
                    0. => 1.,
                    _ => *x,
                })
                .collect::<Array<f64, Ix1>>();
        result
    }

    fn action_evs(&self) -> Array<f64, Ix2> {
        let mut result: Array<f64, Ix2> = Array::zeros((self.children.len(), self.infosets.len()));
        self.children
            .iter()
            .enumerate()
            .map(|(action_index, child)| {
                result
                    .slice_mut(s![action_index, ..])
                    .assign(&self.infoset_evs(&child.payouts(), &child.state_probabilities()));
            })
            .for_each(drop);
        result
    }

    fn current_regret(&self) -> Array<f64, Ix2> {
        (self.action_evs() - self.infoset_evs(&self.evs, &self.state_probabilities))
            * self.sign as f64
    }

    fn regret_match(&self) -> Array<f64, Ix2> {
        const EPSILON: f64 = 1e-8;
        let mut result: Array<f64, Ix2> = Array::zeros((self.children.len(), self.infosets.len()));
        self.regrets
            .axis_iter(Axis(1))
            .enumerate()
            .map(|(infoset_index, x)| {
                let nonzero_regrets: Array<f64, Ix1> = x
                    .iter()
                    .map(|y| match *y < 0. {
                        true => 0.,
                        _ => *y,
                    })
                    .collect();

                if nonzero_regrets.sum() == 0. {
                    result
                        .slice_mut(s![.., infoset_index])
                        .assign(&Array::from_elem(
                            nonzero_regrets.len(),
                            1. / self.children.len() as f64,
                        ));
                } else {
                    result.slice_mut(s![.., infoset_index]).assign(
                        &((&nonzero_regrets + EPSILON) / (&nonzero_regrets + EPSILON).sum()),
                    );
                }
            })
            .for_each(drop);
        result
    }
}

impl Display for ActionNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ActionNode {{")?;
        writeln!(f, "  Name: {}", self.name)?;
        writeln!(f, "  State probabilities: {}", self.state_probabilities)?;
        writeln!(f, "  Total probabilities: {}", self.total_probabilities)?;
        writeln!(f, "  EVs: {}", self.evs)?;
        writeln!(f, "  Infosets: {:?}", self.infosets)?;
        writeln!(f, "  Strategy:\n{}", self.strategy)?;
        writeln!(f, "  Average Strategy:\n{}", self.avg_strategy)?;
        writeln!(f, "  Regrets: {}", self.regrets)?;
        writeln!(
            f,
            "  Children: {:?}",
            self.children
                .iter()
                .map(|x| x.name())
                .collect::<Vec<String>>()
        )?;
        writeln!(f, "}}")?;

        for child in &self.children {
            writeln!(f, "{}", child)?;
        }
        write!(f, "")
    }
}

impl Node for ActionNode {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn state_probabilities(&self) -> Array<f64, Ix1> {
        self.state_probabilities.clone()
    }

    fn payouts(&self) -> Array<f64, Ix1> {
        self.evs.clone()
    }

    fn set_state_probabilities(&mut self, p: Array<f64, Ix1>) {
        self.state_probabilities = p;
    }

    fn update_probabilities(&mut self) {
        if self.total_probabilities.sum() == 0. {
            self.total_probabilities = self.infoset_probabilities(&self.state_probabilities);
        }

        let expanded_strategy = self.expand_strategy();

        self.children
            .par_iter_mut()
            .enumerate()
            .map(|(action_index, child)| {
                child.set_state_probabilities(
                    self.state_probabilities.clone()
                        * expanded_strategy.slice(s![action_index, ..]),
                );
                child.update_probabilities();
            })
            .for_each(drop);
    }

    fn update_ev(&mut self) {
        self.children
            .par_iter_mut()
            .map(|x| x.update_ev())
            .for_each(drop);

        // Compute current node EV from children
        let n_states = self.state_probabilities.len();
        self.evs = self
            .children
            .iter()
            .map(|child| child.payouts() * child.state_probabilities())
            .fold(Array::zeros(n_states), |f, x| f + x)
            / self
                .state_probabilities
                .iter()
                .map(|x| match x {
                    0. => 1.,
                    _ => *x,
                })
                .collect::<Array<f64, Ix1>>();
    }

    fn update_strategy(&mut self) {
        let infoset_probabilities = self.infoset_probabilities(&self.state_probabilities);

        self.regrets = (&self.regrets + self.current_regret() * &infoset_probabilities)
            * self.iter_count as f64
            / (self.iter_count as f64 + 1.);

        self.strategy = self.regret_match();

        self.avg_strategy = (&self.avg_strategy * &self.total_probabilities
            + &self.strategy * &infoset_probabilities)
            / (&self.total_probabilities + &infoset_probabilities);

        self.iter_count += 1;
        self.total_probabilities = &self.total_probabilities + infoset_probabilities;

        self.children
            .par_iter_mut()
            .map(|x| x.update_strategy())
            .for_each(drop);
    }

    fn strategy(&self) -> Option<Array<f64, Ix2>> {
        Some(self.strategy.clone())
    }

    fn avg_strategy(&self) -> Option<Array<f64, Ix2>> {
        Some(self.avg_strategy.clone())
    }

    fn children(&self) -> Option<&Vec<Box<dyn Node>>> {
        Some(&self.children)
    }
}

impl Display for TerminalNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "ActionNode {{")?;
        writeln!(f, "  Name: {}", self.name)?;
        writeln!(f, "  State probabilities: {}", self.state_probabilities)?;
        writeln!(f, "  Payouts: {}", self.payouts)?;
        write!(f, "}}")
    }
}

#[derive(Debug)]
pub struct TerminalNode {
    pub name: String,
    pub state_probabilities: Array<f64, Ix1>,
    pub payouts: Array<f64, Ix1>,
}

impl Node for TerminalNode {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn state_probabilities(&self) -> Array<f64, Ix1> {
        self.state_probabilities.clone()
    }

    fn payouts(&self) -> Array<f64, Ix1> {
        self.payouts.clone()
    }

    fn set_state_probabilities(&mut self, p: Array<f64, Ix1>) {
        self.state_probabilities = p;
    }

    fn update_probabilities(&mut self) {
        // Nothing to do for terminal nodes
    }

    fn update_ev(&mut self) {
        // Nothing to do for terminal nodes
    }

    fn update_strategy(&mut self) {
        // Nothing to do for terminal nodes
    }

    fn strategy(&self) -> Option<Array<f64, Ix2>> {
        // Terminal nodes have no strategy
        None
    }

    fn avg_strategy(&self) -> Option<Array<f64, Ix2>> {
        // Terminal nodes have no strategy
        None
    }

    fn children(&self) -> Option<&Vec<Box<dyn Node>>> {
        // Terminal nodes have no children
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_strategy() {
        let root = ActionNode {
            name: "root".to_string(),
            state_probabilities: Array::from_elem(6, 1. / 6.),
            total_probabilities: Array::zeros(3),
            evs: Array::zeros(6),
            infosets: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            strategy: Array::from_elem((2, 3), 1. / 2.),
            avg_strategy: Array::from_elem((2, 3), 1. / 2.),
            regrets: Array::from_elem((2, 3), 0.),
            children: vec![
                Box::new(TerminalNode {
                    name: "a".to_string(),
                    state_probabilities: Array::from_elem(3, 0.),
                    payouts: array![3., 2., 1.],
                }),
                Box::new(TerminalNode {
                    name: "b".to_string(),
                    state_probabilities: Array::from_elem(3, 0.),
                    payouts: array![1., 2., 3.],
                }),
            ],
            sign: 1,
            iter_count: 1,
        };

        assert_eq!(root.expand_strategy(), Array::from_elem((2, 6), 1. / 2.));
    }

    #[test]
    fn test_update_tree() {
        let mut root = ActionNode {
            name: "root".to_string(),
            state_probabilities: Array::from_elem(3, 1. / 3.),
            total_probabilities: Array::zeros(3),
            evs: Array::zeros(3),
            infosets: vec![vec![0], vec![1], vec![2]],
            strategy: Array::from_elem((3, 3), 1. / 3.),
            avg_strategy: Array::from_elem((3, 3), 1. / 3.),
            regrets: Array::from_elem((3, 3), 0.),
            children: vec![
                Box::new(TerminalNode {
                    name: "a".to_string(),
                    state_probabilities: Array::from_elem(3, 0.),
                    payouts: array![3., 2., 3.],
                }),
                Box::new(TerminalNode {
                    name: "b".to_string(),
                    state_probabilities: Array::from_elem(3, 0.),
                    payouts: array![1., 2.5, 2.],
                }),
                Box::new(TerminalNode {
                    name: "c".to_string(),
                    state_probabilities: Array::from_elem(3, 0.),
                    payouts: array![4., 2., 2.],
                }),
            ],
            sign: 1,
            iter_count: 1,
        };
        println!("{}", root);

        for _ in 0..1 {
            // Run one iteration of CFR
            root.update_probabilities();
            root.update_ev();
            root.update_strategy();
            root.update_probabilities();
        }

        println!("{}", root);

        println!("{}", root.infoset_probabilities(&root.state_probabilities));
        println!("{}", root.infoset_evs(&root.evs, &root.state_probabilities));
        println!("{}", root.action_evs());
        println!("{}", root.current_regret());
    }
}
