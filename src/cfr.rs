use ndarray::*;
use rayon::prelude::*;
use std::fmt::{Debug, Display};

trait Node: Debug + Sync + Send + Display {
    fn name(&self) -> String;
    fn state_probabilities(&self) -> Array<f64, Ix1>;
    fn payouts(&self) -> Array<f64, Ix1>;

    fn set_state_probabilities(&mut self, p: Array<f64, Ix1>);
    fn update_probabilities(&mut self);
    fn update_ev(&mut self);
    fn update_strategy(&mut self);
}

#[derive(Debug)]
struct ActionNode {
    name: String,
    state_probabilities: Array<f64, Ix1>, // Indexed by state
    total_probabilities: Array<f64, Ix1>, // Indexed by infoset
    evs: Array<f64, Ix1>,                 // Indexed by state
    infosets: Vec<Vec<usize>>,            // Indexed by infoset, member(state)
    strategy: Array<f64, Ix2>,            // Indexed by action, infoset
    avg_strategy: Array<f64, Ix2>,        // Indexed by action, infoset
    regrets: Array<f64, Ix2>,             // Indexed by action, infoset
    children: Vec<Box<dyn Node>>,
    sign: i8,        // 1 for positive payout, -1 for negative payout
    iter_count: u64, // CFR iteration count
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
                    .into_iter()
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
            .map(|x| x.into_iter().map(|i| state_probabilities[*i]).sum())
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
                x.into_iter()
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
        todo!()
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
        if self.total_probabilities == Array::zeros(self.total_probabilities.len()) {
            self.total_probabilities = self.state_probabilities.clone()
        }

        self.children
            .par_iter_mut()
            .enumerate()
            .map(|(action_index, child)| {
                child.set_state_probabilities(
                    self.state_probabilities.clone() * self.strategy.slice(s![action_index, ..]),
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
struct TerminalNode {
    name: String,
    state_probabilities: Array<f64, Ix1>,
    payouts: Array<f64, Ix1>,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_strategy() {
        let root = ActionNode {
            name: "root".to_string(),
            state_probabilities: Array::from_elem(6, 1. / 3.),
            total_probabilities: Array::zeros(6),
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
        println!("{}", root);

        // Run one iteration of CFR
        root.update_probabilities();
        root.update_ev();
        root.update_strategy();

        println!("{}", root);

        println!("{}", root.infoset_probabilities(&root.state_probabilities));
        println!("{}", root.infoset_evs(&root.evs, &root.state_probabilities));
        println!("{}", root.action_evs());
        println!("{}", root.current_regret());
    }
}
