use ndarray::Array;
use rayon::prelude::*;
use std::fmt::Debug;

pub trait Node: Debug + Sync + Send {
    fn name(&self) -> String;
    fn state_probabilities(&self) -> Vec<f64>;
    fn payouts(&self) -> Vec<f64>;
    fn update_probabilities(&mut self, new_state_probs: Vec<f64>);
    fn update_child_probabilities(&mut self);
    fn update_ev(&mut self);
    fn update_regret(&mut self);
    fn update_strategy(&mut self);
}

#[derive(Debug)]
struct ActionNode {
    name: String,
    state_probabilities: Vec<f64>,
    evs: Vec<f64>,
    information_sets: Vec<Vec<usize>>, // Indexed by [information_set, member]
    strategy: Vec<Vec<f64>>,           // Indexed by [action, information_set]
    average_strategy: Vec<Vec<f64>>,   // Indexed by [action, information set]
    regrets: Vec<Vec<f64>>,            //Indexed by [action, information set]
    total_probabilities: Vec<f64>,     // Indexed by [information set]
    children: Vec<Box<dyn Node>>,
}

impl ActionNode {
    fn expand_strategy(&self) -> Vec<Vec<f64>> {
        let n = self.state_probabilities.len();
        let result = self
            .strategy
            .iter()
            .map(|action| {
                let mut output = vec![0.; n];

                self.information_sets
                    .iter()
                    .enumerate()
                    .map(|(index, infoset)| {
                        for j in infoset {
                            output[*j] = action[index];
                        }
                    })
                    .for_each(drop);

                output
            })
            .collect();

        result
    }

    fn infoset_probabilities(&self) -> Vec<f64> {
        self.information_sets
            .iter()
            .map(|infoset| infoset.iter().map(|i| self.state_probabilities[*i]).sum())
            .collect()
    }

    fn regret_match(&self) -> Vec<Vec<f64>> {
        todo!()
    }
}

impl Node for ActionNode {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn state_probabilities(&self) -> Vec<f64> {
        self.state_probabilities.clone()
    }

    fn payouts(&self) -> Vec<f64> {
        self.evs.clone()
    }

    fn update_probabilities(&mut self, new_state_probs: Vec<f64>) {
        self.state_probabilities = new_state_probs
    }

    fn update_child_probabilities(&mut self) {
        let expanded_strategy = self.expand_strategy();
        self.children
            .par_iter_mut()
            .enumerate()
            .map(|(i, x)| {
                x.update_probabilities(
                    self.state_probabilities
                        .iter()
                        .zip(expanded_strategy[i].iter())
                        .map(|(x, y)| x * y)
                        .collect(),
                );
                x.update_child_probabilities();
            })
            .for_each(drop);
    }

    fn update_ev(&mut self) {
        // Updates EV, regret, and strategies
        self.children
            .par_iter_mut()
            .map(|x| x.update_ev())
            .for_each(drop);

        let n = self.state_probabilities.len();

        self.evs = self
            .children
            .iter()
            .map(|child| {
                child
                    .state_probabilities()
                    .into_iter()
                    .zip(child.payouts().into_iter())
                    .map(|(a, b)| a * b)
                    .collect::<Vec<f64>>()
            })
            .fold(vec![0.; n], |x, y| {
                x.into_iter()
                    .zip(y.into_iter())
                    .map(|(a, b)| a + b)
                    .collect()
            })
            .into_iter()
            .zip(self.state_probabilities.iter())
            .map(|(a, b)| a / b.max(0.))
            .collect();
    }

    fn update_regret(&mut self) {
        self.children
            .par_iter_mut()
            .map(|x| x.update_regret())
            .for_each(drop);
    }

    fn update_strategy(&mut self) {
        self.children
            .par_iter_mut()
            .map(|x| x.update_strategy())
            .for_each(drop);
    }
}

#[derive(Debug)]
struct TerminalNode {
    name: String,
    payouts: Vec<f64>, // Indexed by [state]
    state_probabilities: Vec<f64>,
}

impl Node for TerminalNode {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn state_probabilities(&self) -> Vec<f64> {
        self.state_probabilities.clone()
    }

    fn payouts(&self) -> Vec<f64> {
        self.payouts.clone()
    }

    fn update_probabilities(&mut self, new_state_probs: Vec<f64>) {
        self.state_probabilities = new_state_probs
    }

    fn update_child_probabilities(&mut self) {
        // Terminal node has no children, so this does nothing
    }

    fn update_ev(&mut self) {
        // EV is fixed for terminal nodes, so this does nothing
    }

    fn update_regret(&mut self) {
        // No regret as we are already at a terminal node
    }

    fn update_strategy(&mut self) {
        // Terminal nodes don't have a strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_strategy() {
        let root = ActionNode {
            name: "root".to_string(),
            children: Vec::new(),
            state_probabilities: vec![1. / 6.; 6],
            evs: vec![0.; 6],
            information_sets: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
            strategy: vec![vec![1. / 2.; 3], vec![1. / 2.; 3]],
            average_strategy: vec![vec![1. / 2.; 3], vec![1. / 2.; 3]],
            total_probabilities: vec![1. / 3.; 3],
            regrets: Vec::new(),
        };

        assert_eq!(
            root.expand_strategy(),
            vec![vec![1. / 2.; 6], vec![1. / 2.; 6]]
        );
        assert_eq!(root.infoset_probabilities(), vec![1. / 3.; 3]);
    }

    #[test]
    fn test_update_tree() {
        let mut root = ActionNode {
            name: "root".to_string(),
            children: vec![
                Box::new(TerminalNode {
                    name: "a".to_string(),
                    payouts: vec![1.0, 2.0, 3.0],
                    state_probabilities: vec![0., 0., 0.],
                }),
                Box::new(TerminalNode {
                    name: "b".to_string(),
                    payouts: vec![-3.0, -2.0, -1.0],
                    state_probabilities: vec![0., 0., 0.],
                }),
            ],
            state_probabilities: vec![1. / 3., 1. / 3., 1. / 3.],
            evs: vec![0., 0., 0.],
            information_sets: vec![vec![0], vec![1], vec![2]],
            strategy: vec![
                vec![1. / 2., 1. / 2., 1. / 2.],
                vec![1. / 2., 1. / 2., 1. / 2.],
            ],
            average_strategy: vec![
                vec![1. / 2., 1. / 2., 1. / 2.],
                vec![1. / 2., 1. / 2., 1. / 2.],
            ],
            total_probabilities: vec![1. / 3.; 3],
            regrets: vec![vec![0.; 3]; 2],
        };

        println!("{:?}", root);

        // Run one iteration of CFR
        root.update_child_probabilities();
        root.update_ev();
        root.update_regret();
        root.update_strategy();

        println!("{:?}", root);
    }
}
