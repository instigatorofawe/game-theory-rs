use rayon::prelude::*;
use std::fmt::Debug;

pub trait Node: Debug + Sync + Send {
    fn name(&self) -> String;
    fn state_probabilities(&self) -> Vec<f64>;
    fn payouts(&self) -> Vec<f64>;
    fn update_probabilities(&mut self, new_state_probs: Vec<f64>);
    fn update_child_probabilities(&mut self);
    fn update_ev(&mut self);
}

#[derive(Debug)]
struct ActionNode {
    name: String,
    state_probabilities: Vec<f64>,
    evs: Vec<f64>,
    strategy: Vec<Vec<f64>>, // Indexed by [action, state]
    children: Vec<Box<dyn Node>>,
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
        self.children
            // .iter_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(i, x)| {
                x.update_probabilities(
                    self.state_probabilities
                        .clone()
                        .into_iter()
                        .zip(self.strategy[i].clone().into_iter())
                        .map(|(x, y)| x * y)
                        .collect(),
                );
                x.update_child_probabilities();
            })
            .for_each(drop);
    }

    fn update_ev(&mut self) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
            strategy: vec![
                vec![1. / 2., 1. / 2., 1. / 2.],
                vec![1. / 2., 1. / 2., 1. / 2.],
            ],
        };

        println!("{:?}", root);

        root.update_child_probabilities();
        root.update_ev();

        println!("{:?}", root);
    }
}
