mod cfr;

use cfr::*;
use ndarray::*;

fn build_kuhn_tree() -> Box<dyn Node> {
    Box::new(ActionNode {
        name: "root".to_string(),
        state_probabilities: Array::from_elem(6, 1. / 6.), // KQ KJ QK QJ JK JQ
        total_probabilities: Array::zeros(3),
        evs: Array::zeros(6),
        infosets: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
        strategy: Array::from_elem((2, 3), 1. / 2.),
        avg_strategy: Array::from_elem((2, 3), 1. / 2.),
        regrets: Array::zeros((2, 3)),
        sign: 1,
        iter_count: 1,
        children: vec![
            Box::new(ActionNode {
                name: "b".to_string(),
                state_probabilities: Array::zeros(6),
                total_probabilities: Array::zeros(3),
                evs: Array::zeros(6),
                infosets: vec![vec![2, 4], vec![0, 5], vec![1, 3]],
                strategy: Array::from_elem((2, 3), 1. / 2.),
                avg_strategy: Array::from_elem((2, 3), 1. / 2.),
                regrets: Array::zeros((2, 3)),
                sign: -1,
                iter_count: 1,
                children: vec![
                    Box::new(TerminalNode {
                        name: "bc".to_string(),
                        state_probabilities: Array::zeros(6),
                        payouts: array![2., 2., -2., 2., -2., -2.],
                    }),
                    Box::new(TerminalNode {
                        name: "bf".to_string(),
                        state_probabilities: Array::zeros(6),
                        payouts: array![1., 1., 1., 1., 1., 1.],
                    }),
                ],
            }),
            Box::new(ActionNode {
                name: "x".to_string(),
                state_probabilities: Array::zeros(6),
                total_probabilities: Array::zeros(3),
                evs: Array::zeros(6),
                infosets: vec![vec![2, 4], vec![0, 5], vec![1, 3]],
                strategy: Array::from_elem((2, 3), 1. / 2.),
                avg_strategy: Array::from_elem((2, 3), 1. / 2.),
                regrets: Array::zeros((2, 3)),
                sign: -1,
                iter_count: 1,
                children: vec![
                    Box::new(ActionNode {
                        name: "xb".to_string(),
                        state_probabilities: Array::zeros(6),
                        total_probabilities: Array::zeros(3),
                        evs: Array::zeros(6),
                        infosets: vec![vec![0, 1], vec![2, 3], vec![4, 5]],
                        strategy: Array::from_elem((2, 3), 1. / 2.),
                        avg_strategy: Array::from_elem((2, 3), 1. / 2.),
                        regrets: Array::zeros((2, 3)),
                        sign: 1,
                        iter_count: 1,
                        children: vec![
                            Box::new(TerminalNode {
                                name: "bc".to_string(),
                                state_probabilities: Array::zeros(6),
                                payouts: array![2., 2., -2., 2., -2., -2.],
                            }),
                            Box::new(TerminalNode {
                                name: "bf".to_string(),
                                state_probabilities: Array::zeros(6),
                                payouts: array![-1., -1., -1., -1., -1., -1.],
                            }),
                        ],
                    }),
                    Box::new(TerminalNode {
                        name: "xx".to_string(),
                        state_probabilities: Array::zeros(6),
                        payouts: array![1., 1., -1., 1., -1., -1.],
                    }),
                ],
            }),
        ],
    })
}

fn main() {
    let mut root = build_kuhn_tree();

    for _ in 0..10000 {
        // Run one iteration of CFR
        root.update_probabilities();
        root.update_ev();
        root.update_strategy();
    }

    println!("{}", root);
}

#[cfg(test)]
mod tests {}
