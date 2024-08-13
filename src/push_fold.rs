mod cfr;
mod utils;

use cfr::*;
use utils::enumerate_combos;

use std::fmt::Display;

use rust_poker::constants::RANK_TO_CHAR;
use rust_poker::equity_calculator::*;
use rust_poker::hand_range::*;

use clap::*;
use ndarray::*;
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(default_value = "10.0", help = "Stack size")]
    stack_size: f64,

    #[arg(default_value = "0.125", short, long, help = "Ante")]
    ante: f64,

    #[arg(default_value = "0.5", short, long, help = "Small blind")]
    sb: f64,
}

struct Hand(usize, usize);

impl Display for Hand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r1 = self.0 / 4;
        let r2 = self.1 / 4;
        let s1 = self.0 % 4;
        let s2 = self.1 % 4;

        if r1 == r2 {
            write!(f, "{}{}", RANK_TO_CHAR[r1], RANK_TO_CHAR[r1])
        } else if s1 == s2 {
            if r1 > r2 {
                write!(f, "{}{}s", RANK_TO_CHAR[r1], RANK_TO_CHAR[r2])
            } else {
                write!(f, "{}{}s", RANK_TO_CHAR[r2], RANK_TO_CHAR[r1])
            }
        } else if r1 > r2 {
            write!(f, "{}{}o", RANK_TO_CHAR[r1], RANK_TO_CHAR[r2])
        } else {
            write!(f, "{}{}o", RANK_TO_CHAR[r2], RANK_TO_CHAR[r1])
        }
    }
}

impl Hand {
    fn index_to_str(x: usize) -> String {
        let mut result = String::with_capacity(3);
        let i = x / 13;
        let j = x % 13;

        use std::cmp::Ordering::*;
        match i.cmp(&j) {
            Greater => {
                result.push(RANK_TO_CHAR[i]);
                result.push(RANK_TO_CHAR[j]);
                result.push('s');
            }
            Less => {
                result.push(RANK_TO_CHAR[j]);
                result.push(RANK_TO_CHAR[i]);
                result.push('o');
            }
            Equal => {
                result.push(RANK_TO_CHAR[i]);
                result.push(RANK_TO_CHAR[j]);
            }
        }

        result
    }
    fn get_index(c1: usize, c2: usize) -> usize {
        let r1 = c1 / 4;
        let s1 = c1 % 4;
        let r2 = c2 / 4;
        let s2 = c2 % 4;

        if (r1 == r2) || (s1 == s2) {
            // Diagonals for pocket pairs, upper triangle for suited combos
            13 * r1 + r2
        } else {
            // Lower triangle for offsuit combos
            13 * r2 + r1
        }
    }
}

#[derive(Debug)]
struct MatchupTable {
    counts: [[u64; 169]; 169],
}

impl MatchupTable {
    fn count_matchup(&mut self, hand_index_1: usize, hand_index_2: usize) {
        self.counts[hand_index_1][hand_index_2] += 1;
        self.counts[hand_index_2][hand_index_1] += 1;
    }

    fn sum(&self) -> u64 {
        self.counts.iter().fold(0, |f, x| f + x.iter().sum::<u64>())
    }
}

impl Default for MatchupTable {
    fn default() -> Self {
        MatchupTable {
            counts: [[0; 169]; 169],
        }
    }
}

fn build_push_fold_tree(stack_size: f64, ante: f64, sb: f64) -> Box<dyn Node> {
    let mut matchup_table = MatchupTable::default();
    enumerate_combos((0..52).collect::<Vec<usize>>(), 4)
        .into_iter()
        .map(|x| {
            matchup_table.count_matchup(Hand::get_index(x[0], x[1]), Hand::get_index(x[2], x[3]));
            matchup_table.count_matchup(Hand::get_index(x[0], x[2]), Hand::get_index(x[1], x[3]));
            matchup_table.count_matchup(Hand::get_index(x[0], x[3]), Hand::get_index(x[1], x[2]));
        })
        .for_each(drop);

    let equities: Vec<Vec<f64>> = (0_usize..169)
        .into_par_iter()
        .map(|i| {
            (i..169)
                .map(|j| {
                    exact_equity(
                        &HandRange::from_strings(vec![
                            Hand::index_to_str(i),
                            Hand::index_to_str(j),
                        ]),
                        get_card_mask(""),
                        1,
                    )
                    .unwrap()[0]
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    let mut equities_square: Array<f64, Ix2> = Array::zeros((169, 169));
    equities
        .into_iter()
        .enumerate()
        .map(|(i, x)| {
            equities_square
                .slice_mut(s![i, i..])
                .assign(&Array::from(x.clone()));
            equities_square
                .slice_mut(s![i.., i])
                .assign(&(1. - Array::from(x)));
        })
        .for_each(drop);

    let total_matchups = matchup_table.sum();
    let state_probabilities: Array<f64, Ix1> = matchup_table
        .counts
        .as_flattened()
        .iter()
        .map(|x| *x as f64 / total_matchups as f64)
        .collect();

    // Compute information sets
    let infosets_p1: Vec<Vec<usize>> = (0_usize..169)
        .map(|i| (0_usize..169).map(|j| i * 169 + j).collect::<Vec<usize>>())
        .collect();
    let infosets_p2: Vec<Vec<usize>> = (0_usize..169)
        .map(|i| (0_usize..169).map(|j| j * 169 + i).collect::<Vec<usize>>())
        .collect();

    (Box::new(ActionNode {
        name: "root".to_string(),
        state_probabilities,
        total_probabilities: Array::zeros(169),
        evs: Array::zeros(169 * 169),
        infosets: infosets_p1,
        strategy: Array::from_elem((2, 169), 1. / 2.),
        avg_strategy: Array::from_elem((2, 169), 1. / 2.),
        regrets: Array::zeros((2, 169)),
        iter_count: 1,
        sign: 1,
        children: vec![
            Box::new(ActionNode {
                name: "b".to_string(),
                state_probabilities: Array::zeros(169 * 169),
                total_probabilities: Array::zeros(169),
                evs: Array::zeros(169 * 169),
                infosets: infosets_p2,
                strategy: Array::from_elem((2, 169), 1. / 2.),
                avg_strategy: Array::from_elem((2, 169), 1. / 2.),
                regrets: Array::zeros((2, 169)),
                iter_count: 1,
                sign: -1,
                children: vec![
                    Box::new(TerminalNode {
                        name: "bc".to_string(),
                        state_probabilities: Array::zeros(169 * 169),
                        payouts: Array::from_elem(169 * 169, stack_size + ante)
                            * 2.
                            * (equities_square.flatten() - 0.5),
                    }),
                    Box::new(TerminalNode {
                        name: "bf".to_string(),
                        state_probabilities: Array::zeros(169 * 169),
                        payouts: Array::from_elem(169 * 169, 1. + ante),
                    }),
                ],
            }),
            Box::new(TerminalNode {
                name: "f".to_string(),
                state_probabilities: Array::zeros(169 * 169),
                payouts: Array::from_elem(169 * 169, -sb - ante),
            }),
        ],
    })) as _
}

fn main() {
    let args = Args::parse();
    let hand_names: Vec<String> = (0..169).map(Hand::index_to_str).collect();

    println!("Building tree...");
    let mut root = build_push_fold_tree(args.stack_size, args.ante, args.sb);

    for _ in 0..10000 {
        root.update_probabilities();
        root.update_ev();
        root.update_strategy();
    }

    hand_names
        .iter()
        .zip(root.avg_strategy().unwrap().slice(s![0, ..]))
        .map(|(name, strategy)| {
            print!("{}: {},", name, strategy);
        })
        .for_each(drop);
    println!();

    hand_names
        .iter()
        .zip(
            root.children().unwrap()[0]
                .avg_strategy()
                .unwrap()
                .slice(s![0, ..]),
        )
        .map(|(name, strategy)| {
            print!("{}: {},", name, strategy);
        })
        .for_each(drop);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten() {
        let a = array![[0, 1, 2], [3, 4, 5]];
        println!("{}", a.flatten());
    }

    #[test]
    fn test_index() {
        assert_eq!(Hand::get_index(1, 0), 0);
        assert_eq!(Hand::get_index(51, 50), 168);
    }

    #[test]
    fn test_display() {
        assert_eq!(Hand(1, 0).to_string(), "22");
        assert_eq!(Hand(48, 0).to_string(), "A2s");
    }

    #[test]
    fn test_from_index() {
        for i in 0..51 {
            for j in i + 1..52 {
                assert_eq!(
                    Hand(i, j).to_string(),
                    Hand::index_to_str(Hand::get_index(j, i))
                );
            }
        }
    }
}
