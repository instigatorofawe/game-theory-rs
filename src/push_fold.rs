mod utils;

use rayon::prelude::*;
use rust_poker::equity_calculator::*;
use rust_poker::hand_range::*;
use utils::enumerate_combos;

fn main() {
    const CARDS: [&str; 52] = [
        "2c", "2d", "2h", "2s", "3c", "3d", "3h", "3s", "4c", "4d", "4h", "4s", "5c", "5d", "5h",
        "5s", "6c", "6d", "6h", "6s", "7c", "7d", "7h", "7s", "8c", "8d", "8h", "8s", "9c", "9d",
        "9h", "9s", "Tc", "Td", "Th", "Ts", "Jc", "Jd", "Jh", "Js", "Qc", "Qd", "Qh", "Qs", "Kc",
        "Kd", "Kh", "Ks", "Ac", "Ad", "Ah", "As",
    ];

    let choose_four: Vec<Vec<usize>> = enumerate_combos((0..52).collect::<Vec<usize>>(), 4);
    let empty = get_card_mask("");

    let equities: Vec<f64> = choose_four
        .into_par_iter()
        .flat_map(|x| {
            let mut equities: Vec<f64> = Vec::with_capacity(6);

            let mut hand_1 = String::with_capacity(4);
            hand_1.push_str(CARDS[x[0]]);
            hand_1.push_str(CARDS[x[1]]);

            let mut hand_2 = String::with_capacity(4);
            hand_2.push_str(CARDS[x[2]]);
            hand_2.push_str(CARDS[x[3]]);

            equities.push(
                exact_equity(&HandRange::from_strings(vec![hand_1, hand_2]), empty, 1).unwrap()[0],
            );

            let mut hand_1 = String::with_capacity(4);
            hand_1.push_str(CARDS[x[0]]);
            hand_1.push_str(CARDS[x[2]]);

            let mut hand_2 = String::with_capacity(4);
            hand_2.push_str(CARDS[x[1]]);
            hand_2.push_str(CARDS[x[3]]);

            equities.push(
                exact_equity(&HandRange::from_strings(vec![hand_1, hand_2]), empty, 1).unwrap()[0],
            );

            let mut hand_1 = String::with_capacity(4);
            hand_1.push_str(CARDS[x[0]]);
            hand_1.push_str(CARDS[x[3]]);

            let mut hand_2 = String::with_capacity(4);
            hand_2.push_str(CARDS[x[1]]);
            hand_2.push_str(CARDS[x[2]]);

            equities.push(
                exact_equity(&HandRange::from_strings(vec![hand_1, hand_2]), empty, 1).unwrap()[0],
            );
            equities
        })
        .collect();
}
