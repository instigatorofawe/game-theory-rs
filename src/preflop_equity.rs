mod utils;

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;

use rust_poker::equity_calculator::*;
use rust_poker::hand_range::*;

use utils::enumerate_combos;

const RANKS: &[char; 13] = &[
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
];

static SUITS: &[char; 4] = &['s', 'h', 'd', 'c'];

static CARDS: &[&str; 52] = &[
    "2s", "2h", "2d", "2c", "3s", "3h", "3d", "3c", "4s", "4h", "4d", "4c", "5s", "5h", "5d", "5c",
    "6s", "6h", "6d", "6c", "7s", "7h", "7d", "7c", "8s", "8h", "8d", "8c", "9s", "9h", "9d", "9c",
    "Ts", "Th", "Td", "Tc", "Js", "Jh", "Jd", "Jc", "Qs", "Qh", "Qd", "Qc", "Ks", "Kh", "Kd", "Kc",
    "As", "Ah", "Ad", "Ac",
];

static HANDS: &[&str; 169] = &[
    "22", "32s", "42s", "52s", "62s", "72s", "82s", "92s", "T2s", "J2s", "Q2s", "K2s", "A2s",
    "32o", "33", "43s", "53s", "63s", "73s", "83s", "93s", "T3s", "J3s", "Q3s", "K3s", "A3s",
    "42o", "43o", "44", "54s", "64s", "74s", "84s", "94s", "T4s", "J4s", "Q4s", "K4s", "A4s",
    "52o", "53o", "54o", "55", "65s", "75s", "85s", "95s", "T5s", "J5s", "Q5s", "K5s", "A5s",
    "62o", "63o", "64o", "65o", "66", "76s", "86s", "96s", "T6s", "J6s", "Q6s", "K6s", "A6s",
    "72o", "73o", "74o", "75o", "76o", "77", "87s", "97s", "T7s", "J7s", "Q7s", "K7s", "A7s",
    "82o", "83o", "84o", "85o", "86o", "87o", "88", "98s", "T8s", "J8s", "Q8s", "K8s", "A8s",
    "92o", "93o", "94o", "95o", "96o", "97o", "98o", "99", "T9s", "J9s", "Q9s", "K9s", "A9s",
    "T2o", "T3o", "T4o", "T5o", "T6o", "T7o", "T8o", "T9o", "TT", "JTs", "QTs", "KTs", "ATs",
    "J2o", "J3o", "J4o", "J5o", "J6o", "J7o", "J8o", "J9o", "JTo", "JJ", "QJs", "KJs", "AJs",
    "Q2o", "Q3o", "Q4o", "Q5o", "Q6o", "Q7o", "Q8o", "Q9o", "QTo", "QJo", "QQ", "KQs", "AQs",
    "K2o", "K3o", "K4o", "K5o", "K6o", "K7o", "K8o", "K9o", "KTo", "KJo", "KQo", "KK", "AKs",
    "A2o", "A3o", "A4o", "A5o", "A6o", "A7o", "A8o", "A9o", "ATo", "AJo", "AQo", "AKo", "AA",
];

fn card_from_str(card: &str) -> Option<u8> {
    static CARD_MAP: OnceLock<HashMap<&str, u8>> = OnceLock::new();
    let card_map = CARD_MAP.get_or_init(|| {
        let mut result = HashMap::new();
        CARDS.iter().enumerate().for_each(|(i, x)| {
            result.insert(*x, i as u8);
        });
        result
    });
    card_map.get(card).copied()
}

fn preflop_hand_from_cards(a: u8, b: u8) -> u8 {
    let rank_a = a / 4;
    let suit_a = a % 4;
    let rank_b = b / 4;
    let suit_b = b % 4;

    let max_rank = rank_a.max(rank_b);
    let min_rank = rank_a.min(rank_b);

    if rank_a == rank_b || suit_a == suit_b {
        return min_rank * 13 + max_rank;
    } else {
        return max_rank * 13 + min_rank;
    }
}

fn preflop_hand_from_str(hand: &str) -> Option<u8> {
    static HAND_MAP: OnceLock<HashMap<&str, u8>> = OnceLock::new();
    let hand_map = HAND_MAP.get_or_init(|| {
        let mut result = HashMap::new();
        HANDS.iter().enumerate().for_each(|(i, x)| {
            result.insert(*x, i as u8);
        });
        result
    });
    hand_map.get(hand).copied()
}

fn build_matchup_equities() {
    let mut equities = [[0.; 169]; 169];

    for i in 0..169 as usize {
        equities[i][i] = 0.5;
    }

    for i in 0..168 as usize {
        for j in i + 1..169 as usize {
            let result = exact_equity(
                &HandRange::from_strings(vec![String::from(HANDS[i]), String::from(HANDS[j])]),
                get_card_mask(""),
                1,
            )
            .unwrap();

            equities[i][j] = result[0];
            equities[j][i] = result[1];
        }
    }

    let mut output_buffer: Vec<u8> = Vec::with_capacity(169 * 169 * 8);
    equities
        .as_flattened()
        .into_iter()
        .for_each(|x| output_buffer.append(&mut Vec::from(x.to_le_bytes())));

    let mut o = File::create("data/precomputed_equities.bin")
        .unwrap_or_else(|_| panic!("Could not create preflop equities"));
    o.write_all(&output_buffer)
        .unwrap_or_else(|_| panic!("Unable to write preflop equities"));
}

fn build_matchup_probabilities() {
    let mut matchups = [[0u64; 169]; 169];

    let combos = enumerate_combos((0..52 as u8).collect(), 4);

    combos.into_iter().for_each(|x| {
        // (0, 1) and (2, 3)
        let hand_1 = preflop_hand_from_cards(x[0], x[1]);
        let hand_2 = preflop_hand_from_cards(x[2], x[3]);

        matchups[hand_1 as usize][hand_2 as usize] += 1;
        matchups[hand_2 as usize][hand_1 as usize] += 1;

        // (0, 2) and (1, 3)
        let hand_1 = preflop_hand_from_cards(x[0], x[2]);
        let hand_2 = preflop_hand_from_cards(x[1], x[3]);

        matchups[hand_1 as usize][hand_2 as usize] += 1;
        matchups[hand_2 as usize][hand_1 as usize] += 1;

        // (0, 3) and (1, 2)
        let hand_1 = preflop_hand_from_cards(x[0], x[3]);
        let hand_2 = preflop_hand_from_cards(x[1], x[2]);

        matchups[hand_1 as usize][hand_2 as usize] += 1;
        matchups[hand_2 as usize][hand_1 as usize] += 1;
    });

    let mut output_buffer: Vec<u8> = Vec::with_capacity(169 * 169 * 8);
    matchups
        .as_flattened()
        .into_iter()
        .for_each(|x| output_buffer.append(&mut Vec::from(x.to_le_bytes())));

    let mut o = File::create("data/precomputed_matchups.bin")
        .unwrap_or_else(|_| panic!("Could not create preflop equities"));
    o.write_all(&output_buffer)
        .unwrap_or_else(|_| panic!("Unable to write preflop equities"));
}

fn main() {
    build_matchup_equities();
    build_matchup_probabilities();
}

#[cfg(test)]
mod tests {}
