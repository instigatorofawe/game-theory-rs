use clap::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, required=true, num_args=1.., help = "Stack sizes")]
    stacks: Vec<f64>,

    #[arg(short, required=true, num_args=1.., help = "Payouts")]
    payouts: Vec<f64>,
}

pub fn main() {
    let args = Args::parse();
    let n_players = args.stacks.len();
    let n_places = args.payouts.len();

    let mut place_probabilities = vec![vec![0.0; n_places]; n_players];
    traverse(&args.stacks, 0, n_places, &mut place_probabilities, 1.0);

    let icm: Vec<f64> = place_probabilities
        .iter()
        .map(|p| {
            p.into_iter()
                .zip(args.payouts.iter())
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();

    println!("Place probabilities:");
    place_probabilities.iter().for_each(|x| println!("{:?}", x));
    println!("ICM:");
    println!("{:?}", icm);
}

fn traverse(
    stacks: &Vec<f64>,
    place: usize,
    n_places: usize,
    place_probabilities: &mut Vec<Vec<f64>>,
    p: f64,
) {
    let total_chips: f64 = stacks.iter().sum();
    let players: Vec<usize> = stacks
        .iter()
        .enumerate()
        .filter(|(i, x)| **x > 0.0)
        .map(|(i, x)| i)
        .collect();
    let current_probabilties: Vec<f64> = stacks
        .iter()
        .filter(|x| **x > 0.0)
        .map(|x| p * x / total_chips)
        .collect();
    players
        .iter()
        .zip(current_probabilties.iter())
        .for_each(|(i, prob)| {
            place_probabilities[*i][place] += prob;
        });
    if place + 1 < n_places {
        let new_stacks: Vec<Vec<f64>> = players
            .iter()
            .map(|i| {
                let mut result = stacks.clone();
                result[*i] = 0.0;
                result
            })
            .collect();
        new_stacks
            .iter()
            .zip(current_probabilties.iter())
            .for_each(|(s, prob)| {
                traverse(s, place + 1, n_places, place_probabilities, *prob);
            });
    }
}
