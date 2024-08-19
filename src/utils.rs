use rayon::prelude::*;

pub fn enumerate_combos<T: Clone + Send + Sync>(items: Vec<T>, k: usize) -> Vec<Vec<T>> {
    // Base case: k = 1 or k == length of items
    if k == 1 {
        return items.into_iter().map(|x| vec![x]).collect();
    }

    if k == items.len() {
        return vec![items];
    }
    // For 0..n-k, choose prefix, append to enumerated combos of (n-k .. n) choose k-1
    // For each prefix, concatenate prefix to remainder
    items[0..items.len() - k + 1]
        .into_par_iter()
        .enumerate()
        .flat_map(|(i, x)| {
            let remainder = enumerate_combos(Vec::from(&items[i + 1..items.len()]), k - 1);
            remainder
                .into_iter()
                .map(|mut y| {
                    let mut result: Vec<T> = Vec::with_capacity(y.len() + 1);
                    result.push(x.clone());
                    result.append(&mut y);
                    result
                })
                .collect::<Vec<Vec<T>>>()
        })
        .collect::<Vec<Vec<T>>>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enumerate_combos() {
        let x = enumerate_combos(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            2,
        );
        assert_eq!(x.len(), 120);

        let x = enumerate_combos(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            3,
        );
        assert_eq!(x.len(), 560);
    }
}
