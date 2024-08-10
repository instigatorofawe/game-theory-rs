pub fn factorial(x: usize) -> usize {
    (1..x + 1).product()
}

pub fn choose(n: usize, k: usize) -> usize {
    if k <= n - k {
        (n - k + 1..n + 1).product::<usize>() / factorial(k)
    } else {
        choose(n, n - k)
    }
}

pub fn enumerate_combos<T: Clone>(items: Vec<T>, k: usize) -> Vec<Vec<T>> {
    // Base case: k = 1
    if k == 1 {
        return items
            .into_iter()
            .map(|x| {
                let mut vec: Vec<T> = Vec::with_capacity(1);
                vec.push(x);
                vec
            })
            .collect();
    }
    // For 0..n-k, choose prefix, append to enumerated combos of (n-k .. n) choose k-1
    let prefixes = Vec::from(&items[0..items.len() - k + 1]);

    // For each prefix, prefix to remainder
    // Return result
    let combos: Vec<Vec<T>> = prefixes
        .into_iter()
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
        .collect();

    combos
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
        assert_eq!(x.len(), choose(16, 2));

        let x = enumerate_combos(
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            3,
        );
        assert_eq!(x.len(), choose(16, 3));
    }
}
