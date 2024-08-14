use hashbrown::HashMap;
use rayon::prelude::*;
use std::fmt::Display;
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq)]
pub enum Tile {
    Empty,
    X,
    O,
}

impl Display for Tile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.string())
    }
}

impl Tile {
    pub fn string(&self) -> String {
        use Tile::*;
        match self {
            Empty => " ".to_string(),
            X => "X".to_string(),
            O => "O".to_string(),
        }
    }

    pub fn hash(&self) -> u32 {
        use Tile::*;
        match self {
            Empty => 0,
            X => 1,
            O => 2,
        }
    }

    pub fn from_hash(hash: u32) -> Option<Self> {
        use Tile::*;
        match hash {
            0 => Some(Empty),
            1 => Some(X),
            2 => Some(O),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct Board {
    tiles: [Tile; 9],
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Tile::*;
        write!(
            f,
            "-------\n|{}|{}|{}|\n-------\n|{}|{}|{}|\n-------\n|{}|{}|{}|\n-------",
            match self.tiles[0] {
                Empty => "0".to_string(),
                _ => self.tiles[0].string(),
            },
            match self.tiles[1] {
                Empty => "1".to_string(),
                _ => self.tiles[1].string(),
            },
            match self.tiles[2] {
                Empty => "2".to_string(),
                _ => self.tiles[2].string(),
            },
            match self.tiles[3] {
                Empty => "3".to_string(),
                _ => self.tiles[3].string(),
            },
            match self.tiles[4] {
                Empty => "4".to_string(),
                _ => self.tiles[4].string(),
            },
            match self.tiles[5] {
                Empty => "5".to_string(),
                _ => self.tiles[5].string(),
            },
            match self.tiles[6] {
                Empty => "6".to_string(),
                _ => self.tiles[6].string(),
            },
            match self.tiles[7] {
                Empty => "7".to_string(),
                _ => self.tiles[7].string(),
            },
            match self.tiles[8] {
                Empty => "8".to_string(),
                _ => self.tiles[8].string(),
            },
        )
    }
}

impl Default for Board {
    fn default() -> Self {
        use Tile::Empty;
        Board {
            tiles: [
                Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
            ],
        }
    }
}

impl Board {
    /// Computes board from ternary hash
    pub fn from_hash(hash: u32) -> Self {
        let tiles: [Tile; 9] = [0; 9]
            .into_iter()
            .scan(hash, |a, _| {
                let result = Tile::from_hash(*a % 3);
                *a /= 3;
                result
            })
            .collect::<Vec<Tile>>()
            .try_into()
            .unwrap();
        Board { tiles }
    }

    /// Computes rotation invariant hash
    pub fn invariant_hash(&self) -> u32 {
        let mut hashes = Vec::with_capacity(8);
        let mut board: Board = self.clone();
        let mut rotated_hashes: Vec<u32> = [0; 3]
            .into_iter()
            .map(|_| {
                board = board.rotate();
                board.hash()
            })
            .collect();
        hashes.push(self.hash());
        hashes.append(&mut rotated_hashes);

        // Reflect vertically, horizontally, and diagonally
        hashes.push(
            Board {
                tiles: [6, 7, 8, 3, 4, 5, 0, 1, 2]
                    .into_iter()
                    .map(|i| self.tiles[i].clone())
                    .collect::<Vec<Tile>>()
                    .try_into()
                    .unwrap(),
            }
            .hash(),
        );
        hashes.push(
            Board {
                tiles: [2, 1, 0, 5, 4, 3, 8, 7, 6]
                    .into_iter()
                    .map(|i| self.tiles[i].clone())
                    .collect::<Vec<Tile>>()
                    .try_into()
                    .unwrap(),
            }
            .hash(),
        );
        hashes.push(
            Board {
                tiles: [8, 5, 2, 7, 4, 1, 6, 3, 0]
                    .into_iter()
                    .map(|i| self.tiles[i].clone())
                    .collect::<Vec<Tile>>()
                    .try_into()
                    .unwrap(),
            }
            .hash(),
        );
        hashes.push(
            Board {
                tiles: [0, 3, 6, 1, 4, 7, 2, 5, 8]
                    .into_iter()
                    .map(|i| self.tiles[i].clone())
                    .collect::<Vec<Tile>>()
                    .try_into()
                    .unwrap(),
            }
            .hash(),
        );

        hashes.into_iter().min().unwrap()
    }

    /// Computes naive ternary hash of board
    pub fn hash(&self) -> u32 {
        self.tiles
            .iter()
            .enumerate()
            .map(|(index, tile)| 3_u32.pow(index as u32) * tile.hash())
            .sum()
    }

    pub fn get_tiles(&self, indices: Vec<usize>) -> Vec<Tile> {
        indices.into_iter().map(|i| self.tiles[i].clone()).collect()
    }

    pub fn rotate(&self) -> Self {
        Board {
            tiles: [2, 5, 8, 1, 4, 7, 0, 3, 6]
                .into_iter()
                .map(|i| self.tiles[i].clone())
                .collect::<Vec<Tile>>()
                .try_into()
                .unwrap(),
        }
    }

    pub fn empty(&self) -> Vec<usize> {
        use Tile::Empty;
        let result: Vec<usize> = self
            .tiles
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                Empty => Some(i),
                _ => None,
            })
            .collect();
        result
    }

    pub fn turn(&self) -> Tile {
        use Tile::*;
        match self.empty().len() % 2 {
            1 => X,
            _ => O,
        }
    }

    pub fn act(&self, index: usize) -> Self {
        let mut tiles = self.tiles.clone();
        tiles[index] = self.turn();
        Board { tiles }
    }

    pub fn winner(&self) -> Tile {
        use Tile::*;
        if self.empty().len() > 5 {
            return Empty;
        }

        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        .map(|x| {
            match [
                self.tiles[x[0]].clone(),
                self.tiles[x[1]].clone(),
                self.tiles[x[2]].clone(),
            ] {
                [X, X, X] => X,
                [O, O, O] => O,
                _ => Empty,
            }
        })
        .into_iter()
        .fold(Empty, |value, x| match x {
            Empty => value,
            _ => x,
        })
    }
}

pub struct SolutionTable {
    value_table: Mutex<HashMap<u32, i8>>,
}

impl Default for SolutionTable {
    fn default() -> Self {
        Self {
            value_table: Mutex::new(HashMap::new()),
        }
    }
}

impl SolutionTable {
    /// Builds full solution table
    pub fn build() -> Self {
        let result = SolutionTable {
            value_table: Mutex::new(HashMap::new()),
        };
        let root = Board::default();
        result.evaluate_recursive(root);

        #[cfg(test)]
        println!("{}", result.value_table.lock().unwrap().len());

        result
    }

    fn evaluate_recursive(&self, board: Board) -> i8 {
        use Tile::*;
        let hash = board.invariant_hash();
        let value: Option<i8>;
        {
            let lock = self.value_table.lock().unwrap();
            value = lock.get(&hash).copied();
        }

        match value {
            Some(x) => x,
            None => {
                // Check if leaf node
                match board.winner() {
                    X => {
                        let value = 1;
                        self.value_table.lock().unwrap().insert(hash, value);
                        return value;
                    }
                    O => {
                        let value = -1;
                        self.value_table.lock().unwrap().insert(hash, value);
                        return value;
                    }
                    _ => {
                        if board.empty().is_empty() {
                            let value = 0;
                            self.value_table.lock().unwrap().insert(hash, value);
                            return value;
                        }
                    }
                }

                // Otherwise evaluate children recursively
                let children: Vec<Board> =
                    board.empty().into_iter().map(|i| board.act(i)).collect();
                let child_values: Vec<i8> = children
                    .into_par_iter()
                    .map(|x| self.evaluate_recursive(x))
                    .collect();

                let value = match board.turn() {
                    X => child_values.into_iter().max().unwrap(),
                    O => child_values.into_iter().min().unwrap(),
                    _ => panic!("Impossible branch, invalid turn"),
                };

                self.value_table.lock().unwrap().insert(hash, value);
                value
            }
        }
    }

    pub fn get(&self, invariant_hash: &u32) -> Option<i8> {
        self.value_table
            .lock()
            .unwrap()
            .get(invariant_hash)
            .copied()
    }

    pub fn solve(&mut self, board: &Board) -> usize {
        use Tile::*;
        let candidates = board.empty();
        let candidate_values: Vec<i8> = candidates
            .par_iter()
            .map(|i| self.evaluate_recursive(board.act(*i)))
            .collect();

        let (argmax, _) = candidates.into_iter().zip(candidate_values).fold(
            (0, 2),
            |(argmax, max), (index, value)| {
                if (max > value && board.turn() == X) || (max < value && board.turn() == O) {
                    (argmax, max)
                } else {
                    (index, value)
                }
            },
        );
        argmax
    }
}

fn main() {
    use std::io::stdin;
    use Tile::*;

    let mut board = Board::default();
    let mut solution = SolutionTable::default();

    println!("{board}");

    while board.winner() == Empty && !board.empty().is_empty() {
        if board.turn() == X {
            let mut input_buffer = String::new();
            let _ = stdin().read_line(&mut input_buffer);
            let i = input_buffer.trim().parse::<usize>().unwrap();
            board = board.act(i);
            // Read input
        } else {
            let argmin = solution.solve(&board);
            board = board.act(argmin);
        }

        println!("{board}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash() {
        use Tile::*;
        let board = Board {
            tiles: [X, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        };
        assert_eq!(board.hash(), 1);
        let x = Board::from_hash(1);
        assert_eq!(x, board);
    }

    #[test]
    fn test_invariant_hash() {
        use Tile::*;
        let board = Board {
            tiles: [X, O, Empty, Empty, X, Empty, Empty, Empty, Empty],
        };

        println!("{}", board);
        let mut rotated_board = board.rotate();

        for _ in 0..3 {
            assert_eq!(board.invariant_hash(), rotated_board.invariant_hash());
            println!("{}", rotated_board);
            rotated_board = rotated_board.rotate();
        }
    }

    #[test]
    fn test_empty() {
        use Tile::*;
        let board = Board {
            tiles: [X, O, Empty, Empty, X, Empty, Empty, Empty, Empty],
        };

        assert_eq!(board.empty(), vec![2, 3, 5, 6, 7, 8]);
    }

    #[test]
    fn test_evaluate_winner() {
        use Tile::*;
        let board = Board::default();
        assert_eq!(board.winner(), Empty);

        let board = Board {
            tiles: [X, X, X, O, O, Empty, Empty, Empty, Empty],
        };
        assert_eq!(board.winner(), X);

        let board = Board {
            tiles: [Empty, Empty, Empty, X, X, X, O, O, Empty],
        };
        assert_eq!(board.winner(), X);

        let board = Board {
            tiles: [X, Empty, Empty, Empty, X, X, O, O, O],
        };
        assert_eq!(board.winner(), O);
    }

    #[test]
    fn test_build_tree() {
        use Tile::*;
        let tree = SolutionTable::build();
        assert_eq!(*tree.value_table.lock().unwrap().get(&0).unwrap(), 0);
        assert_eq!(
            *tree
                .value_table
                .lock()
                .unwrap()
                .get(
                    &Board {
                        tiles: [X, X, X, O, O, Empty, Empty, Empty, Empty]
                    }
                    .invariant_hash()
                )
                .unwrap(),
            1
        );

        assert_eq!(
            *tree
                .value_table
                .lock()
                .unwrap()
                .get(
                    &Board {
                        tiles: [X, Empty, X, O, O, Empty, Empty, Empty, Empty]
                    }
                    .invariant_hash()
                )
                .unwrap(),
            1
        );
    }
}
