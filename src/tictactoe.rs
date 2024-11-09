use hashbrown::HashMap;
use std::fmt::Display;

/// Tile on a tic-tac-toe board
#[derive(PartialEq, Clone, Debug)]
pub enum Tile {
    Empty,
    X,
    O,
}

impl Tile {
    /// Computes the numerical representation of the tile
    pub fn hash(&self) -> u16 {
        match self {
            Tile::Empty => 0,
            Tile::X => 1,
            Tile::O => 2,
        }
    }
}

impl Display for Tile {
    /// Prints the string representation of the tile
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty => write!(f, " "),
            Tile::X => write!(f, "X"),
            Tile::O => write!(f, "O"),
        }
    }
}

/// A tic-tac-toe board is simply 9 tiles
#[derive(Clone)]
pub struct Board {
    tiles: [Tile; 9],
}

impl Board {
    /// Computes the numerical representation of the current board state
    pub fn hash(&self) -> u16 {
        self.tiles.iter().fold(0, |i, x| i * 3 + x.hash())
    }

    /// Computes the rotation and reflection invariant representation of the current board state
    pub fn invariant_hash(&self) -> u16 {
        const TRANSFORMATIONS: [[usize; 9]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8], // Rotations
            [2, 5, 8, 1, 4, 7, 0, 3, 6],
            [8, 7, 6, 5, 4, 3, 2, 1, 0],
            [6, 3, 0, 7, 4, 1, 8, 5, 2],
            [6, 7, 8, 3, 4, 5, 0, 1, 2], // Reflections
            [2, 1, 0, 5, 4, 3, 8, 7, 6],
            [8, 5, 2, 7, 4, 1, 6, 3, 0],
            [0, 3, 6, 1, 4, 7, 2, 5, 8],
        ];

        TRANSFORMATIONS
            .iter()
            .map(|x| x.iter().fold(0, |i, x| i * 3 + self.tiles[*x].hash()))
            .min()
            .unwrap()
    }

    /// Computes list of empty indices
    pub fn empty(&self) -> Vec<usize> {
        self.tiles
            .iter()
            .enumerate()
            .filter_map(|(i, x)| match x {
                Tile::Empty => Some(i),
                _ => None,
            })
            .collect()
    }

    /// Computes winner if there is one
    pub fn winner(&self) -> Option<Tile> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ];

        for line in LINES {
            if self.tiles[line[0]] != Tile::Empty
                && self.tiles[line[0]] == self.tiles[line[1]]
                && self.tiles[line[1]] == self.tiles[line[2]]
            {
                return Some(self.tiles[line[0]].clone());
            }
        }

        return None;
    }

    /// Return player whose turn it is
    pub fn turn(&self) -> Tile {
        let n_empty: u8 = self
            .tiles
            .iter()
            .map(|x| match x {
                Tile::Empty => 1,
                _ => 0,
            })
            .sum();
        match n_empty % 2 {
            1 => Tile::X,
            _ => Tile::O,
        }
    }

    /// The player whose turn it is makes their move. The board is modified in-place.
    pub fn act(&mut self, index: usize) -> Result<(), String> {
        if index < self.tiles.len() && self.tiles[index] == Tile::Empty {
            self.tiles[index] = self.turn();
            Ok(())
        } else {
            Err(String::from("Invalid move"))
        }
    }
}

impl Default for Board {
    /// Default board is empty tiles
    fn default() -> Board {
        use Tile::Empty;
        return Board {
            tiles: [
                Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
            ],
        };
    }
}

impl Display for Board {
    /// Display board state
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n",
            self.tiles[0],
            self.tiles[1],
            self.tiles[2],
            self.tiles[3],
            self.tiles[4],
            self.tiles[5],
            self.tiles[6],
            self.tiles[7],
            self.tiles[8]
        )
    }
}

/// Minimax solution table
pub struct SolutionTable {
    value_table: HashMap<u16, i8>,
}

impl SolutionTable {
    /// Returns the minimax solution for the current board state, for the player whose turn it is
    fn solve(&mut self, board: &Board) -> usize {
        use Tile::*;
        let empty = board.empty();
        let values: Vec<i8> = empty
            .iter()
            .map(|i| {
                let mut new_board = (*board).clone();
                let _ = new_board.act(*i);
                self.eval_recursive(&new_board)
            })
            .collect();
        match board.turn() {
            X => {
                // Argmax
                let (argmax, _) = empty.into_iter().zip(values.into_iter()).fold(
                    (0 as usize, i8::MIN),
                    |(argmax, max), (index, value)| match max > value {
                        true => (argmax, max),
                        false => (index, value),
                    },
                );
                argmax
            }
            O => {
                // Argmin
                let (argmin, _) = empty.into_iter().zip(values.into_iter()).fold(
                    (0 as usize, i8::MAX),
                    |(argmin, min), (index, value)| match min < value {
                        true => (argmin, min),
                        false => (index, value),
                    },
                );
                argmin
            }
            _ => {
                panic!("Impossible branch, invalid turn");
            }
        }
    }

    /// Computes the minimax value of the current board state
    fn eval_recursive(&mut self, board: &Board) -> i8 {
        use Tile::*;
        let hash = board.invariant_hash();
        match self.value_table.get(&hash) {
            // If the current position is in our value table, simply return the value from the hash table
            Some(x) => *x,
            None => match board.winner() {
                // Otherwise, check if we are in a terminal state
                Some(X) => {
                    let value = (board.empty().len() + 1) as i8;
                    self.value_table.insert(hash, value);
                    value
                }
                Some(O) => {
                    let value = -((board.empty().len() + 1) as i8);
                    self.value_table.insert(hash, value);
                    value
                }
                _ => {
                    let empty = board.empty();
                    match empty.is_empty() {
                        true => {
                            let value = 0;
                            self.value_table.insert(hash, value);
                            value
                        }
                        // Otherwise, compute values for all children
                        false => {
                            let children: Vec<Board> = empty
                                .into_iter()
                                .map(|i| {
                                    let mut new_board = (*board).clone();
                                    let _ = new_board.act(i);
                                    new_board
                                })
                                .collect();
                            let child_values: Vec<i8> = children
                                .into_iter()
                                .map(|x| self.eval_recursive(&x))
                                .collect();
                            let value = match board.turn() {
                                X => child_values.into_iter().max().unwrap(),
                                O => child_values.into_iter().min().unwrap(),
                                _ => panic!("Impossible branch, invalid turn"),
                            };

                            self.value_table.insert(hash, value);
                            value
                        }
                    }
                }
            },
        }
    }
}

impl Default for SolutionTable {
    fn default() -> Self {
        SolutionTable {
            value_table: HashMap::new(),
        }
    }
}

fn main() {
    use std::io::stdin;
    let args: Vec<String> = std::env::args().collect();
    println!("{:?}", args);

    let player_turn: Tile = match args.get(1) {
        Some(x) => match x.as_str() {
            "O" => Tile::O,
            _ => Tile::X,
        },
        None => Tile::X,
    };

    let mut board = Board::default();
    let mut solution = SolutionTable::default();

    println!("{board}");

    while board.winner().is_none() && !board.empty().is_empty() {
        if board.turn() == player_turn {
            let mut input_buffer = String::new();
            let _ = stdin().read_line(&mut input_buffer);
            let i = input_buffer.trim().parse::<usize>();
            match i {
                Ok(i) => {
                    let _ = board.act(i);
                }
                _ => {
                    println!("Invalid move!");
                }
            }
            // Read input
        } else {
            let argmin = solution.solve(&board);
            let _ = board.act(argmin);
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
            tiles: [
                Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty,
            ],
        };
        assert_eq!(board.hash(), 0);

        let board = Board {
            tiles: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, X],
        };
        assert_eq!(board.hash(), 1);

        let board = Board {
            tiles: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, X, Empty],
        };
        assert_eq!(board.hash(), 3);
    }

    #[test]
    fn test_invariant_hash() {
        use Tile::*;

        let board = Board {
            tiles: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty, X],
        };
        assert_eq!(board.invariant_hash(), 1);
        let board = Board {
            tiles: [X, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        };
        assert_eq!(board.invariant_hash(), 1);
        let board = Board {
            tiles: [Empty, Empty, X, Empty, Empty, Empty, Empty, Empty, Empty],
        };
        assert_eq!(board.invariant_hash(), 1);
        let board = Board {
            tiles: [Empty, Empty, Empty, Empty, Empty, Empty, X, Empty, Empty],
        };
        assert_eq!(board.invariant_hash(), 1);

        assert_eq!(
            Board {
                tiles: [Empty, X, Empty, Empty, Empty, Empty, Empty, Empty, Empty]
            }
            .invariant_hash(),
            Board {
                tiles: [Empty, Empty, Empty, Empty, Empty, Empty, Empty, X, Empty]
            }
            .invariant_hash()
        )
    }

    #[test]
    fn test_winner() {
        use Tile::*;

        assert_eq!(Board::default().winner(), None);

        let board = Board {
            tiles: [X, O, O, Empty, X, Empty, Empty, Empty, X],
        };

        assert_eq!(board.winner(), Some(X));
    }

    #[test]
    fn test_print() {
        let x = format!("{}", Board::default());
        assert_eq!(x, " | | \n-----\n | | \n-----\n | | \n");
    }

    #[test]
    fn test_turn() {
        use Tile::*;
        let x = Board::default();
        assert_eq!(x.turn(), X);
    }

    #[test]
    fn test_state_transitions() {
        use Tile::*;

        let mut board = Board::default();
        let _ = board.act(0);
        assert_eq!(board.winner(), None);

        let _ = board.act(3);
        assert_eq!(board.winner(), None);

        let _ = board.act(1);
        assert_eq!(board.winner(), None);

        let _ = board.act(4);
        assert_eq!(board.winner(), None);

        let _ = board.act(2);
        assert_eq!(board.winner(), Some(X));
    }

    #[test]
    fn test_empty() {
        use Tile::*;

        let x = Board::default();
        assert_eq!(x.empty(), vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);

        let x = Board {
            tiles: [X, Empty, Empty, Empty, Empty, Empty, Empty, Empty, Empty],
        };
        assert_eq!(x.empty(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_solver() {
        use Tile::*;
        let mut solver = SolutionTable::default();
        assert_eq!(solver.eval_recursive(&Board::default()), 0); // Theoretical draw
        assert_eq!(solver.value_table.len(), 765);

        assert_eq!(
            solver.eval_recursive(&Board {
                tiles: [X, O, Empty, Empty, Empty, Empty, Empty, Empty, Empty]
            }),
            3
        );
    }
}
