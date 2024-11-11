use hashbrown::HashMap;
use std::fmt::Display;

/// Number of tiles on the board
const BOARD_SIZE: usize = 9;

/// Possible winning configurations
const WIN_LINES: [u16; 8] = [
    0b111_000_000,
    0b000_111_000,
    0b000_000_111,
    0b100_100_100,
    0b010_010_010,
    0b001_001_001,
    0b100_010_001,
    0b001_010_100,
];

/// Rotations and reflections
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

/// Bitboard representation of a tic tac toe board
#[derive(Clone)]
struct Board {
    /// Whether each tile is empty: 0 = empty, 1 = not empty
    occupied: u16,
    /// If the tile is not empty, which player occupies the tile: 0 = O, 1 = X
    player: u16,
}

impl Default for Board {
    /// Default value is an empty Board
    fn default() -> Self {
        Board {
            occupied: 0,
            player: 0,
        }
    }
}

/// Possible values of a tile on the board: occupied by an X, O, or Empty
#[derive(Debug, PartialEq)]
enum Tile {
    X,
    O,
    Empty,
}

impl Tile {
    /// String representation of the current tile; can pass a string to represent the empty tile
    fn str<'a>(&self, empty: Option<&'a str>) -> &'a str {
        match self {
            Tile::X => "X",
            Tile::O => "O",
            Tile::Empty => match empty {
                Some(x) => x,
                _ => " ",
            },
        }
    }

    /// Computes hash value of the current tile
    fn hash(&self) -> u16 {
        match self {
            Tile::Empty => 0,
            Tile::X => 1,
            Tile::O => 2,
        }
    }
}

/// Error type for bound checking for statically sized arrays and other data structures
#[derive(Debug)]
enum GameError {
    OutOfBoundsError,
    InvalidMoveError,
}

impl Board {
    /// Gets the tile at the specified index
    fn get(&self, index: usize) -> Result<Tile, GameError> {
        // Bound checking
        if index > BOARD_SIZE {
            return Err(GameError::OutOfBoundsError);
        } else {
            let occupied = (1 << index) & self.occupied > 0;
            let player = (1 << index) & self.player > 0;

            match occupied {
                false => Ok(Tile::Empty),
                true => match player {
                    true => Ok(Tile::X),
                    false => Ok(Tile::O),
                },
            }
        }
    }

    /// Sets the tile at the specified index
    fn set(&mut self, index: usize, tile: Tile) -> Result<(), GameError> {
        // Bound checking
        if index > BOARD_SIZE {
            return Err(GameError::OutOfBoundsError);
        } else {
            match tile {
                Tile::Empty => self.occupied &= !(1 << index),
                Tile::X => {
                    self.occupied |= 1 << index;
                    self.player |= 1 << index;
                }
                Tile::O => {
                    self.occupied |= 1 << index;
                    self.player &= !(1 << index);
                }
            }
            Ok(())
        }
    }

    /// Determines whose turn it is, X or O
    fn turn(&self) -> Tile {
        let moves = self.occupied.count_ones();
        match moves % 2 {
            0 => Tile::X,
            _ => Tile::O,
        }
    }

    /// Computes the current winner, if there is one
    fn winner(&self) -> Tile {
        let x_pos = self.occupied & self.player;
        let o_pos = self.occupied & !self.player;

        for line in WIN_LINES {
            if x_pos & line == line {
                return Tile::X;
            }
            if o_pos & line == line {
                return Tile::O;
            }
        }
        Tile::Empty
    }

    /// Lists indices of valid moves
    fn valid_moves(&self) -> Vec<usize> {
        (0..BOARD_SIZE)
            .into_iter()
            .filter(|x| self.occupied & (1 << x) == 0)
            .collect()
    }

    /// Tries to set the index to the tile of the player whose turn it is to act
    fn act(&mut self, index: usize) -> Result<(), GameError> {
        let current_value = self.get(index)?;
        match current_value {
            Tile::Empty => self.set(index, self.turn()),
            _ => Err(GameError::InvalidMoveError),
        }
    }

    /// Computes transformation invariant hash of the current board state
    fn invariant_hash(&self) -> u16 {
        let hash_values: Vec<u16> = (0..BOARD_SIZE)
            .into_iter()
            .map(|x| self.get(x).expect("Unable to get tile").hash())
            .collect();
        TRANSFORMATIONS
            .iter()
            .map(|x| x.iter().fold(0, |i, x| i * 3 + hash_values[*x]))
            .min()
            .expect("Empty iterator")
    }
}

impl Display for Board {
    /// Print formatted representation of board
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n",
            self.get(0).expect("Couldn't get tile 0").str(Some("0")),
            self.get(1).expect("Couldn't get tile 1").str(Some("1")),
            self.get(2).expect("Couldn't get tile 2").str(Some("2")),
            self.get(3).expect("Couldn't get tile 3").str(Some("3")),
            self.get(4).expect("Couldn't get tile 4").str(Some("4")),
            self.get(5).expect("Couldn't get tile 5").str(Some("5")),
            self.get(6).expect("Couldn't get tile 6").str(Some("6")),
            self.get(7).expect("Couldn't get tile 7").str(Some("7")),
            self.get(8).expect("Couldn't get tile 8").str(Some("8")),
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
        let empty = board.valid_moves();
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
                X => {
                    let value = BOARD_SIZE as i8 - board.occupied.count_ones() as i8 + 1;
                    self.value_table.insert(hash, value);
                    value
                }
                O => {
                    let value = -(BOARD_SIZE as i8 - board.occupied.count_ones() as i8 + 1);
                    self.value_table.insert(hash, value);
                    value
                }
                _ => {
                    let valid_moves = board.valid_moves();
                    match valid_moves.is_empty() {
                        true => {
                            let value = 0;
                            self.value_table.insert(hash, value);
                            value
                        }
                        // Otherwise, compute values for all children
                        false => {
                            let children: Vec<Board> = valid_moves
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

    while board.winner() == Tile::Empty && board.occupied.count_ones() < BOARD_SIZE as u32 {
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
    fn test_board_get() {
        for index in 0..BOARD_SIZE {
            let board_x = Board {
                occupied: 1 << index,
                player: 1 << index,
            };
            let board_o = Board {
                occupied: 1 << index,
                player: !0 & !(1 << index),
            };

            for j in 0..BOARD_SIZE {
                let result = board_x.get(j);
                assert!(result.is_ok());
                match index == j {
                    false => assert_eq!(result.unwrap(), Tile::Empty),
                    true => assert_eq!(result.unwrap(), Tile::X),
                }

                let result = board_o.get(j);
                assert!(result.is_ok());
                match index == j {
                    false => assert_eq!(result.unwrap(), Tile::Empty),
                    true => assert_eq!(result.unwrap(), Tile::O),
                }
            }

            assert!(board_x.get(BOARD_SIZE + 1).is_err());
        }
    }

    #[test]
    fn test_board_format() {
        let board = Board::default();
        let str: String = format!("{}", board);
        assert_eq!(str, "0|1|2\n-----\n3|4|5\n-----\n6|7|8\n");

        let board = Board {
            occupied: 1,
            player: 1,
        };
        let str: String = format!("{}", board);
        assert_eq!(str, "X|1|2\n-----\n3|4|5\n-----\n6|7|8\n");

        let board = Board {
            occupied: 1 << 5,
            player: 0,
        };
        let str: String = format!("{}", board);
        assert_eq!(str, "0|1|2\n-----\n3|4|O\n-----\n6|7|8\n");
    }

    #[test]
    fn test_board_turn() {
        let board = Board::default();
        assert_eq!(board.turn(), Tile::X);

        let board = Board {
            occupied: 1,
            player: 1,
        };
        assert_eq!(board.turn(), Tile::O);

        let board = Board {
            occupied: 3,
            player: 1,
        };
        assert_eq!(board.turn(), Tile::X);
    }

    #[test]
    fn test_board_set() {
        let mut board = Board::default();
        assert!(board.set(0, Tile::X).is_ok());
        assert_eq!(board.get(0).expect("Unable to get tile 0"), Tile::X);

        assert!(board.set(1, Tile::X).is_ok());
        assert_eq!(board.get(1).expect("Unable to get tile 1"), Tile::X);

        assert!(board.set(0, Tile::O).is_ok());
        assert_eq!(board.get(0).expect("Unable to get tile 0"), Tile::O);
    }

    #[test]
    fn test_board_invariant_hash() {
        // Default position hashes to 0
        assert_eq!(Board::default().invariant_hash(), 0);
        // Adding a tile changes hash
        assert_ne!(
            Board::default().invariant_hash(),
            Board {
                occupied: 0b100_000_000,
                player: 0
            }
            .invariant_hash()
        );
        assert_ne!(
            Board::default().invariant_hash(),
            Board {
                occupied: 0b100_000_000,
                player: 0b100_000_000
            }
            .invariant_hash()
        );
        // Player matters
        assert_ne!(
            Board {
                occupied: 0b010_000_000,
                player: 0
            }
            .invariant_hash(),
            Board {
                occupied: 0b010_000_000,
                player: 0b010_000_000
            }
            .invariant_hash()
        );
        // Position matters
        assert_ne!(
            Board {
                occupied: 0b010_000_000,
                player: 0
            }
            .invariant_hash(),
            Board {
                occupied: 0b100_000_000,
                player: 0
            }
            .invariant_hash()
        );
        // Reflection and rotation invariant
        assert_eq!(
            Board {
                occupied: 0b100_000_000,
                player: 0
            }
            .invariant_hash(),
            Board {
                occupied: 0b001_000_000,
                player: 0
            }
            .invariant_hash()
        );
        assert_eq!(
            Board {
                occupied: 0b100_000_000,
                player: 0
            }
            .invariant_hash(),
            Board {
                occupied: 0b000_000_001,
                player: 0
            }
            .invariant_hash()
        );
        // More complicated positions
        assert_eq!(
            Board {
                occupied: 0b110_000_000,
                player: 0b100_000_000
            }
            .invariant_hash(),
            Board {
                occupied: 0b011_000_000,
                player: 0b001_000_000
            }
            .invariant_hash()
        );
    }

    #[test]
    fn test_board_winner() {
        assert_eq!(Board::default().winner(), Tile::Empty);
        for line in WIN_LINES {
            let board = Board {
                occupied: line,
                player: line,
            };
            assert_eq!(board.winner(), Tile::X);

            let board = Board {
                occupied: line,
                player: !line,
            };
            assert_eq!(board.winner(), Tile::O);
        }
    }

    #[test]
    fn test_board_valid_moves() {
        assert_eq!(
            Board::default().valid_moves(),
            (0..BOARD_SIZE).collect::<Vec<_>>()
        );
        for i in 0..BOARD_SIZE {
            assert_eq!(
                Board {
                    occupied: !(1 << i),
                    player: 0
                }
                .valid_moves(),
                vec![i]
            );
        }
    }

    #[test]
    fn test_solver() {
        let mut solver = SolutionTable::default();
        assert_eq!(solver.eval_recursive(&Board::default()), 0); // Theoretical draw
        assert_eq!(solver.value_table.len(), 765);

        assert_eq!(
            solver.eval_recursive(&Board {
                occupied: 0b110_000_000,
                player: 0b100_000_000
            }),
            3 // Win for X
        );
    }
}
