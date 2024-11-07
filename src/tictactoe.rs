use std::fmt::Display;

/// Tile on a tic-tac-toe board
#[derive(PartialEq, Clone, Debug)]
enum Tile {
    Empty,
    X,
    O,
}

impl Tile {
    /// Computes the numerical representation of the tile
    pub fn hash(&self) -> u32 {
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
struct Board {
    tiles: [Tile; 9],
}

impl Board {
    /// Computes the numerical representation of the current board state
    pub fn hash(&self) -> u32 {
        self.tiles.iter().fold(0, |i, x| i * 3 + x.hash())
    }

    /// Computes the rotation and reflection invariant representation of the current board state
    pub fn invariant_hash(&self) -> u32 {
        const CONFIGURATIONS: [[usize; 9]; 8] = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8], // Rotations
            [2, 5, 8, 1, 4, 7, 0, 3, 6],
            [8, 7, 6, 5, 4, 3, 2, 1, 0],
            [6, 3, 0, 7, 4, 1, 8, 5, 2],
            [6, 7, 8, 3, 4, 5, 0, 1, 2], // Reflections
            [2, 1, 0, 5, 4, 3, 8, 7, 6],
            [8, 5, 2, 7, 4, 1, 6, 3, 0],
            [0, 3, 6, 1, 4, 7, 2, 5, 8],
        ];

        CONFIGURATIONS
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

    /// The player whose turn it is makes their move. The board is modified in-place.
    pub fn act(&mut self, index: usize) {
        todo!()
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

fn main() {}

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
}
