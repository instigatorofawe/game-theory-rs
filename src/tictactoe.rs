use std::fmt::Display;

/// Tile on a tic-tac-toe board
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tile::Empty => write!(f, " "),
            Tile::X => write!(f, "X"),
            Tile::O => write!(f, "O"),
        }
    }
}

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
}
