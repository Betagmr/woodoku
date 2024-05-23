from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
from .blocks import blocks
from src.environment.render_mode import AnsiRender


class WoodokuEnv(gym.Env):
    metadata = {
        "game_modes": ["woodoku"],
        "render_modes": ["ansi", "rgb_array", "human"],
        "render_fps": 10,
        "score_modes": ["woodoku"],
    }

    def __init__(
        self,
        score_mode="woodoku",
        render_mode=None,
        crash33=True,
    ):
        if render_mode is None or render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"{render_mode} is not in {self.metadata['render_modes']}")

        if score_mode is None or score_mode not in self.metadata["score_modes"]:
            raise ValueError(f"{score_mode} is not in {self.metadata['score_modes']}")

        # RENDER SETTINGS
        self.render_mode = AnsiRender()
        self.score_mode = score_mode
        self.crash33 = crash33
        self.clock = None
        self._block_list = blocks

        # GAME PROPERTIES

        self.MAX_BLOCK_NUM = 3
        self.BLOCK_LENGTH = 5
        self.BOARD_HEIGHT = 9
        self.BOARD_WIDTH = 9
        self.BOARD_LENGTH = 9  # Will be removed in the future
        self.BOARD_SCUARES = self.BOARD_HEIGHT * self.BOARD_WIDTH
        self.NUMBER_BLOCKS = self.MAX_BLOCK_NUM * self.BOARD_SCUARES
        self.BOARD_SHAPE = (self.BOARD_HEIGHT, self.BOARD_WIDTH)

        # ACTION AND OBSERVATION SPACE
        self.action_space = spaces.Discrete(self.NUMBER_BLOCKS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=1,
                    shape=self.BOARD_SHAPE,
                    dtype=np.int8,
                ),
                "block_1": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_2": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_3": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
            }
        )

    def _get_3_blocks(
        self,
    ) -> Tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]]:
        n_blocks = 3
        blocks = self.np_random.choice(
            range(self._block_list.shape[0]),
            n_blocks,
            replace=False,
        )

        self._block_valid_pos = []
        for i in range(n_blocks):
            valid_list = []
            for row in range(self.BLOCK_LENGTH):
                for col in range(self.BLOCK_LENGTH):
                    if self._block_list[blocks[i]][row][col] == 1:
                        valid_list.append((row, col))

            self._block_valid_pos.append(valid_list)

        return (
            self._block_list[blocks[0]].copy(),
            self._block_list[blocks[1]].copy(),
            self._block_list[blocks[2]].copy(),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # game data
        self.step_count = 0
        self._score = 0
        self._board = np.zeros(self.BOARD_SHAPE, dtype=np.int8)

        # get 3 blocks
        self._get_3_blocks_random()
        self._block_exist = [True, True, True]

        # Whether a block can be placed in its place.
        self.legality = np.zeros((self.NUMBER_BLOCKS,), dtype=np.int8)
        self._get_legal_actions()

        # Shows how many pieces are broken in a row
        # (this is different from how many pieces are broken at once)
        self.is_legal = True
        self.n_cell = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), self._get_info()

    def _get_3_blocks_random(self):
        # randomly select three blocks
        self._block_exist = [True, True, True]
        self._block_1, self._block_2, self._block_3 = self._get_3_blocks()

    def _get_legal_actions(self):
        """
        Checks whether there is a block corresponding to the action
        Check if the block can be placed at the location.
        """
        for action in range(self.NUMBER_BLOCKS):
            if self._is_valid_block(action) and self._is_valid_position(action):
                self.legality[action] = 1
            else:
                self.legality[action] = 0

    def _get_obs(self) -> Dict[str, NDArray[np.int8]]:
        return {
            "board": self._board,
            "block_1": self._block_1,
            "block_2": self._block_2,
            "block_3": self._block_3,
        }

    def _get_info(self):
        return {
            "action_mask": self.legality,
            "score": self._score,
            "is_legal": self.is_legal,
            "n_cell": self.n_cell,
        }

    def _is_terminated(self) -> bool:
        # Check if the game can be continued with the blocks you own
        # If any number of cases can proceed, False is returned.
        for blk_num in range(self.MAX_BLOCK_NUM):
            if self._block_exist[blk_num]:
                for act in range(
                    blk_num * self.BOARD_SCUARES, (blk_num + 1) * self.BOARD_SCUARES
                ):
                    if self._is_valid_position(act):
                        return False
        return True

    def _nonexist_block(self, action: int):
        # Deactivate the block corresponding to the action and set the array to 0.
        self._block_exist[action // self.BOARD_SCUARES] = False
        block, _ = self.action_to_blk_pos(action)
        block.fill(0)

    def _is_valid_position(self, action: int) -> bool:
        block, location = self.action_to_blk_pos(action)

        for row, col in self._block_valid_pos[action // self.BOARD_SCUARES]:
            # location - 2 : leftmost top (0, 0)
            # When the block is located outside the board
            if not (
                0 <= (location[0] - 2 + row) < 9 and 0 <= (location[1] - 2 + col) < 9
            ):
                return False

            # When there is already another block
            if self._board[location[0] - 2 + row][location[1] - 2 + col] == 1:
                return False

        return True

    # Check whether the block corresponding to the action is valid.
    def _is_valid_block(self, action: int) -> bool:
        if self._block_exist[action // self.BOARD_SCUARES]:
            return True
        else:
            return False

    def _get_score(self):
        return self.n_cell

    def action_to_blk_pos(
        self, action: int
    ) -> Tuple[NDArray[np.int8], Tuple[int, int]]:
        """Converts an integer representing the action into the corresponding `_block_*` and the desired placement coordinates, and returns them.
        The placement coordinates refer to the location of the index (2, 2), which is the center of the 5x5 block.

        Args:
            action (int): action, [0, 242]

        Returns:
            Tuple[NDArray[np.int8], Tuple[int, int]]: (corresponding `_block_*`, (r coordinate, c coordinate))
        """
        # First Block
        if 0 <= action <= 80:
            block = self._block_1
            location = (action // 9, action % 9)

        # Second Block
        elif self.BOARD_SCUARES <= action <= 161:
            block = self._block_2
            location = (
                (action - self.BOARD_SCUARES) // 9,
                (action - self.BOARD_SCUARES) % 9,
            )

        # Third Block
        # 162 <= action < self.NUMBER_BLOCKS
        else:
            block = self._block_3
            location = ((action - 162) // 9, (action - 162) % 9)

        return block, location

    def get_block_square(self, block: NDArray[np.int8]) -> Tuple[int, int, int, int]:
        """Creates a rectangle that encloses only the actual blocks within a 5x5 grid,
        and returns the relative edge coordinates to the center.

        Args:
            block (NDArray[np.int8]): An array representing the block shape with a size of 5x5.

        Returns:
            Tuple[int, int, int, int]: (r_min, r_max, c_min, c_max),
            The position of the corner enclosing the block in the 5x5 array representing the block.
            The rows of the actual block are in the range [r_min, r_max] and the columns are in the range [c_min, c_max].
            In r and c, index 0 represents the top, leftmost corner.
        """

        r_min = 6
        r_max = -1
        c_min = 6
        c_max = -1

        # `.sum()` is faster than `.any()`
        for r in range(block.shape[0]):
            if block[r, :].sum() > 0:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
        for c in range(block.shape[1]):
            if block[:, c].sum() > 0:
                c_min = min(c_min, c)
                c_max = max(c_max, c)

        return (r_min, r_max, c_min, c_max)

    def place_block(self, action: int):
        """Finds the block and its position corresponding to the action, and places the block at that location.

        Args:
            action (int): action
        """
        # c_loc : where the center of the block is placed
        block, c_loc = self.action_to_blk_pos(action)
        r_min, r_max, c_min, c_max = self.get_block_square(block)
        self._board[
            c_loc[0] - 2 + r_min : c_loc[0] + r_max - 1,
            c_loc[1] + c_min - 2 : c_loc[1] + c_max - 1,
        ] += block[r_min : r_max + 1, c_min : c_max + 1]

    def step(
        self, action: int
    ) -> Tuple[
        Dict[str, NDArray[np.int8]],
        int,
        bool,
        bool,
        Dict[str, NDArray[np.int8] | int | bool],
    ]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        terminated = False

        self.combo = 0
        self.n_cell = 0

        # invalid action, attempting to place a block where a block is already placed
        if not self.legality[action]:
            self.is_legal = False
            reward = 0
            terminated = False
        else:
            self.is_legal = True
            self.place_block(action)

            # If there is a block to destroy, destroy it and get the corresponding reward.
            reward = self._get_score()
            self._score += self._get_score()

            # make block zero and _block_exist to False
            self._nonexist_block(action)

            # Check if all 3 blocks have been used.
            # If the block does not exist, a new block is obtained.
            if sum(self._block_exist) == 0:
                self._get_3_blocks_random()

            # Check if the game is terminated.
            terminated = self._is_terminated()

            self._get_legal_actions()

        self.step_count += 1
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        self.render_mode.render(self)

    def close(self):
        pass
        # if self.window is not None:
        #     pygame.display.quit()
        #     pygame.quit()
