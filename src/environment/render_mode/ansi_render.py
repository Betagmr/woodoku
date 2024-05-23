import numpy as np
from numpy.typing import NDArray


class AnsiRender:
    def __init__(self):
        self.display_height = 17
        self.display_width = 21

    def _line_printer(self, line: NDArray) -> str:
        return np.array2string(
            line,
            separator=" ",
            formatter={"str_kind": lambda x: x},
        )

    def render(self, env):
        new_board = np.where(env._board == 1, "■", "□")
        new_block_1 = np.where(env._block_1 == 1, "■", "□")
        new_block_2 = np.where(env._block_2 == 1, "■", "□")
        new_block_3 = np.where(env._block_3 == 1, "■", "□")
        game_display = np.full(
            (self.display_height, self.display_width), " ", dtype="<U1"
        )

        # copy board
        game_display[1:10, 1:10] = new_board

        # copy block
        for i, block in enumerate([new_block_1, new_block_2, new_block_3]):
            game_display[11:16, 7 * i + 1 : 7 * i + 6] = block

        # Display game_display
        for i in range(self.display_height):
            print(self._line_printer(game_display[i])[1:-1])
