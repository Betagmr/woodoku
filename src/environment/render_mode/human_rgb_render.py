import pygame
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)


class HumanRGBRender:
    def __init__(self) -> None:
        self.window = None
        self.board_square_size = 32
        self.block_square_size = 24
        self.window_size = 1.5 * 512
        
        self.mode = 'human'
        self.render_mode = 'human'
        self.clock = None

    def create_window(self) -> None:
        pygame.init()

        # render
        board_total_size = self.board_square_size * 9
        block_total_size = self.block_square_size * 5
        board_left_margin = (self.window_size - board_total_size) // 2
        block_left_margin = (self.window_size - block_total_size * 3) // 4
        top_margin = (self.window_size - board_total_size - block_total_size) // 3

        # Initialize the positions of the squares on the board.
        self.board_row_pos = np.zeros(self.BOARD_LENGTH, dtype=np.uint32)
        self.board_col_pos = np.zeros(self.BOARD_LENGTH, dtype=np.uint32)

        for i in range(self.BOARD_LENGTH):
            self.board_col_pos[i] = board_left_margin + self.board_square_size * i
            self.board_row_pos[i] = top_margin + self.board_square_size * i

        # Initializes the position of the square in the block.
        self.block_row_pos = np.zeros(self.BLOCK_LENGTH, dtype=np.uint32)
        self.block_col_pos = np.zeros(
            (self.MAX_BLOCK_NUM, self.BLOCK_LENGTH), dtype=np.uint32
        )

        for i in range(self.BLOCK_LENGTH):
            self.block_row_pos[i] = (
                self.window_size
                - top_margin
                - block_total_size
                + self.block_square_size * i
            )

        for b in range(self.MAX_BLOCK_NUM):
            for i in range(self.BLOCK_LENGTH):
                self.block_col_pos[b][i] = (
                    block_left_margin
                    + (block_left_margin + block_total_size) * b
                    + self.block_square_size * i
                )

        if self.mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        elif self.mode == "rgb_array":
            self.window = pygame.Surface((self.window_size, self.window_size))

    def render(self, obs) -> None:
        pygame.font.init()
        if self.window is None:
            self.create_window()

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)

        # draw board square
        for row in range(self.BOARD_LENGTH):
            for col in range(self.BOARD_LENGTH):
                if self._board[row][col] == 1:
                    pygame.draw.rect(
                        canvas,
                        BLACK,
                        pygame.Rect(
                            (self.board_col_pos[col], self.board_row_pos[row]),  # pos
                            (self.board_square_size, self.board_square_size),
                        ),
                    )
                pygame.draw.rect(
                    canvas,
                    GRAY,
                    pygame.Rect(
                        (self.board_col_pos[col], self.board_row_pos[row]),  # pos
                        (self.board_square_size, self.board_square_size),
                    ),
                    2,
                )

        # draw block square
        for idx, block in enumerate([self._block_1, self._block_2, self._block_3]):
            for row in range(self.BLOCK_LENGTH):
                for col in range(self.BLOCK_LENGTH):
                    if block[row][col] == 1:
                        pygame.draw.rect(
                            canvas,
                            BLACK,
                            pygame.Rect(
                                (
                                    self.block_col_pos[idx][col],
                                    self.block_row_pos[row],
                                ),  # pos
                                (self.block_square_size, self.block_square_size),
                            ),
                        )
                    pygame.draw.rect(
                        canvas,
                        GRAY,
                        pygame.Rect(
                            (
                                self.block_col_pos[idx][col],
                                self.block_row_pos[row],
                            ),  # pos
                            (self.block_square_size, self.block_square_size),
                        ),
                        2,
                    )

        myFont = pygame.font.SysFont(None, 30)
        num = myFont.render(f"step: {self.step_count}", True, (0, 0, 0))
        score = myFont.render(f"score: {self._score}", True, (0, 0, 0))
        straight = myFont.render(f"straight: {self.straight}", True, (0, 0, 0))
        canvas.blit(score, (10, 5))
        canvas.blit(num, (200, 5))
        canvas.blit(straight, (340, 5))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
