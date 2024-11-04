import pygame
import sys
import time
import os
from othello_game import OthelloGame
from ai_agent_ga import GeneticOthelloAI
from ai_agent import MinimaxOthelloAI
from ai_agent_SA import SimulatedAnnealingOthelloAI
from ai_agent_v3 import MinimaxV3

# Constants and colors
WIDTH, HEIGHT = 480, 560
BOARD_SIZE = 8
SQUARE_SIZE = (HEIGHT - 80) // BOARD_SIZE
BLACK_COLOR = (0, 0, 0)
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 128, 0)

class OthelloGUI:
    def __init__(self, player_mode="friend", agent_a = None, agent_b = None):
        """
        A graphical user interface (GUI) for playing the Othello game.

        Args:
            player_mode (str): The mode of the game, either "friend" or "ai" (default is "friend").
        """
        self.win = self.initialize_pygame()
        self.game = OthelloGame(player_mode=player_mode)
        self.message_font = pygame.font.SysFont(None, 24)
        self.message = ""
        self.invalid_move_message = ""
        self.flip_sound = pygame.mixer.Sound("./utils/sounds/disk_flip.mp3")
        self.end_game_sound = pygame.mixer.Sound("./utils/sounds/end_game.mp3")
        self.invalid_play_sound = pygame.mixer.Sound("./utils/sounds/invalid_play.mp3")
        self.agent_type_a = agent_a
        self.agent_type_b = agent_b
        self.log_file = time.strftime("log-%Y%m%d-%H:%M:%S.log", time.localtime())
        self.make_log_file()
        self.set_agents()
        
    def initialize_pygame(self):
        """
        Initialize the Pygame library and create the game window.

        Returns:
            pygame.Surface: The Pygame surface representing the game window.
        """
        pygame.init()
        win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Othello")
        return win
    
    def set_agents(self):
        self.message = "Initializing AI"
        self.draw_board()
        if self.game.player_mode == "ai":
            self.agent_a = self.initialize_agent(self.agent_type_a)
        elif self.game.player_mode == "agent":
            self.agent_a = self.initialize_agent(self.agent_type_a)
            self.agent_b = self.initialize_agent(self.agent_type_b)

    def initialize_agent(self, agent_type):
        """Returns the appropriate agent based on the specified type."""
        if agent_type == "Genetic Algorithm":
            return GeneticOthelloAI()
        elif agent_type == "Minimax":
            return MinimaxOthelloAI()
        elif agent_type == "MinimaxV3":
            return MinimaxV3()
        elif agent_type == "Simulated Annealing":
            return SimulatedAnnealingOthelloAI()
        else:
            return None  # This means it's a human player

    def draw_board(self):
        """
        Draw the Othello game board and messaging area on the window.
        """
        self.win.fill(GREEN_COLOR)

        # Draw board grid and disks
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(
                    self.win,
                    BLACK_COLOR,
                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE),
                    1,
                )
                if self.game.board[row][col] == 1:
                    pygame.draw.circle(
                        self.win,
                        BLACK_COLOR,
                        ((col + 0.5) * SQUARE_SIZE, (row + 0.5) * SQUARE_SIZE),
                        SQUARE_SIZE // 2 - 4,
                    )
                elif self.game.board[row][col] == -1:
                    pygame.draw.circle(
                        self.win,
                        WHITE_COLOR,
                        ((col + 0.5) * SQUARE_SIZE, (row + 0.5) * SQUARE_SIZE),
                        SQUARE_SIZE // 2 - 4,
                    )

        # Draw messaging area
        message_area_rect = pygame.Rect(
            0, BOARD_SIZE * SQUARE_SIZE, WIDTH, HEIGHT - (BOARD_SIZE * SQUARE_SIZE)
        )
        pygame.draw.rect(self.win, WHITE_COLOR, message_area_rect)

        # Draw player's turn message
        player_turn = "Black's" if self.game.current_player == 1 else "White's"
        turn_message = f"{player_turn} turn"
        message_surface = self.message_font.render(turn_message, True, BLACK_COLOR)
        message_rect = message_surface.get_rect(
            center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 - 20)
        )
        self.win.blit(message_surface, message_rect)

        # Draw invalid move message
        if self.message:
            invalid_move_message = self.message
            message_surface = self.message_font.render(
                invalid_move_message, True, BLACK_COLOR
            )
            message_rect = message_surface.get_rect(
                center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 + 20)
            )
            self.win.blit(message_surface, message_rect)

        # Draw invalid move message
        if self.invalid_move_message:
            message_surface = self.message_font.render(
                self.invalid_move_message, True, BLACK_COLOR
            )
            message_rect = message_surface.get_rect(
                center=(WIDTH // 2, (HEIGHT + BOARD_SIZE * SQUARE_SIZE) // 2 + 20)
            )
            self.win.blit(message_surface, message_rect)

        pygame.display.update()

    def handle_input(self):
        """
        Handle user input events such as mouse clicks and game quitting.
        """
        row, col = (0, 0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                if self.game.is_valid_move(row, col):
                    self.game.make_move(row, col)
                    self.invalid_move_message = (
                        ""  # Clear any previous invalid move message
                    )
                    self.flip_sound.play()  # Play flip sound effect
                else:
                    self.invalid_move_message = "Invalid move! Try again."
                    self.invalid_play_sound.play()  # Play invalid play sound effect
        return (row, col)
    
    def agent_move(self, agent):
        """Runs the turn for the specified AI agent."""
        # Show AI thinking message based on the current player
        if self.game.current_player == 1:
            self.message = f"{self.agent_type_a} AI is thinking..."
        else:
            self.message = f"{self.agent_type_b} AI is thinking..."
            
        self.draw_board()  # Update the board with the message
        pygame.time.delay(500)  # Short delay for AI thinking effect

        # Get the best move from the specified AI agent
        ai_move = agent.get_best_move(self.game)
        
        # Make the AI move on the game board
        if ai_move:
            self.game.make_move(*ai_move)
            return ai_move

    def make_log_file(self):
        if not os.path.exists('./game-log'):
            os.mkdir('./game-log')
        with open(f'./game-log/{self.log_file}', 'w') as file:
            player_1 = 'Human'
            player_2 = self.agent_type_b or self.agent_type_a
            if self.game.player_mode == 'agent':
                player_1 = self.agent_type_a
            file.write(f"Black ({player_1}) vs White ({player_2})\n")

    def run_game(self, return_to_menu_callback=None):
        """
        Run the main game loop until the game is over and display the result.
        """
        execution_time_a = []
        execution_time_b = []
        while not self.game.is_game_over():
            player = 'Human'
            start_time = time.time()
            move = None
            turn = 'Black' if self.game.current_player == 1 else 'White'
            if self.game.player_mode == "friend":
                move = self.handle_input()

            # If it's the AI player's turn
            if self.game.player_mode == "ai" and self.game.current_player == -1:
                move = self.agent_move(self.agent_a)
                player = self.agent_type_a
            
            elif self.game.player_mode == "ai" and self.game.current_player == 1:
                self.handle_input()
            
            # If it's the AI player's turn
            elif self.game.player_mode == "agent" and self.game.current_player == 1:
                move = self.agent_move(self.agent_a)
                player = self.agent_type_a

            elif self.game.player_mode == "agent" and self.game.current_player == -1:
                move = self.agent_move(self.agent_b)
                player = self.agent_type_b

            with open(f'./game-log/{self.log_file}', 'a') as file:
                time_taken = time.time() - start_time
                if self.game.current_player == 1:
                    execution_time_a.append(time_taken)
                else:
                    execution_time_b.append(time_taken)
                file.write(f"{turn} ({player}) move {move} took {time_taken:.2f}s\n")

            self.message = ""  # Clear any previous messages
            self.draw_board()

        winner = self.game.get_winner()
        if winner == 1:
            message = "Black wins!"
        elif winner == -1:
            message = "White wins!"
        else:
            message = "It's a tie!"

        with open(f'./game-log/{self.log_file}', 'a') as file:
            score_diff = sum(row.count(1) for row in self.game.board) - sum(row.count(-1) for row in self.game.board)
            file.write(f"{message} with Score Difference of {score_diff}\n")
            file.write(f"Average White Execution Time : {avg(execution_time_a, 2)}s\n")
            file.write(f"Average Black Execution Time : {avg(execution_time_b, 2)}s\n")


        self.message = message
        self.draw_board()
        self.end_game_sound.play()  # Play end game sound effect
        pygame.time.delay(5000)  # Display the result for 2 seconds before returning

        # Call the return_to_menu_callback if provided
        if return_to_menu_callback:
            return_to_menu_callback()

def avg(num_list, n):
    if len(num_list) > 0:
        return round(sum(num_list)/len(num_list), n)
    return 0
    
def run_game():
    """
    Start and run the Othello game.
    """
    othello_gui = OthelloGUI()
    othello_gui.run_game()
