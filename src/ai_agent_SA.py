import random
import time
import numpy as np
from othello_game import OthelloGame

#* Initial Parameter for Simulated Annealing
INITIAL_TEMPERATURE = 100
COOLING_RATE = 0.95
MAX_ITERATIONS = 150  #* Max number iteration
MIN_TEMPERATURE = 0.1   #* Temperature Termination
MAX_MOVES = 150  
MAX_TIME = 5 #* 5 Second

class SimulatedAnnealingOthelloAI:
    def __init__(self, max_time=MAX_TIME):
        #print("Initializing SimulatedAnnealingOthelloAI...")
        self.max_time = max_time

    def create_individual(self):
        """ Create random initialization for individual """
        return {feature: random.uniform(-10, 10) for feature in ["coin_parity", "mobility", "corner_occupancy", "stability", "edge_occupancy"]}

    def evaluate_game_state(self, game, weights):
        """ Evaluate based on utility function """
        coin_parity = sum(row.count(game.current_player) for row in game.board) - sum(row.count(-game.current_player) for row in game.board)
        player_moves = len(game.get_valid_moves())
        game.current_player *= -1
        opponent_moves = len(game.get_valid_moves())
        game.current_player *= -1
        mobility = player_moves - opponent_moves
        corner_occupancy = sum(game.board[i][j] == game.current_player for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)])
        stability = self.calculate_stability(game)
        edge_occupancy = sum(game.board[i][j] == game.current_player for i in [0, 7] for j in range(1, 7)) + sum(game.board[i][j] == game.current_player for i in range(1, 7) for j in [0, 7])

        score = (weights["coin_parity"] * coin_parity +
                 weights["mobility"] * mobility +
                 weights["corner_occupancy"] * corner_occupancy +
                 weights["stability"] * stability +
                 weights["edge_occupancy"] * edge_occupancy)
        
        return score

    def calculate_stability(self, game):
        """Calculate stability which one of the parameter in utility function"""
        def neighbors(row, col):
            return [(row + dr, col + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8]
        
        def is_stable(row, col):
            return all(game.board[r][c] == game.current_player for r, c in neighbors(row, col)) if game.board[row][col] != 0 else False

        return sum(is_stable(row, col) for row in range(8) for col in range(8) if game.board[row][col] == game.current_player)

    def simulated_annealing(self, game):
        """Start Simulated Annealing Algorithm"""
        current_weights = self.create_individual() #* Define individual weight state
        current_fitness = self.calculate_fitness(game, current_weights) #* Calculate Fitness
        temperature = INITIAL_TEMPERATURE
        start_time = time.time()
        while temperature > MIN_TEMPERATURE:
            for _ in range(MAX_ITERATIONS):
                if (time.time()-start_time>MAX_TIME):
                    print("Time limit Exceed")
                    return current_weights
                new_weights = self.generate_neighbor(current_weights)
                new_fitness = self.calculate_fitness(game, new_weights)

                if self.acceptance_probability(current_fitness, new_fitness, temperature) > random.random():
                    current_weights = new_weights
                    current_fitness = new_fitness

            temperature *= COOLING_RATE

        return current_weights

    def calculate_fitness(self, game, weights):
        """Evaluates a board state based on the given weights."""
        return self.evaluate_game_state(game, weights)

    def generate_neighbor(self, weights):
        """Generate 1 neighbor based on current state"""
        neighbor = weights.copy()
        feature = random.choice(list(weights.keys()))
        neighbor[feature] += random.uniform(-1, 1)
        neighbor[feature] = np.clip(neighbor[feature], -10, 10)
        return neighbor

    def acceptance_probability(self, old_fitness, new_fitness, temperature):
        if new_fitness > old_fitness:
            return 1.0
        return np.exp((new_fitness - old_fitness) / temperature)

    def get_best_move(self, game):
        """Choose the best move"""
        optimized_weights = self.simulated_annealing(game)
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None  
        best_move = None
        best_score = float('-inf') #* Define from - infinity
        for move in valid_moves:
            simulated_game = self.simulate_move(game, move)
            move_score = self.evaluate_game_state(simulated_game, optimized_weights)
            if move_score > best_score:
                best_score = move_score
                best_move = move
        return best_move

    def simulate_move(self, game, move):
        """Simulate the game"""
        simulated_game = OthelloGame(player_mode=game.player_mode)
        simulated_game.board = [row[:] for row in game.board]
        simulated_game.current_player = game.current_player
        simulated_game.make_move(*move)
        return simulated_game
