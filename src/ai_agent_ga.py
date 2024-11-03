import random
import time
import json
import os
import numpy as np
from othello_game import OthelloGame
from ai_agent import MinimaxOthelloAI

# GA parameters
POPULATION_SIZE = 15
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
GENERATIONS = 5
TOURNAMENT_SIZE = 4
GAMES_PER_INDIVIDUAL = 4

def save_weights_to_json(weights, filename = "ga_weights.json"):
    """Saves the genetic algorithm weights to a JSON file."""
    with open(filename, "w") as file:
        json.dump(weights, file, indent=4)

def load_weights_from_json(filename = "ga_weights.json"):
    """Loads the genetic algorithm weights from a JSON file."""
    if not os.path.exists(filename):
        save_weights_to_json({})
    with open(filename, "r") as file:
        weights = json.load(file)
    return weights

class GeneticOthelloAI:
    def __init__(self, opponent = 'heuristic', max_time = 5, train = True):
        """Initialize the AI agent by running the genetic algorithm to evolve the optimal weights."""
        self.opponent = opponent
        self.max_time = max_time
        self.train = train
        self.best_weights = self.genetic_algorithm()

    def create_individual(self):
        """Creates a random individual with weights within specified ranges."""
        return {feature: random.uniform(-10, 10) for feature in ["coin_parity", "mobility", "corner_occupancy", "stability", "edge_occupancy"]}

    def evaluate_game_state(self, game, weights):
        """Evaluates the current game state based on the given weights."""
        coin_parity = sum(row.count(game.current_player) for row in game.board) - sum(row.count(-game.current_player) for row in game.board)
        player_moves = len(game.get_valid_moves())
        game.current_player *= -1
        opponent_moves = len(game.get_valid_moves())
        game.current_player *= -1
        mobility = player_moves - opponent_moves
        corner_occupancy = sum(game.board[i][j] == game.current_player for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)])
        stability = self.calculate_stability(game)
        edge_occupancy = sum(game.board[i][j] == game.current_player for i in [0, 7] for j in range(1, 7)) + sum(game.board[i][j] == game.current_player for i in range(1, 7) for j in [0, 7])

        return (weights["coin_parity"] * coin_parity +
                weights["mobility"] * mobility +
                weights["corner_occupancy"] * corner_occupancy +
                weights["stability"] * stability +
                weights["edge_occupancy"] * edge_occupancy)

    def calculate_stability(self, game):
        """Calculates stability of the player's disks."""
        def neighbors(row, col):
            return [(row + dr, col + dc) for dr in [-1, 0, 1] for dc in [-1, 0, 1] if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8]
        
        def is_stable(row, col):
            return all(game.board[r][c] == game.current_player for r, c in neighbors(row, col)) if game.board[row][col] != 0 else False

        return sum(is_stable(row, col) for row in range(8) for col in range(8) if game.board[row][col] == game.current_player)

    def calculate_fitness(self, individual):
        """Calculates the fitness by playing games with a heuristic-based opponent."""
        fitness = 0
        n_simulation = 1 if self.opponent == 'minimax' else GAMES_PER_INDIVIDUAL
        for _ in range(n_simulation):
            game = OthelloGame(player_mode="ai")
            while not game.is_game_over():
                if game.current_player == 1:
                    move = self.get_best_move(game, individual)
                else:
                    move = self.get_simple_move(game, move_type = self.opponent) if game.get_valid_moves() else None
                if move:
                    game.make_move(*move)
            winner = game.get_winner()
            score_diff = sum(row.count(1) for row in game.board) - sum(row.count(-1) for row in game.board)
            fitness += (2 if winner == 1 else (0 if winner == 0 else -1)) + 0.1 * score_diff
        return fitness / n_simulation

    def genetic_algorithm(self):
        weights = load_weights_from_json()
        if self.train or not weights.get(self.opponent):
            """Evolves weights using the genetic algorithm."""
            population = [self.create_individual() for _ in range(POPULATION_SIZE)]
            for generation in range(GENERATIONS):
                fitnesses = [self.calculate_fitness(p) for p in population]
                new_population = []
                while len(new_population) < POPULATION_SIZE:
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)
                    child = self.crossover(parent1, parent2)
                    self.mutate(child)
                    new_population.append(child)
                population = new_population
                print(f"Generation {generation + 1} completed with max fitness: {max(fitnesses)}")
            weight = max(population, key=lambda ind: self.calculate_fitness(ind))
            weights[self.opponent] = weight
            save_weights_to_json(weights)
            return weight
        return weights.get(self.opponent)
    
    def heuristic_move(self, valid_moves):
        heuristic_matrix = [ #https://github.com/mdulin2/Othello/blob/master/Heuristic.py
            [95, 10, 80, 75, 75, 80, 10, 95],
            [10, 10, 45, 45, 45, 45, 10, 10],
            [65, 40, 70, 50, 50, 70, 40, 65],
            [60, 40, 40, 40, 40, 40, 40, 60],
            [60, 40, 40, 40, 40, 40, 40, 60],
            [65, 40, 70, 50, 50, 70, 40, 65],
            [10, 10, 45, 45, 45, 45, 10, 10],
            [95, 10, 65, 60, 60, 65, 10, 95]
        ]
        max_value = float('-inf')
        best_move = None
        for move in valid_moves:
            row, col = move
            heuristic_value = heuristic_matrix[row][col]
            if heuristic_value > max_value:
                best_move = move
                max_value = heuristic_value
        return best_move
    
    def random_move(self, valid_move):
        return random.choice(valid_move)
    
    def minimax_move(self, game):
        max_depth = 4
        minimax_agent = MinimaxOthelloAI(max_depth)
        return minimax_agent.get_best_move(game)
                    
    def get_simple_move(self, game, move_type):
        """Generate a simple move for the opponent."""
        valid_moves = game.get_valid_moves()
        if move_type == 'heuristic':
            return self.heuristic_move(valid_moves)
        elif move_type == 'random':
            return self.random_move(valid_moves)
        elif 'minimax':
            return self.minimax_move(game)

    def tournament_selection(self, population, fitnesses):
        """Select an individual using tournament selection."""
        tournament = random.sample(list(zip(population, fitnesses)), TOURNAMENT_SIZE)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1, parent2):
        """Create offspring by crossover of two parents."""
        return {feature: (parent1[feature] + parent2[feature]) / 2 if random.random() < CROSSOVER_RATE else parent1[feature] for feature in parent1}

    def mutate(self, individual):
        """Mutate an individual by randomly altering weights."""
        for feature in individual:
            if random.random() < MUTATION_RATE:
                individual[feature] += random.uniform(-1, 1)
                individual[feature] = np.clip(individual[feature], -10, 10)

    def get_best_move(self, game, weights=None):
        """Selects the best move based on evolved weights."""
        weights = weights or self.best_weights
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None  # No valid moves available
        
        best_move = None
        best_score = float('-inf')
        start_time = time.time()

        for move in valid_moves:
            # Check if the evaluation has exceeded max_time
            if time.time() - start_time > self.max_time:
                print("Timeout reached. Returning the best move found so far.")
                break

            # Evaluate the move
            simulated_game = OthelloGame(player_mode=game.player_mode)
            simulated_game.board = [row[:] for row in game.board]
            simulated_game.current_player = game.current_player
            simulated_game.make_move(*move)
            
            move_score = self.evaluate_game_state(simulated_game, weights)

            # Update the best move if this move's score is better
            if move_score > best_score:
                best_score = move_score
                best_move = move

        return best_move