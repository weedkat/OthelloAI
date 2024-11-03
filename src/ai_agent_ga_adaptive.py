import random
from othello_game import OthelloGame

class GameStateIndividual:
    def __init__(self, game_state, mutation_rate=0.05):
        """Initialize with a specific game state."""
        self.game_state = game_state
        self.fitness = None
        self.mutation_rate = mutation_rate

    def calculate_fitness(self):
        """Calculates fitness based on game state properties."""
        coin_parity = sum(row.count(self.game_state.current_player) for row in self.game_state.board) - \
                      sum(row.count(-self.game_state.current_player) for row in self.game_state.board)
        stability = sum(row.count(self.game_state.current_player) for row in self.game_state.board if row in [0, 7])
        
        # Add a small baseline value to avoid zero fitness issues
        self.fitness = coin_parity + stability + 0.1  # Example fitness function
        return self.fitness

    def mutate(self):
        """Applies mutation with a low probability to introduce rare changes."""
        if random.random() < self.mutation_rate:
            self.apply_rare_mutation()

    def apply_rare_mutation(self):
        """Applies a rare mutation by flipping a strategic piece."""
        strategic_positions = [(0, 0), (0, 7), (7, 0), (7, 7)] + \
                              [(i, 0) for i in range(1, 7)] + [(i, 7) for i in range(1, 7)] + \
                              [(0, j) for j in range(1, 7)] + [(7, j) for j in range(1, 7)]

        opponent_pieces = [(row, col) for row, col in strategic_positions
                           if self.game_state.board[row][col] == -self.game_state.current_player]
        
        if opponent_pieces:
            row, col = random.choice(opponent_pieces)
            self.game_state.board[row][col] = self.game_state.current_player

    @staticmethod
    def crossover(parent1, parent2):
        """Creates a new game state by combining the game states of two parents."""
        new_game = OthelloGame(player_mode=parent1.game_state.player_mode)
        for row in range(8):
            for col in range(8):
                if random.random() < 0.5:
                    new_game.board[row][col] = parent1.game_state.board[row][col]
                else:
                    new_game.board[row][col] = parent2.game_state.board[row][col]
        return GameStateIndividual(new_game)

class GeneticAdaptiveOthelloAI:
    def __init__(self, population_size=15, generations=5):
        self.population_size = population_size
        self.generations = generations

    def evaluate_population(self, population):
        for individual in population:
            individual.calculate_fitness()

    def select_parents(self, population):
        """Select parents based on fitness with weighted probability, with fallback if all fitnesses are zero."""
        fitnesses = [ind.fitness for ind in population]
        
        # Check if all fitness values are zero
        if all(fitness == 0 for fitness in fitnesses):
            # If all fitnesses are zero, select parents randomly without weights
            return random.sample(population, 2)
        
        # Otherwise, select with weighted probability
        try:
            return random.choices(population, k=2, weights=fitnesses)
        except ValueError:
            # Fallback to random sampling if there's an issue with weights
            print("Warning: Fitness weights are invalid, using uniform random selection.")
            return random.sample(population, 2)

    def evolve_population(self, initial_game_state):
        """Runs genetic evolution to find optimal game states from an initial game state."""
        population = [GameStateIndividual(initial_game_state) for _ in range(self.population_size)]
        for _ in range(self.generations):
            self.evaluate_population(population)
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population)
                child1 = GameStateIndividual.crossover(parent1, parent2)
                child2 = GameStateIndividual.crossover(parent1, parent2)
                child1.mutate()
                child2.mutate()
                new_population.extend([child1, child2])
            population = new_population
        self.evaluate_population(population)
        return max(population, key=lambda ind: ind.fitness)

    def get_best_move(self, game):
        """
        Determines the best move by using a genetic algorithm to evolve each possible move's game state.
        
        Args:
            game (OthelloGame): The current game state.
            
        Returns:
            tuple: The best move (row, col) based on the fitness evaluation, or None if no valid moves exist.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None  # No valid moves available

        best_move = None
        best_score = float('-inf')

        for move in valid_moves:
            # Simulate the move to create a new game state
            simulated_game = OthelloGame(player_mode=game.player_mode)
            simulated_game.board = [row[:] for row in game.board]  # Copy board
            simulated_game.current_player = game.current_player
            simulated_game.make_move(*move)  # Apply the move

            # Evolve the simulated game state using genetic algorithm to optimize it
            evolved_individual = self.evolve_population(simulated_game)
            move_score = evolved_individual.fitness

            # Track the best move and score
            if move_score > best_score:
                best_score = move_score
                best_move = move

        return best_move

    def evaluate_game_state(self, game):
        """Evaluates the game state based on heuristics for Othello."""
        coin_parity = sum(row.count(game.current_player) for row in game.board) - sum(row.count(-game.current_player) for row in game.board)
        player_moves = len(game.get_valid_moves())
        game.current_player *= -1
        opponent_moves = len(game.get_valid_moves())
        game.current_player *= -1
        mobility = player_moves - opponent_moves
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        corner_occupancy = sum(1 for i, j in corners if game.board[i][j] == game.current_player)
        stability = sum(1 for row in [0, 7] for piece in game.board[row] if piece == game.current_player) + \
                    sum(1 for col in [0, 7] for row in range(1, 7) if game.board[row][col] == game.current_player)
        
        return (coin_parity * 1.0 + mobility * 1.5 + corner_occupancy * 2.0 + stability * 0.5)
