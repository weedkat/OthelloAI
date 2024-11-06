import random
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from othello_game import OthelloGame
from ai_agent_v2 import MinimaxV2
from ai_agent_v1 import MinimaxV1

# Genetic Algorithm Parameters
HEURISTIC_MATRIX = [ #https://github.com/mdulin2/Othello/blob/master/Heuristic.py
    [1.0, 0.0, 0.7895, 0.7368, 0.7368, 0.7895, 0.0, 1.0],
    [0.0, 0.0, 0.3684, 0.3684, 0.3684, 0.3684, 0.0, 0.0],
    [0.6842, 0.3158, 0.7368, 0.4737, 0.4737, 0.7368, 0.3158, 0.6842],
    [0.6316, 0.3158, 0.3158, 0.3158, 0.3158, 0.3158, 0.3158, 0.6316],
    [0.6316, 0.3158, 0.3158, 0.3158, 0.3158, 0.3158, 0.3158, 0.6316],
    [0.6842, 0.3158, 0.7368, 0.4737, 0.4737, 0.7368, 0.3158, 0.6842],
    [0.0, 0.0, 0.3684, 0.3684, 0.3684, 0.3684, 0.0, 0.0],
    [1.0, 0.0, 0.6842, 0.6316, 0.6316, 0.6842, 0.0, 1.0]
]
MUTATION_RATE = 0.05
MUTATION_SWING = 1
ELITE_PERCENTAGE = 0.1
MAX_WEIGHT = 5
MIN_WEIGHT = -5
MAX_TIME = 5

class GeneticOthelloAI:
    def __init__(self):
        self.best_weights = None
        
    def create_individual(self):
        # Individu direpresentasikan dengan dictionary evaluasi weight
        features = ['coin_parity', 'mobility', 'corner_occupancy', 'stability', 'heuristic_pos', 'edge_occupancy']
        return {feature : random.uniform(MAX_WEIGHT, MIN_WEIGHT) for feature in features}
    
    def evaluate_game_state(self, game, weights):
        turn = game.current_player
        # Perbedaan jumlah keping
        player_disk_count = sum(row.count(turn) for row in game.board)
        opponent_disk_count = sum(row.count(-turn) for row in game.board)
        coin_parity = player_disk_count - opponent_disk_count

        # Keleluasaan jumlah pergerakkan yang diperbolahkan
        player_valid_moves = len(game.get_valid_moves())
        opponent_valid_moves = len(
            OthelloGame(player_mode=-turn).get_valid_moves()
        )
        mobility = player_valid_moves - opponent_valid_moves

        # Apakah corner sebagai posisi strategis sudah terisi
        corner_occupancy = sum(
            game.board[i][j] * turn for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
        )

        # Stability (number of stable disks)
        stability = self.calculate_stability(game)

        # Heuristic berdasarkan value posisi
        heuristic_pos = self.calculate_heuristic_pos(game)

        # Jumlah keping yang di pinggir
        edge_occupancy = sum(game.board[i][j] * turn for i in [0, 7] for j in range(1, 7)) + sum(
            game.board[i][j] * turn for i in range(1, 7) for j in [0, 7]
        )

        return (
            coin_parity * weights['coin_parity']
            + mobility * weights['mobility']
            + corner_occupancy * weights['corner_occupancy']
            + stability * weights['stability']
            + heuristic_pos * weights['heuristic_pos']
            + edge_occupancy * weights['edge_occupancy']
        )
    
    def calculate_heuristic_pos(self, game):
        evaluation = 0
        for row in range(8):
            for col in range(8):
                if game.board[row][col] == game.current_player:
                    evaluation += HEURISTIC_MATRIX[row][col]
                else:
                    evaluation -= HEURISTIC_MATRIX[row][col]
        return evaluation
    
    def calculate_stability(self, game):
        def neighbors(row, col):
            return [
                (row + dr, col + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
            ]
        
        def is_stable_disk(row, col):
            return (
                all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
                or (row, col) in edges + corners
            )
        
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
            (i, j) for i in range(1, 7) for j in [0, 7]
        ]
        inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
        regions = [corners, edges, inner_region]

        stable_count = 0

        for region in regions:
            for row, col in region:
                if game.board[row][col] == game.current_player and is_stable_disk(row, col):
                    stable_count += 1
        return stable_count
    
    def calculate_fitness(self, game, remaining_moves, individual):
        fitness = 0
        move_count = 0
        turn = game.current_player
        # Simulasi game dengan bot lawan
        while not game.is_game_over():
            if game.current_player == turn:
                move = self.get_best_move(game, individual)
            else:
                move = self.get_opponent_move(game, remaining_moves)
                print(move)
            if move:
                move_count += 1
                game.make_move(*move)
        # Perhitungan fitness
        score_diff = sum(row.count(turn) for row in game.board)
        score_diff -= sum(row.count(-turn) for row in game.board)
        winner = game.get_winner()
        fitness += (3 if winner == 1 else (0 if winner == 0 else -3)) + 0.05 * score_diff - 0.005 * move_count
        return fitness
    
    def calculate_fitnesses(self, population, game):
        remaining_moves = sum(row.count(0) for row in game.board)
        with ProcessPoolExecutor() as executor:
            fitness_func = partial(self.calculate_fitness, game, remaining_moves)
            fitnesses = list(executor.map(fitness_func, population))
        return fitnesses
    
    def get_opponent_move(self, game, remaining_moves):
        if remaining_moves <= 10:
            minimax_agent = MinimaxV2()
            return minimax_agent.get_best_move(game)
        elif remaining_moves <= 20:
            minimax_agent = MinimaxV1()
            return minimax_agent.get_best_move(game)
        else:
            return self.get_best_move(game, self.heuristic_weight())

    def roulette_wheel(self, population, fitnesses):
        # Ambil satu individu dengan probabilitas sesuai fitenesses
        total_fitness = sum(fitnesses)
        selection_probs = [f / total_fitness for f in fitnesses]
        return random.choices(population, weights=selection_probs, k=1)[0]

    def crossover(self, parent_a, parent_b):
        # Pewarisan genetik parent ke individu baru
        child = {}
        for feature in parent_a:
            rand = random.random()
            child[feature] = parent_a[feature] * rand + parent_b[feature] * (1-rand)
        return child

    def mutate(self, individual):
        # Mutasi beban dengan rentang -1 dan 1
        for feature in individual:
            if random.random() < MUTATION_RATE:
                individual[feature] += random.uniform(-MUTATION_SWING, MUTATION_SWING)
                individual[feature] = max(MIN_WEIGHT, min(MAX_WEIGHT, individual[feature]))
    
    def genetic_algorithm(self, game, n_population = 6, n_generations = 4):
        start_time = time.time()
        population = [self.create_individual() for _ in range(n_population-1)]
        population.append(self.best_weights or self.heuristic_weight())
        elite_count = max(1, int(n_population * ELITE_PERCENTAGE))
        for gen in range(n_generations):
            # elitism
            fitnesses = self.calculate_fitnesses(population, game)
            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_count]
            new_population = [elite[0] for elite in elites]
            if time.time() - start_time > MAX_TIME:
                print(f"Time limit exceeded")
                return new_population[0]
            while len(new_population) < n_population:
                parent_a = self.roulette_wheel(population, fitnesses)
                parent_b = self.roulette_wheel(population, fitnesses)
                child = self.crossover(parent_a, parent_b)
                self.mutate(child)
                new_population.append(child)
            print(f"Generation {gen + 1} completed with max fitness of {max(fitnesses)}")
            population = new_population

        fitnesses = self.calculate_fitnesses(population, game)
        time_taken = time.time() - start_time
        print(f"Training time : {time_taken:.2f}s")
        return population[fitnesses.index(max(fitnesses))]
    
    def heuristic_weight(self):
        weights = (0.5, 3.0, 2.0, 1.0, 0.5, 2.0)
        features = ['coin_parity', 'mobility', 'corner_occupancy', 'stability', 'heuristic_pos', 'edge_occupancy']
        return {feature : weights[i] for i, feature in enumerate(features)}

    def get_best_move(self, game, weights = None):
        # weights dari input atau mencari lewat genetic algorithm atau weight heuristik jika tidka ada yg terinisialisasi
        if weights is None:
            remaining_moves = sum(row.count(0) for row in game.board)
            if remaining_moves <= 20:
                self.best_weights = self.genetic_algorithm(game)
            elif remaining_moves <= 40:
                self.best_weights = self.genetic_algorithm(game, 24, 8)
            else:
                self.best_weights = self.genetic_algorithm(game, 12, 8)

            weights = self.best_weights or self.heuristic_weight()
            
        valid_moves = game.get_valid_moves()
        best_move = None
        best_eval = float('-inf')
        for move in valid_moves:
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)
        
            move_eval = self.evaluate_game_state(new_game, weights)
            if move_eval > best_eval:
                best_eval = move_eval
                best_move = move

        return best_move
