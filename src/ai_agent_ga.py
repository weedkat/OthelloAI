import random
import time
import json
import os
from multiprocessing import Pool
from othello_game import OthelloGame
from ai_agent import MinimaxOthelloAI
from ai_agent_SA import SimulatedAnnealingOthelloAI

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
N_POPULATION = 16
MUTATION_RATE = 0.1
MUTATION_SWING = 1
N_GENERATIONS = 6
ELITE_PERCENTAGE = 0.1
MAX_WEIGHT = 10
MIN_WEIGHT = -10

def save_weights_to_json(weights, filename = 'ga_weights.json'):
    with open(filename, 'w') as file:
        json.dump(weights, file, indent = 2)

def load_weights_from_json(filename = 'ga_weights.json'):
    if not os.path.exists(filename):
        save_weights_to_json({})
    with open(filename, 'r') as file:
        weights = json.load(file)
    return weights

class GeneticOthelloAI:
    def __init__(self, opponent = 'minimaxV3', train = False):
        self.opponent = opponent
        self.train = train
        # Load weight sesuai dengan lawan, jika belum ada, maka lakukan training
        weights = load_weights_from_json()
        if not weights.get(opponent) or train:       
            weights[opponent] =  self.genetic_algorithm()
            self.best_weights = weights[opponent]
            save_weights_to_json(weights)
        else :
            self.best_weights = weights.get(opponent)
    
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

        evaluation = (
            coin_parity * weights['coin_parity']
            + mobility * weights['mobility']
            + corner_occupancy * weights['corner_occupancy']
            + stability * weights['stability']
            + heuristic_pos * weights['heuristic_pos']
            + edge_occupancy * weights['edge_occupancy']
        )
        return evaluation
    
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
    
    def calculate_fitness(self, individual):
        fitness = 0
        game = OthelloGame(player_mode = 'ai')
        move_count = 0
        for turn in [-1, 1]:
            # Simulate games with opponent
            while not game.is_game_over():
                if game.current_player == turn:
                    move = self.get_best_move(game, individual)
                else:
                    move = self.get_opponent_move(game, self.opponent)
                if move:
                    move_count += 1
                    game.make_move(*move)
            # score difference
            score_diff = sum(row.count(turn) for row in game.board)
            score_diff -= sum(row.count(-turn) for row in game.board)
            winner = game.get_winner()
            fitness += (3 if winner == 1 else (0 if winner == 0 else -1)) + 0.05 * score_diff - 0.005 * move_count
        return fitness
    
    def calculate_fitnesses(self, population):
        with Pool() as pool:
            fitnesses = pool.map(self.calculate_fitness, population)
        return fitnesses
    
    def get_opponent_move(self, game, opponent):
        if opponent == 'heuristic':
            return self.heuristic_move(game)
        elif opponent == 'random':
            return self.random_move(game)
        elif opponent == 'minimax':
            return self.minimax_move(game)
        elif opponent == 'minimaxV3':
            return self.minimaxV3_move(game)
        elif opponent == 'simulated_annealing':
            return self.simulated_annealing_move(game)
        else:
            raise Exception('Opponent ia not listed')
        
    def heuristic_move(self, game):
        max_value = float('-inf')
        best_move = None
        for move in game.get_valid_moves():
            row, col = move
            heuristic_value = HEURISTIC_MATRIX[row][col]
            if heuristic_value > max_value:
                best_move = move
                max_value = heuristic_value
        return best_move

    def random_move(self, game):
        return random.choice(game.get_valid_moves())
    
    def minimax_move(self, game, max_depth = 8):
        minimax_agent = MinimaxOthelloAI(max_depth)
        best_move = minimax_agent.get_best_move(game)
        print(best_move)
        return best_move
    
    def minimaxV3_move(self, game):
        minimax_agent = MinimaxOthelloAI()
        best_move = minimax_agent.get_best_move(game)
        print(best_move)
        return best_move
    
    def simulated_annealing_move(self, game):
        sa_agent = SimulatedAnnealingOthelloAI()
        best_move = sa_agent.get_best_move(game)
        print(best_move)
        return best_move

    def roulette_wheel(self, population, fitnesses):
        # Pick one individual with weighted probability
        total_fitness = sum(fitnesses)
        selection_probs = [f / total_fitness for f in fitnesses]
        return random.choices(population, weights=selection_probs, k=1)[0]

    def crossover(self, parent_a, parent_b):
        # get random genetics from parents
        child = {}
        for feature in parent_a:
            rand = random.random()
            child[feature] = parent_a[feature] * rand + parent_b[feature] * (1-rand)
        return child

    def mutate(self, individual):
        # mutate weight between -1 and 1
        for feature in individual:
            if random.random() < MUTATION_RATE:
                individual[feature] += random.uniform(-MUTATION_SWING, MUTATION_SWING)
                individual[feature] = max(MIN_WEIGHT, min(MAX_WEIGHT, individual[feature]))
    
    def genetic_algorithm(self):
        population = [self.create_individual() for _ in range(N_POPULATION)]
        elite_count = max(1, int(N_POPULATION * ELITE_PERCENTAGE))
        for gen in range(N_GENERATIONS):
            # elitism
            fitnesses = self.calculate_fitnesses(population)
            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_count]
            new_population = [elite[0] for elite in elites]

            while len(new_population) < N_POPULATION:
                parent_a = self.roulette_wheel(population, fitnesses)
                parent_b = self.roulette_wheel(population, fitnesses)
                child = self.crossover(parent_a, parent_b)
                self.mutate(child)
                new_population.append(child)
            print(f"Generation {gen + 1} completed with max fitness of {max(fitnesses)}")
            population = new_population

        fitnesses = self.calculate_fitnesses(population)
        return population[fitnesses.index(max(fitnesses))]
    
    def get_best_move(self, game, weights = None):
        valid_moves = game.get_valid_moves()

        best_move = None
        best_eval = float('-inf')
        
        for move in valid_moves:
            # simulate game
            new_game = OthelloGame(player_mode=game.player_mode)
            new_game.board = [row[:] for row in game.board]
            new_game.current_player = game.current_player
            new_game.make_move(*move)

            move_eval = self.evaluate_game_state(new_game, weights or self.best_weights)

            if move_eval > best_eval:
                best_eval = move_eval
                best_move = move

        return best_move
