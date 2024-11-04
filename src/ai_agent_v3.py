import time
from othello_game import OthelloGame

class MinimaxV3:
    def __init__(self):
        self.time_limit = 5

    def get_best_move(self, game):
        valid_moves = game.get_valid_moves()
        best_move = None
        self.start_time = time.time()

        # Iterative Deepening Search
        for max_depth in range(2, max(self.get_depth(len(valid_moves)), 2) + 1):
            # Menghitung sisa waktu untuk memeriksa batas waktu
            if time.time() - self.start_time >= self.time_limit:
                break
            _, best_move = self.alphabeta(game, max_depth)
        return best_move

    def get_depth(self, valid_moves_count):
        if valid_moves_count <= 5:
            return 4
        elif valid_moves_count <= 10:
            return 5
        else: return 4

    def alphabeta(
        self, game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
    ):
        # Memeriksa batas waktu
        if time.time() - self.start_time >= self.time_limit:
            return float("-inf") if maximizing_player else float("inf"), None

        if max_depth == 0:
            ''' Quiescence Search untuk menangani masalah horizon effect 
                (pencarian utama berhenti di kedalaman tertentu,
                tetapi mungkin masih ada langkah yang signifikan).'''
            return self.quiescence_search(game, alpha, beta), None

        elif game.is_game_over():
            return self.evaluate_game_state(game), None

        valid_moves = self.move_ordering(game.get_valid_moves())
        best_move = None

        if maximizing_player:
            max_eval = float("-inf")
            for idx, move in enumerate(valid_moves):
                if idx % 5 == 0 and time.time() - self.start_time >= self.time_limit:
                    return max_eval, best_move
                new_game = self.create_game_copy(game, move)
                eval, _ = self.alphabeta(new_game, max_depth - 1, False, alpha, beta)

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval, best_move
        else:
            min_eval = float("inf")
            for idx, move in enumerate(valid_moves):
                if idx % 5 == 0 and time.time() - self.start_time >= self.time_limit:
                    return min_eval, best_move
                new_game = self.create_game_copy(game, move)
                eval, _ = self.alphabeta(new_game, max_depth - 1, True, alpha, beta)

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def quiescence_search(self, game, alpha, beta, depth=1):
        stand_pat = self.evaluate_game_state(game)
        if time.time() - self.start_time >= self.time_limit or depth > 2:
            return stand_pat
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        for move in self.get_aggressive_moves(game):
            if time.time() - self.start_time >= self.time_limit:
                return alpha
            new_game = self.create_game_copy(game, move)
            score = -self.quiescence_search(new_game, -beta, -alpha, depth + 1)  # Pass `new_game` correctly
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha


    def get_aggressive_moves(self, game):
        aggressive_moves = []
        valid_moves = game.get_valid_moves()
        edge_positions = [(i, j) for i in [0, 7] for j in range(1, 7)] + [(i, j) for i in range(1, 7) for j in [0, 7]]

        for move in valid_moves:
            new_game = self.create_game_copy(game, move)
            opponent_disks_flipped = self.calculate_flipped_disks(new_game)
            if opponent_disks_flipped >= 3 or move in edge_positions:
                aggressive_moves.append(move)

        return aggressive_moves if aggressive_moves else valid_moves

    def calculate_flipped_disks(self, game):
        opponent = -game.current_player
        flipped_disks = 0

        for row in game.board:
            flipped_disks += row.count(opponent)

        return flipped_disks

    def move_ordering(self, moves):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edge_moves = [(i, j) for i in [0, 7] for j in range(1, 7)] + [(i, j) for i in range(1, 7) for j in [0, 7]]
        ordered = sorted(moves, key=lambda move: self.evaluate_quick_move(move, corners, edge_moves), reverse=True)
        return ordered

    def evaluate_quick_move(self, move, corners, edge_moves):
        if move in corners:
            return 4
        elif move in edge_moves:
            return 3
        else:
            return 2

    def create_game_copy(self, game, move):
        new_game = OthelloGame(player_mode=game.player_mode)
        new_game.board = [row[:] for row in game.board]
        new_game.current_player = game.current_player
        new_game.make_move(*move)
        return new_game

    def evaluate_game_state(self, game):
        # Pembagian fase berdasarkan remaining moves untuk menentukan weight
        remaining_moves = sum(row.count(0) for row in game.board)
        if remaining_moves > 40:
            weight = (0.5, 3.0, 3.0, 1.0, 2.5)
        elif 15 < remaining_moves <= 40:
            weight = (1.0, 2.5, 4.0, 3.0, 3.0)
        else:
            weight = (2.5, 1.5, 6.0, 0.5, 2.5)

        # Coin parity (difference in disk count)
        player_disk_count = sum(row.count(game.current_player) for row in game.board)
        opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
        coin_parity = player_disk_count - opponent_disk_count

        # Mobility (number of valid moves for the current player)
        player_valid_moves = len(game.get_valid_moves())
        opponent_valid_moves = len(
            OthelloGame(player_mode=-game.current_player).get_valid_moves()
        )
        mobility = player_valid_moves - opponent_valid_moves

        # Corner occupancy (number of player disks in the corners)
        corner_occupancy = sum(
            game.board[i][j] for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)]
        )

        # Stability (number of stable disks)
        stability = self.calculate_stability(game)

        # Edge occupancy (number of player disks on the edges)
        edge_occupancy = sum(game.board[i][j] for i in [0, 7] for j in range(1, 7)) + sum(
           game.board[i][j] for i in range(1, 7) for j in [0, 7]
        )

        # Combine the factors with the corresponding weights to get the final evaluation value
        evaluation = (
            coin_parity * weight[0]
            + mobility * weight[1]
            + corner_occupancy * weight[2]
            + stability * weight[3]
            + edge_occupancy * weight[4]
        )

        return evaluation


    def calculate_stability(self, game):
        def neighbors(row, col):
            return [
                (row + dr, col + dc)
                for dr in [-1, 0, 1]
                for dc in [-1, 0, 1]
                if (dr, dc) != (0, 0) and 0 <= row + dr < 8 and 0 <= col + dc < 8
            ]

        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(i, j) for i in [0, 7] for j in range(1, 7)] + [
            (i, j) for i in range(1, 7) for j in [0, 7]
        ]
        inner_region = [(i, j) for i in range(2, 6) for j in range(2, 6)]
        regions = [corners, edges, inner_region]

        stable_count = 0

        def is_stable_disk(row, col):
            return (
                all(game.board[r][c] == game.current_player for r, c in neighbors(row, col))
                or (row, col) in edges + corners
            )

        for region in regions:
            for row, col in region:
                if game.board[row][col] == game.current_player and is_stable_disk(row, col):
                    stable_count += 1

        return stable_count