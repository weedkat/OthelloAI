import time
from othello_game import OthelloGame

class MinimaxV1:
    def __init__(self):
        self.time_limit = 5

    def get_best_move(self, game):
        self.start_time = time.time()

        # Menentukan max depth berdasarkan valid moves
        self.player_valid_moves = len(game.get_valid_moves())
        if self.player_valid_moves > 20:
            max_depth = 3
        elif self.player_valid_moves > 10:
            max_depth = 4
        elif self.player_valid_moves > 5:
            max_depth = 5
        else:
            max_depth = 6

        _, best_move = self.alphabeta(game, max_depth)
        return best_move

    def alphabeta(
        self, game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
    ):
        # Memeriksa batas waktu
        if time.time() - self.start_time >= self.time_limit:
            return float("-inf") if maximizing_player else float("inf"), None

        if max_depth == 0 or game.is_game_over():
            return self.evaluate_game_state(game), None

        valid_moves = game.get_valid_moves()
        best_move = None

        if maximizing_player:
            max_eval = float("-inf")

            for move in valid_moves:
                if time.time() - self.start_time >= self.time_limit:
                    return max_eval, best_move
                new_game = OthelloGame(player_mode=game.player_mode)
                new_game.board = [row[:] for row in game.board]
                new_game.current_player = game.current_player
                new_game.make_move(*move)
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

            for move in valid_moves:
                if time.time() - self.start_time >= self.time_limit:
                    return min_eval, best_move
                new_game = OthelloGame(player_mode=game.player_mode)
                new_game.board = [row[:] for row in game.board]
                new_game.current_player = game.current_player
                new_game.make_move(*move)
                eval, _ = self.alphabeta(new_game, max_depth - 1, True, alpha, beta)

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval, best_move

    def evaluate_game_state(self, game):
        # Membagi weight ke dalam 3 fase berdasarkan remaining moves
        remaining_moves = sum(row.count(0) for row in game.board)
        if remaining_moves > 40:
            weight = (0.5, 3.0, 2.0, 1.0, 2.0)
        elif remaining_moves > 15:
            weight = (1.0, 2.5, 4.0, 3.0, 3.0)
        else:
            weight = (2.0, 1.0, 6.0, 5.0, 2.5)

        # Coin parity (difference in disk count)
        player_disk_count = sum(row.count(game.current_player) for row in game.board)
        opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
        coin_parity = player_disk_count - opponent_disk_count

        # Mobility (number of valid moves for the current player)
        opponent_valid_moves = len(
            OthelloGame(player_mode=-game.current_player).get_valid_moves()
        )
        mobility = self.player_valid_moves - opponent_valid_moves

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