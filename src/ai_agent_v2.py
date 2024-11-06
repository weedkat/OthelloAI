import time
from othello_game import OthelloGame

class MinimaxV2:
    def __init__(self):
        self.time_limit = 5
        # Cache untuk menyimpan nilai stability
        self.stability_cache = {}

    def get_best_move(self, game):
        # Menghapus cache sebelum memulai pencarian baru
        self.stability_cache.clear()

        self.start_time = time.time()

        # Hitung total disk yang sudah terisi
        total_disks = sum(row.count(1) + row.count(-1) for row in game.board)
        # Hitung valid moves untuk pemain utama
        self.player_valid_moves = len(game.get_valid_moves())
        # Pembagian fase berdasarkan kombinasi total disk dan valid moves
        self.current_phase = "early" if total_disks < 32 and self.player_valid_moves > 8 else "late"
        max_depth = 4 if self.current_phase == "early" else 8

        # Iterative Deepening Search
        for max_depth in range(2, max_depth + 1):
            # Menghitung sisa waktu untuk memeriksa batas waktu
            if time.time() - self.start_time >= self.time_limit:
                break
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
        valid_moves.sort(key=lambda move: self.evaluate_move_heuristic(move), reverse=maximizing_player)
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

    def evaluate_move_heuristic(self, move):
        row, col = move
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        if (row, col) in corners:
            return 100 if self.current_phase == "late" else 50
        near_corners = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7), (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
        if (row, col) in near_corners:
            return -100
        if row == 0 or row == 7 or col == 0 or col == 7:
            return 10 if self.current_phase == "late" else 5
        return 0

    def evaluate_game_state(self, game):
        # Coin parity (difference in disk count)
        player_disk_count = sum(row.count(game.current_player) for row in game.board)
        opponent_disk_count = sum(row.count(-game.current_player) for row in game.board)
        coin_parity = player_disk_count - opponent_disk_count

        # Mobility (number of valid moves for the current player)
        opponent_valid_moves = len(
            OthelloGame(player_mode=-game.current_player).get_valid_moves()
        )
        mobility = self.player_valid_moves - opponent_valid_moves

        # Corner occupancy and influence around corners
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        player_corner_occupancy = sum(1 for i, j in corners if game.board[i][j] == game.current_player)
        opponent_corner_occupancy = sum(1 for i, j in corners if game.board[i][j] == -game.current_player)

        # Near corner penalty
        near_corners = [(0, 1), (1, 0), (1, 1), (0, 6), (1, 6), (1, 7),
                        (6, 0), (6, 1), (7, 1), (6, 6), (6, 7), (7, 6)]
        player_near_corner_penalty = sum(1 for i, j in near_corners if game.board[i][j] == game.current_player)
        opponent_near_corner_penalty = sum(1 for i, j in near_corners if game.board[i][j] == -game.current_player)
        near_corner_penalty = player_near_corner_penalty - opponent_near_corner_penalty

        # Stability (number of stable disks)
        player_stability = self.calculate_stability(game, game.current_player)
        opponent_stability = self.calculate_stability(game, -game.current_player)

        # Edge occupancy (number of player disks on the edges)
        player_edge_occupancy = sum(
            1 for i in [0, 7] for j in range(1, 7) if game.board[i][j] == game.current_player
        ) + sum(
            1 for i in range(1, 7) for j in [0, 7] if game.board[i][j] == game.current_player
        )
        opponent_edge_occupancy = sum(
            1 for i in [0, 7] for j in range(1, 7) if game.board[i][j] == -game.current_player
        ) + sum(
            1 for i in range(1, 7) for j in [0, 7] if game.board[i][j] == -game.current_player
        )

        if self.current_phase == "early":
            weight = (1.5, 3.0, 2.0, -2.5, 1.0, 1.5)
            corner_occupancy = player_corner_occupancy
            stability = player_stability
            edge_occupancy = player_edge_occupancy
        else:
            weight = (3.0, 1.5, 5.0, -1.0, 4.0, 3.0)
            corner_occupancy = player_corner_occupancy - opponent_corner_occupancy
            stability = player_stability - opponent_stability
            edge_occupancy = player_edge_occupancy - opponent_edge_occupancy

        # Combine the factors with the corresponding weights to get the final evaluation value
        evaluation = (
            coin_parity * weight[0]
            + mobility * weight[1]
            + corner_occupancy * weight[2]
            + near_corner_penalty * weight[3]
            + stability * weight[4]
            + edge_occupancy * weight[5]
        )

        return evaluation


    def calculate_stability(self, game, current_player):
        # Tambahkan current_player ke dalam tuple agar cache menyimpan hasil berdasarkan player
        board_tuple = (tuple(map(tuple, game.board)), current_player)
        if board_tuple in self.stability_cache:
            return self.stability_cache[board_tuple]

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
                all(game.board[r][c] == current_player for r, c in neighbors(row, col))
                or (row, col) in edges + corners
            )

        for region in regions:
            for row, col in region:
                if game.board[row][col] == current_player and is_stable_disk(row, col):
                    stable_count += 1

        # Menyimpan hasil dalam cache
        self.stability_cache[board_tuple] = stable_count
        return stable_count