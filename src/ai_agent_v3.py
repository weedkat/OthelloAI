import time
from othello_game import OthelloGame

class MinimaxV3:
    def __init__(self):
        self.time_limit = 5
        # Cache untuk menyimpan nilai stability
        self.stability_cache = {}

    def get_best_move(self, game):
        # Menghapus cache sebelum memulai pencarian baru
        self.stability_cache.clear()

        self.start_time = time.time()

        # Menentukan fase permainan
        remaining_moves = sum(row.count(0) for row in game.board)
        self.current_phase = "early" if remaining_moves > 40 else "mid" if remaining_moves > 15 else "late"

        valid_moves = game.get_valid_moves()

        # Iterative Deepening Search
        for max_depth in range(2, self.get_depth(len(valid_moves)) + 1):
            # Menghitung sisa waktu untuk memeriksa batas waktu
            if time.time() - self.start_time >= self.time_limit:
                break
            _, best_move = self.alphabeta(game, max_depth)
        return best_move

    def get_depth(self, valid_moves_count):
        if valid_moves_count > 20:
            return 4
        elif valid_moves_count > 10:
            return 5
        elif valid_moves_count > 5:
            return 6
        else:
            return 7

    def alphabeta(
        self, game, max_depth, maximizing_player=True, alpha=float("-inf"), beta=float("inf")
    ):
        # Memeriksa batas waktu
        if time.time() - self.start_time >= self.time_limit:
            return float("-inf") if maximizing_player else float("inf"), None

        if max_depth == 0:
            # Quiescence search untuk mengatasi horizon effect
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
        edge_positions = [(i, j) for i in [0, 7] for j in range(1, 7)]
        central_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        opponent = -game.current_player

        for move in valid_moves:
            flipped_disks = self.count_flipped_disks(game, move, opponent)
            if flipped_disks >= 3 or move in edge_positions or move in central_positions:
                aggressive_moves.append((move, flipped_disks))

        # Prioritas langkah agresif pada masing-masing fase permainan
        if self.current_phase == "early":
            aggressive_moves.sort(key=lambda x: (x[1], x[0] in central_positions), reverse=True)
        elif self.current_phase == "mid":
            aggressive_moves.sort(key=lambda x: (x[1], x[0] in edge_positions), reverse=True)
        else:
            aggressive_moves.sort(key=lambda x: (x[1], x[0] in edge_positions or x[0] in central_positions),
                                  reverse=True)

        return [move for move, _ in aggressive_moves] if aggressive_moves else valid_moves

    def count_flipped_disks(self, game, move, opponent):
        flipped_disks = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            x, y = move[0] + dx, move[1] + dy
            disks_to_flip = 0

            while 0 <= x < 8 and 0 <= y < 8 and game.board[x][y] == opponent:
                disks_to_flip += 1
                x += dx
                y += dy

            if 0 <= x < 8 and 0 <= y < 8 and game.board[x][y] == game.current_player:
                flipped_disks += disks_to_flip

        return flipped_disks

    def move_ordering(self, moves):
        # Posisi-posisi penting pada papan
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        x_squares = [(1, 1), (1, 6), (6, 1), (6, 6)]
        edge_moves = [(i, j) for i in [0, 7] for j in range(1, 7)] + [(i, j) for i in range(1, 7) for j in [0, 7]]
        central_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]

        def evaluate_move(move):
            if self.current_phase == "early":
                if move in corners:
                    return 5
                elif move in central_positions:
                    return 3
                else:
                    return 1 # Langkah lainnya

            elif self.current_phase == "mid":
                if move in corners:
                    return 6
                elif move in edge_moves:
                    return 4
                elif move in x_squares:
                    return -2  # Hindari X-squares
                elif move in central_positions:
                    return 3
                else:
                    return 2 # Langkah lainnya

            else:
                if move in corners:
                    return 7
                elif move in edge_moves:
                    return 5
                elif move in x_squares:
                    return -3  # Hindari X-squares
                elif move in central_positions:
                    return 4
                else:
                    return 2  # Langkah lainnya

        # Urutkan langkah berdasarkan evaluasi
        ordered_moves = sorted(moves, key=evaluate_move, reverse=True)
        return ordered_moves

    def create_game_copy(self, game, move):
        new_game = OthelloGame(player_mode=game.player_mode)
        new_game.board = [row[:] for row in game.board]
        new_game.current_player = game.current_player
        new_game.make_move(*move)
        return new_game

    def evaluate_game_state(self, game):
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
        player_corner_occupancy = sum(
            1 for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)] if game.board[i][j] == game.current_player
        )
        opponent_corner_occupancy = sum(
            1 for i, j in [(0, 0), (0, 7), (7, 0), (7, 7)] if game.board[i][j] == -game.current_player
        )

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

        # Central control
        central_positions = [(3, 3), (3, 4), (4, 3), (4, 4)]
        player_central_control = sum(
            1 for i, j in central_positions if game.board[i][j] == game.current_player
        )
        opponent_central_control = sum(
            1 for i, j in central_positions if game.board[i][j] == -game.current_player
        )

        if self.current_phase == "early":
            weight = (0.5, 3.5, 2.0, 0.5, 0.8, 2.5)
            corner_occupancy = player_corner_occupancy
            stability = player_stability
            edge_occupancy = player_edge_occupancy
            central_control = player_central_control - opponent_central_control
        elif self.current_phase == "mid":
            weight = (1.0, 2.8, 4.0, 3.2, 2.8, 1.8)
            corner_occupancy = player_corner_occupancy - opponent_corner_occupancy
            stability = player_stability - opponent_stability
            edge_occupancy = player_edge_occupancy - opponent_edge_occupancy
            central_control = player_central_control - opponent_central_control
        else:
            weight = (2.0, 1.2, 6.0, 5.0, 1.5, 1.0)
            corner_occupancy = player_corner_occupancy - opponent_corner_occupancy
            stability = player_stability - opponent_stability
            edge_occupancy = player_edge_occupancy - opponent_edge_occupancy
            central_control = player_central_control

        # Combine the factors with the corresponding weights to get the final evaluation value
        evaluation = (
            coin_parity * weight[0]
            + mobility * weight[1]
            + corner_occupancy * weight[2]
            + stability * weight[3]
            + edge_occupancy * weight[4]
            + central_control * weight[5]
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