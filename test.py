import chess.pgn

# Change this on your own :)
pgn_file_path = "D:/lichess_db_standard_rated_2025-06.pgn/lichess_db_standard_rated_2025-06.pgn"

with open(pgn_file_path) as pgn:
    game = chess.pgn.read_game(pgn)

    if game is None:
        print("No games found in the PGN file.")
    else:
        print("Game Headers:")
        for key, value in game.headers.items():
            print(f"{key}: {value}")

        print("\nMoves and Times:")
        board = game.board()
        move_number = 0
        node = game

        for move in game.mainline_moves():
            move_number += 1
            san_move = board.san(move)

            node = node.variations[0] if node.variations else node  
            clock_time = node.comment if node.comment else "No clock data"

            if "[%clk" in clock_time:
                clock_time = clock_time.split("[%clk ")[1].split("]")[0]

            board.push(move)

            if move_number % 2 == 1:
                print(f"{move_number // 2 + 1}. {san_move} ({clock_time})", end=" ")
            else:
                print(f"{san_move} ({clock_time})")

        print("\nFinal Board Position:")
        print(board)
