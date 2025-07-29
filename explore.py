import chess.pgn
import chess.engine
import pandas as pd
import numpy as np
from statistics import mean, variance
import os
import pickle

# Change this on your own :)
pgn_file_path = "D:/lichess_db_standard_rated_2025-06.pgn/lichess_db_standard_rated_2025-06.pgn"
stockfish_path = "D:/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
progress_file = "data/progress.pkl"
output_file = "game_features_with_cheating.parquet"
elo_ratings_file = "data/elo_ratings.pkl"
max_games = 1000000 # For testing

engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

def parse_clock_time(clock):
    """Convert clock time (e.g., '0:03:00') to seconds."""
    try:
        h, m, s = map(int, clock.split(":"))
        return h * 3600 + m * 60 + s
    except:
        return 0

def analyze_game(game):
    """Extract features from a single game."""
    board = game.board()
    moves = list(game.mainline_moves())
    clock_times = []
    move_cpls = []
    move_matches = []
    
    for node in game.mainline():
        move = node.move
        if move:
            clock = node.comment if node.comment else ""
            clock_time = clock.split("[%clk ")[1].split("]")[0] if "[%clk" in clock else "0:00:00"
            clock_times.append(parse_clock_time(clock_time))
            
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            best_move = info["pv"][0]
            eval_score = info["score"].relative.score(mate_score=10000) / 100.0
            move_cpls.append(abs(eval_score))
            move_matches.append(move == best_move)
            
            board.push(move)
    
    avg_cpl = mean(move_cpls) if move_cpls else 0
    move_match_rate = sum(move_matches) / len(move_matches) if move_matches else 0
    time_variance = variance(clock_times) if len(clock_times) > 1 else 0
    avg_time = mean(clock_times) if clock_times else 0
    
    headers = game.headers
    white_elo = int(headers.get("WhiteElo", 0))
    black_elo = int(headers.get("BlackElo", 0))
    
    return {
        "game_id": headers.get("Site", "").split("/")[-1],
        "white_elo": white_elo,
        "black_elo": black_elo,
        "avg_elo": (white_elo + black_elo) / 2,
        "result": headers.get("Result", ""),
        "avg_cpl": avg_cpl,
        "move_match_rate": move_match_rate,
        "time_variance": time_variance,
        "avg_time": avg_time,
        "num_moves": len(moves)
    }

def save_progress(file_position, game_count, games_data):
    """Save processing progress to a file."""
    with open(progress_file, "wb") as f:
        pickle.dump({"file_position": file_position, "game_count": game_count}, f)
    df = pd.DataFrame(games_data)
    df.to_parquet(output_file, index=False, compression="snappy")

def load_progress():
    """Load processing progress from a file."""
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            progress = pickle.load(f)
        return progress["file_position"], progress["game_count"]
    return 0, 0

def load_elo_ratings():
    """Load Elo ratings if they exist."""
    if os.path.exists(elo_ratings_file):
        with open(elo_ratings_file, "rb") as f:
            return pickle.load(f)
    return []

def save_elo_ratings(elo_ratings):
    """Save Elo ratings to a file."""
    with open(elo_ratings_file, "wb") as f:
        pickle.dump(elo_ratings, f)

elo_ratings = load_elo_ratings()
elo_bins = None
if not elo_ratings:
    with open(pgn_file_path, "r", encoding="utf-8") as pgn:
        game_count = 0
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if len(list(game.mainline_moves())) >= 10 and game.headers.get("Event", "").startswith("Rated"):
                headers = game.headers
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
                avg_elo = (white_elo + black_elo) / 2
                elo_ratings.append(avg_elo)
                game_count += 1
                if game_count % 10000 == 0:
                    print(f"Collected Elo for {game_count} games")
                    save_elo_ratings(elo_ratings)

    num_bins = 10
    elo_bins = pd.qcut(elo_ratings, q=num_bins, duplicates="drop").categories
    print("Dynamic Elo bins:", [f"{int(b.left)}-{int(b.right)}" for b in elo_bins])
    save_elo_ratings(elo_ratings)
else:
    num_bins = 10
    elo_bins = pd.qcut(elo_ratings, q=num_bins, duplicates="drop").categories
    print("Loaded Elo bins:", [f"{int(b.left)}-{int(b.right)}" for b in elo_bins])


games_data = []
start_position, start_game_count = load_progress()
with open(pgn_file_path, "r", encoding="utf-8") as pgn:
    pgn.seek(start_position)
    game_count = start_game_count
    chunk_size = 1000
    try:
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if len(list(game.mainline_moves())) >= 10 and game.headers.get("Event", "").startswith("Rated"):
                game_features = analyze_game(game)
                games_data.append(game_features)
                game_count += 1
                if game_count % chunk_size == 0:
                    save_progress(pgn.tell(), game_count, games_data)
                    print(f"Processed {game_count} games, saved progress")
                    games_data = []  # Clear to save memory
            if game_count % 10000 == 0:
                print(f"Processed {game_count} games")
    except KeyboardInterrupt:
        print("Paused processing, saving progress...")
        save_progress(pgn.tell(), game_count, games_data)
        games_data = []
        print("Progress saved. Rerun the script to resume.")
        engine.quit()
        exit()

if games_data:
    save_progress(pgn.tell(), game_count, games_data)

df = pd.read_parquet(output_file)

def assign_elo_bin(avg_elo):
    for bin_range in elo_bins:
        if bin_range.left <= avg_elo <= bin_range.right:
            return f"{int(bin_range.left)}-{int(bin_range.right)}"
    return "Other"

df["elo_bin"] = df["avg_elo"].apply(assign_elo_bin)

bin_stats = df.groupby("elo_bin").agg({
    "avg_cpl": ["mean", "std"],
    "move_match_rate": ["mean", "std"]
}).reset_index()

def flag_outliers(row):
    bin_data = bin_stats[bin_stats["elo_bin"] == row["elo_bin"]]
    if bin_data.empty:
        return False
    cpl_mean = bin_data["avg_cpl"]["mean"].iloc[0]
    cpl_std = bin_data["avg_cpl"]["std"].iloc[0]
    match_mean = bin_data["move_match_rate"]["mean"].iloc[0]
    match_std = bin_data["move_match_rate"]["std"].iloc[0]
    
    is_outlier = (row["avg_cpl"] < cpl_mean - 2 * cpl_std) or \
                 (row["move_match_rate"] > match_mean + 2 * match_std)
    return is_outlier

df["is_cheating"] = df.apply(flag_outliers, axis=1)

df.to_parquet(output_file, index=False, compression="snappy")
print(f"Flagged {df['is_cheating'].sum()} games as potential cheating")

engine.quit()