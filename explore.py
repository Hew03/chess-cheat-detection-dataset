import chess.pgn
import chess.engine
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
from contextlib import contextmanager

# File paths and constants
pgn_file_path = "D:/lichess_db_standard_rated_2025-06.pgn/lichess_db_standard_rated_2025-06.pgn"
stockfish_path = "D:/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
progress_file = "data/progress.pkl"
output_file = "data/game_features_with_cheating.parquet"
elo_ratings_file = "data/elo_ratings.pkl"
max_games = 1000000
chunk_size = 100

@contextmanager
def stockfish_engine(stockfish_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        yield engine
    finally:
        engine.quit()

def parse_clock_time(clock):
    try:
        match = re.search(r"\[%clk (\d+):(\d{2}):(\d{2})\]", clock)
        if match:
            h, m, s = map(int, match.groups())
            return h * 3600 + m * 60 + s
        return 0
    except:
        return 0

def analyze_game(game, engine):
    if not game or not game.headers.get("Site"):
        return None
    board = game.board()
    moves = list(game.mainline_moves())
    clock_times = []
    move_cpls = []
    move_matches = []
    
    for node in tqdm(game.mainline(), total=len(moves), desc="Analyzing moves", leave=False):
        move = node.move
        if move:
            clock = node.comment if node.comment else ""
            clock_time = clock.split("[%clk ")[1].split("]")[0] if "[%clk" in clock else "0:00:00"
            clock_times.append(parse_clock_time(clock_time))
            info = engine.analyse(board, chess.engine.Limit(depth=12), multipv=3)
            best_moves = [pv["pv"][0] for pv in info]
            eval_score = info[0]["score"].relative.score(mate_score=10000) / 100.0
            move_cpls.append(abs(eval_score))
            move_matches.append(move in best_moves)
            board.push(move)
    
    return {
        "game_id": game.headers.get("Site", "").split("/")[-1],
        "white_elo": int(game.headers.get("WhiteElo", 0)),
        "black_elo": int(game.headers.get("BlackElo", 0)),
        "avg_elo": (int(game.headers.get("WhiteElo", 0)) + int(game.headers.get("BlackElo", 0))) / 2,
        "result": game.headers.get("Result", ""),
        "avg_cpl": np.mean(move_cpls) if move_cpls else 0,
        "move_match_rate": sum(move_matches) / len(move_matches) if move_matches else 0,
        "time_variance": np.var(clock_times) if len(clock_times) > 1 else 0,
        "avg_time": np.mean(clock_times) if clock_times else 0,
        "num_moves": len(moves)
    }

def analyze_game_parallel(game, engine_path):
    with stockfish_engine(engine_path) as engine:
        return analyze_game(game, engine)

def save_progress(file_position, game_count, games_data):
    try:
        progress_dir = os.path.dirname(progress_file) or "."
        os.makedirs(progress_dir, exist_ok=True)
        temp_file = progress_file + ".tmp"
        with open(temp_file, "wb") as f:
            pickle.dump({"file_position": file_position, "game_count": game_count}, f)
        os.replace(temp_file, progress_file)
        if games_data:
            df_new = pd.DataFrame(games_data)
            temp_output = output_file + ".tmp"
            if os.path.exists(output_file):
                df_existing = pd.read_parquet(output_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_parquet(temp_output, index=False, compression="snappy")
            else:
                df_new.to_parquet(temp_output, index=False, compression="snappy")
            os.replace(temp_output, output_file)
    except Exception as e:
        print(f"Error saving progress: {e}")
        raise

def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            progress = pickle.load(f)
        return progress["file_position"], progress["game_count"]
    return 0, 0

def load_elo_ratings():
    if os.path.exists(elo_ratings_file):
        with open(elo_ratings_file, "rb") as f:
            return pickle.load(f), True
    return [], False

def save_elo_ratings(elo_ratings):
    with open(elo_ratings_file, "wb") as f:
        pickle.dump(elo_ratings, f)

# Load or compute Elo ratings
elo_ratings, is_cached = load_elo_ratings()
if not is_cached:
    with open(pgn_file_path, "r", encoding="utf-8") as pgn:
        game_count = 0
        pbar = tqdm(total=max_games, desc="Collecting Elo ratings")
        while game_count < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if len(list(game.mainline_moves())) >= 10 and game.headers.get("Event", "").startswith("Rated"):
                headers = game.headers
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
                elo_ratings.append((white_elo + black_elo) / 2)
                game_count += 1
                pbar.update(1)
                if game_count % 10000 == 0:
                    save_elo_ratings(elo_ratings)
        pbar.close()
    save_elo_ratings(elo_ratings)

num_bins = 10
elo_bins = pd.qcut(elo_ratings, q=num_bins, duplicates="drop").categories
print("Elo bins:", [f"{int(b.left)}-{int(b.right)}" for b in elo_bins])

# Main processing loop
games_data = []
start_position, start_game_count = load_progress()
with open(pgn_file_path, "r", encoding="utf-8") as pgn:
    pgn.seek(start_position)
    game_count = start_game_count
    pbar = tqdm(total=max_games - start_game_count, desc="Processing games", initial=start_game_count)
    games = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        try:
            while game_count < max_games:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                if len(list(game.mainline_moves())) >= 10 and game.headers.get("Event", "").startswith("Rated"):
                    games.append(game)
                if len(games) >= chunk_size:
                    games_data.extend([g for g in executor.map(lambda g: analyze_game_parallel(g, stockfish_path), games) if g])
                    game_count += len(games)
                    pbar.update(len(games))
                    save_progress(pgn.tell(), game_count, games_data)
                    games_data = []
                    games = []
        except KeyboardInterrupt:
            print("Keyboard interrupt detected, saving progress...")
            save_progress(pgn.tell(), game_count, games_data)
            pbar.close()
            exit()
    if games:
        games_data.extend([g for g in executor.map(lambda g: analyze_game_parallel(g, stockfish_path), games) if g])
        game_count += len(games)
        save_progress(pgn.tell(), game_count, games_data)
    pbar.close()

# Assign Elo bins and flag outliers
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
    return (row["avg_cpl"] < cpl_mean - 2 * cpl_std) or \
           (row["move_match_rate"] > match_mean + 2 * match_std)

df["is_cheating"] = df.apply(flag_outliers, axis=1)
df.to_parquet(output_file, index=False, compression="snappy")
print(f"Flagged {df['is_cheating'].sum()} games as potential cheating")