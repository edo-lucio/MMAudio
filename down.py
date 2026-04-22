import os
import soundfile as sf
from datasets import load_dataset
from multiprocessing import Pool

# Use a high-speed scratch directory, NOT your home folder
OUTPUT_DIR = "MMAudio/data/video"
NUM_WORKERS = 8  # Match this to --cpus-per-task in your SLURM script

def save_audio(item):
    try:
        # 16k audio is usually pre-processed, so this is fast
        audio_array = item['audio']['array']
        sampling_rate = item['audio']['sampling_rate']
        
        # Use a unique ID to avoid collisions
        v_id = item.get('youtube_id', item.get('video_id', 'unknown'))
        start = item.get('start_time', 0)
        filename = f"{v_id}_{start}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if not os.path.exists(filepath):
            sf.write(filepath, audio_array, sampling_rate)
    except Exception as e:
        return f"Error: {e}"
    return None

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # streaming=True is CRITICAL for HPC to avoid huge cache files
    dataset = load_dataset("txya900619/vggsound-16k", split="train", streaming=True)
    
    print(f"Starting parallel export on {NUM_WORKERS} cores...")
    
    # We use a Pool to handle the IO-heavy task of writing files
    with Pool(NUM_WORKERS) as pool:
        # Use imap to handle the generator-like nature of streaming datasets
        for result in pool.imap_unordered(save_audio, dataset, chunksize=10):
            if result:
                print(result)