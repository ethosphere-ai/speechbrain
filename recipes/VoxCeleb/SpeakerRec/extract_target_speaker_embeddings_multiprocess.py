import os
import glob
import torchaudio
from speechbrain.inference.encoders import MelSpectrogramEncoder
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import functools
import time

# Configuration
train_dir = '/home/ubuntu/VoiceFilter/output/all_variations_w_srcid/train'
test_dir = '/home/ubuntu/VoiceFilter/output/all_variations_w_srcid/test'
input_wav = '*-norm.wav'
dvec = '*-dvec.txt'
ecapa_embedding = '*-ecapa-tdnn_embedding.pt'
data_dir = train_dir

# Multiprocessing configuration
NUM_WORKERS = min(cpu_count(), 64)  # Adjust based on your system
print(f"Using {NUM_WORKERS} worker processes")

def init_worker():
    """Initialize each worker process with its own model instance"""
    global spk_emb_encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")
    spk_emb_encoder = spk_emb_encoder.to(device)
    print(f"Worker initialized with device: {device}")

def process_single_file(args):
    """Process a single audio file - designed for multiprocessing"""
    dvec_file, data_dir = args
    
    try:
        # Generate embedding path
        embedding_name = os.path.basename(dvec_file).replace('-dvec.txt', '-ecapa-tdnn_embedding.pt')
        embedding_path = os.path.join(data_dir, embedding_name)
        
        # Skip if already exists
        if os.path.exists(embedding_path):
            return f"Skipped (exists): {embedding_name}"
        
        # Read audio path from dvec file
        with open(dvec_file, 'r') as f:
            dvec_path = f.readline().strip()
            dvec_path = dvec_path.replace("./data", "/home/ubuntu/VoiceFilter/data")
        
        # Check if audio file exists
        if not os.path.exists(dvec_path):
            return f"Error: Audio not found: {dvec_path}"
        
        # Load audio
        ref_signal, fs_file = torchaudio.load(dvec_path)
        
        # Convert to mono if stereo
        if ref_signal.shape[0] > 1:
            ref_signal = torch.mean(ref_signal, dim=0, keepdim=True)
        
        # Move to appropriate device
        device = next(spk_emb_encoder.parameters()).device
        ref_signal = ref_signal.to(device)
        
        # Generate embedding
        with torch.no_grad():
            spk_embedding = spk_emb_encoder.encode_waveform(ref_signal)
        
        # Save embedding
        torch.save(spk_embedding.cpu(), embedding_path)
        
        return f"Processed: {embedding_name}"
        
    except Exception as e:
        return f"Error processing {os.path.basename(dvec_file)}: {str(e)}"

def find_all(file_format):
    """Find all files matching the pattern"""
    return sorted(glob.glob(os.path.join(data_dir, file_format)))

def get_unprocessed_files():
    """Get list of files that need processing"""
    # Find all dvec files
    dvec_list = find_all(dvec)
    ecapa_list = find_all(ecapa_embedding)
    
    print(f"Found {len(dvec_list)} dvec files")
    print(f"Found {len(ecapa_list)} existing embedding files")
    
    # Find files that need processing
    dvec_ids = [os.path.basename(dvec) for dvec in dvec_list]
    all_ecapa_ids = [dvec_id.replace("-dvec.txt", "") + "-ecapa-tdnn_embedding.pt" for dvec_id in dvec_ids]
    ecapa_ids = [os.path.basename(ecapa) for ecapa in ecapa_list]
    
    # Get remaining files to process
    remaining_ecapa_ids = set(all_ecapa_ids) - set(ecapa_ids)
    remaining_dvec_files = [
        os.path.join(data_dir, ecapa_id.replace('-ecapa-tdnn_embedding.pt', '-dvec.txt'))
        for ecapa_id in remaining_ecapa_ids
        if os.path.exists(os.path.join(data_dir, ecapa_id.replace('-ecapa-tdnn_embedding.pt', '-dvec.txt')))
    ]
    
    print(f"Need to process {len(remaining_dvec_files)} files")
    return remaining_dvec_files

def main():
    """Main processing function with multiprocessing"""
    start_time = time.time()
    
    # Get unprocessed files
    remaining_dvec_files = get_unprocessed_files()
    
    if len(remaining_dvec_files) == 0:
        print("All files already processed!")
        return
    
    # Prepare arguments for multiprocessing
    args_list = [(dvec_file, data_dir) for dvec_file in remaining_dvec_files]
    
    # Process files in parallel
    print(f"Starting parallel processing with {NUM_WORKERS} workers...")
    
    with Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc="Processing embeddings"
        ))
    
    # Count results
    processed = sum(1 for r in results if r.startswith("Processed:"))
    skipped = sum(1 for r in results if r.startswith("Skipped"))
    errors = sum(1 for r in results if r.startswith("Error"))
    
    elapsed_time = time.time() - start_time
    
    print(f"\nCompleted in {elapsed_time:.2f} seconds!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    
    # Print error details if any
    if errors > 0:
        print("\nError details:")
        for result in results:
            if result.startswith("Error"):
                print(f"  {result}")

if __name__ == "__main__":
    main()
