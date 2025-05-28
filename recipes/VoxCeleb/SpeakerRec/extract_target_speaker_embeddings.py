import os
import glob
import torchaudio
from speechbrain.inference.encoders import MelSpectrogramEncoder
from speechbrain.utils.fetching import fetch
from speechbrain.utils.data_utils import split_path
import torch

spk_emb_encoder = MelSpectrogramEncoder.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb-mel-spec")

train_dir = '/home/ubuntu/VoiceFilter/output/train'
test_dir = '/home/ubuntu/VoiceFilter/output/test'
input_wav = '*-norm.wav'
dvec = '*-dvec.txt'
ecapa_embedding = '*-ecapa-tdnn_embedding.pt'
data_dir = train_dir #train_dir

def find_all(file_format):
    print(os.path.join(data_dir, file_format))
    return sorted(glob.glob(os.path.join(data_dir, file_format)))

dvec_list = find_all(dvec)
ecapa_list = find_all(ecapa_embedding)

print(len(dvec_list), dvec_list[0]) # 99999
print(len(ecapa_list), ecapa_list[0]) # 99999

dvec_ids = [dvec.split('/')[-1] for dvec in dvec_list]
all_ecapa_ids = [dvec.split('/')[-1].split('-')[0]+ecapa_embedding[1:] for dvec in dvec_list]
ecapa_ids = [ecapa.split('/')[-1] for ecapa in ecapa_list]
print(len(dvec_ids), dvec_ids[0]) # 99999
print(len(all_ecapa_ids), all_ecapa_ids[0]) # 99999
print(len(ecapa_ids), ecapa_ids[0]) # 99999   

remaining_ecapa_ids = list(set(all_ecapa_ids)^set(ecapa_ids))
print(len(remaining_ecapa_ids)) # 99999


for idx in remaining_ecapa_ids: #range(len(dvec_list)):
    # break
    embedding_save_path = idx #dvec_list[idx].replace('-dvec.txt', '-ecapa-tdnn_embedding.pt').split('/')[-1]
    print(embedding_save_path)

    if embedding_save_path not in os.listdir(data_dir):
        embedding_save_path = os.path.join(data_dir,embedding_save_path)

        # print(os.listdir(data_dir)[:6], embedding_save_path)
        # break 
        dvec_txt = os.path.join(data_dir, 
                                    idx.replace('-ecapa-tdnn_embedding.pt', '-dvec.txt')
                                )     
        with open(dvec_txt, 'r') as f:#, 'r') as f:
            #   print(self.dvec_list[idx])
            dvec_path = f.readline().strip()

            dvec_path = os.path.join('/home/ubuntu/VoiceFilter/', dvec_path)

            print(dvec_path)    

        INPUT_SPEECH = dvec_path #"/home/ubuntu/Ahad_Signature.wav"# 
        source, fl = split_path(INPUT_SPEECH)
        print(f"Fetching {fl} from {source}")
        path = fetch(fl, source=source, savedir="tmpdir")
        ref_signal, fs_file = torchaudio.load(path)
        spk_embedding = spk_emb_encoder.encode_waveform(ref_signal)

        print(f"Speaker embedding shape: {spk_embedding.shape}")
        # print(f"Speaker embedding: {spk_embedding}") 
        print(f"Saving {embedding_save_path}")
        # Save the embedding as a .pt file for later use in a PyTorch model
        torch.save(spk_embedding, embedding_save_path)

    else:
        print(f"Speaker embedding already exists: skipping {embedding_save_path}")    
        # if idx == 1:
        #     break