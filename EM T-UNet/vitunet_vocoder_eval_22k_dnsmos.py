import torch
from torch.utils.data import Dataset

import numpy as np
import h5py
import csv
import logging
import os
import yaml
from tqdm import tqdm

import librosa
import soundfile as sf
import pysepm
from pesq import pesq
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import configargparse
from utils.mel_utils import AverageMeter
from parallel_wavegan.utils import load_model
from cnn_transformer.transunet import TransUnet as TransUnet

import onnxruntime as ort


class ComputeDNSMOS:
    """
    DNSMOS Speech Quality Assessment (Local ONNX Model)
    Based on Microsoft DNS-Challenge official implementation (16kHz)
    Input dims: [batch, time_frames=901, freq_bins=161]
    """
    def __init__(self, primary_model_path, sig_model_path=None, sr=16000):
        self.sr = sr
        self.window_length = 320   # 20ms @ 16kHz
        self.hop_length = 160      # 10ms hop (50% overlap)
        self.n_fft = 320           # 161 freq bins (0-160)
        self.expected_frames = 901  # ~10s expected
        
        self.primary_model = ort.InferenceSession(primary_model_path)
        self.sig_model = ort.InferenceSession(sig_model_path) if sig_model_path else None
        
        self.primary_input = self.primary_model.get_inputs()[0].name
        self.primary_outputs = [o.name for o in self.primary_model.get_outputs()]
        
        if self.sig_model:
            self.sig_input = self.sig_model.get_inputs()[0].name
            self.sig_outputs = [o.name for o in self.sig_model.get_outputs()]
            
    def _get_magnitude(self, audio):
        """Compute STFT magnitude, strictly match dims [1, 901, 161]"""
        if len(audio) == 0:
            return np.zeros((1, self.expected_frames, self.n_fft//2 + 1), dtype=np.float32)
            
        # Pad or truncate to 10s (160000 samples @ 16kHz)
        target_length = self.sr * 10
        if len(audio) < target_length:
            repeats = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeats)[:target_length]
        else:
            audio = audio[:target_length]
            
        # Normalize
        audio = audio - np.mean(audio)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        # STFT with model-matching params
        D = librosa.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.window_length,
            window='hann',
            center=True,
            pad_mode='constant'
        )
        
        magnitude = np.abs(D).T  # (T, 161)
        
        # Align time dim to 901 frames
        current_frames = magnitude.shape[0]
        if current_frames < self.expected_frames:
            pad_width = self.expected_frames - current_frames
            magnitude = np.pad(magnitude, ((0, pad_width), (0, 0)), mode='constant')
        elif current_frames > self.expected_frames:
            magnitude = magnitude[:self.expected_frames, :]
            
        # Add batch dim: (1, 901, 161)
        magnitude = np.expand_dims(magnitude, axis=0).astype(np.float32)
        
        return magnitude
    
    def __call__(self, audio, original_sr):
        """Compute DNSMOS scores (SIG, BAK, OVR) range 1-5"""
        if original_sr != self.sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sr)
        
        mag = self._get_magnitude(audio)
        
        # Primary model (BAK, OVR)
        primary_out = self.primary_model.run(None, {self.primary_input: mag})
        primary_scores = primary_out[0][0]
        
        if self.sig_model:
            sig_out = self.sig_model.run(None, {self.sig_input: mag})
            sig_score = float(sig_out[0][0][0])
            bak_score = float(primary_scores[0])
            ovr_score = float(primary_scores[1])
        else:
            if len(primary_scores) >= 3:
                sig_score = float(primary_scores[0])
                bak_score = float(primary_scores[1])
                ovr_score = float(primary_scores[2])
            else:
                sig_score = np.nan
                bak_score = float(primary_scores[0])
                ovr_score = float(primary_scores[1])
        
        return {
            'SIG': sig_score,
            'BAK': bak_score,
            'OVR': ovr_score
        }


class ArgParser(object):
    def __init__(self):
        parser = configargparse.ArgumentParser(
            description="Evaluate to recover speech from radio signal",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add("--config", is_config_file=True, help="config file path")

        # Transformer Unet settings
        parser.add_argument('--hidden_size', help='hidden size', type=int)
        parser.add_argument('--transformer_num_layers', help='transformer number of layers', type=int)
        parser.add_argument('--mlp_dim', help='transformer mlp dim', type=int)
        parser.add_argument('--num_heads', help='transformer number of heads', type=int)
        parser.add_argument('--transformer_dropout_rate', help='transformer dropout rate', type=float)
        parser.add_argument('--transformer_attention_dropout_rate', help='transformer attention dropout rate', type=float)

        parser.add_argument('--audRate', help='audio sample rate', type=int, default=22050)

        # Vocoder configuration
        parser.add_argument('--vocoder_ckpt', help='checkpoint path to recover vocoder')
        parser.add_argument('--vocoder_config', help='vocoder configuration')

        # Evaluation index file
        parser.add_argument('--dataset_name', help='dataset name')
        parser.add_argument('--list_val', help='list of validation data')
        parser.add_argument('--audio_path', help='path of raw audio files')
        parser.add_argument('--load_best_model', help='load from best model')
        parser.add_argument('--save_wave_path', help='path to save generated wave files', type=str, default='examples/alienwarehei-ljspeech/audio')

        # DNSMOS model paths
        parser.add_argument('--dnsmos_primary_model', 
                           help='DNSMOS primary model path (bak_ovr.onnx or sig_bak_ovr.onnx)',
                           type=str, 
                           default='/root/autodl-tmp/DNS-Challenge-master/DNSMOS/DNSMOS/bak_ovr.onnx')
        parser.add_argument('--dnsmos_sig_model', 
                           help='DNSMOS SIG model path (sig.onnx), optional if primary includes SIG',
                           type=str, 
                           default='/root/autodl-tmp/DNS-Challenge-master/DNSMOS/DNSMOS/sig.onnx')
        parser.add_argument('--dnsmos_sr', 
                           help='DNSMOS model sample rate (16000)',
                           type=int, 
                           default=16000)

        self.parser = parser

    def parse_train_arguments(self):
        args = self.parser.parse_args()
        return args


def compute_mcd_22050(ref_audio, pred_audio, sr=22050, n_mfcc=13, debug=False):
    """22050Hz MCD calculation with forced normalization"""
    ref_audio = np.asarray(ref_audio, dtype=np.float32).squeeze()
    pred_audio = np.asarray(pred_audio, dtype=np.float32).squeeze()
    
    # Force normalize to [-1, 1]
    ref_max = np.max(np.abs(ref_audio))
    pred_max = np.max(np.abs(pred_audio))
    
    if ref_max > 0:
        ref_audio = ref_audio / ref_max
    if pred_max > 0:
        pred_audio = pred_audio / pred_max
    
    if debug:
        print(f"  [MCD] Ref: [{ref_audio.min():.3f},{ref_audio.max():.3f}], "
              f"mean={np.mean(np.abs(ref_audio)):.4f}, std={np.std(ref_audio):.4f}")
        print(f"  [MCD] Pred: [{pred_audio.min():.3f},{pred_audio.max():.3f}], "
              f"mean={np.mean(np.abs(pred_audio)):.4f}, std={np.std(pred_audio):.4f}")
    
    min_samples = sr
    if len(ref_audio) < min_samples or len(pred_audio) < min_samples:
        print(f"  [MCD Warning] Audio too short ({len(ref_audio)}/{len(pred_audio)} < {min_samples})")
        return np.nan
    
    frame_length = 551   # 25ms
    hop_length = 220     # 10ms
    n_fft = 1024
    
    try:
        ref_audio = ref_audio - np.mean(ref_audio)
        pred_audio = pred_audio - np.mean(pred_audio)
        ref_audio = librosa.effects.preemphasis(ref_audio, coef=0.97)
        pred_audio = librosa.effects.preemphasis(pred_audio, coef=0.97)
        
        mfcc_ref = librosa.feature.mfcc(
            y=ref_audio, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
            window='hamming', center=True, norm='ortho'
        )
        mfcc_pred = librosa.feature.mfcc(
            y=pred_audio, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length, win_length=frame_length,
            window='hamming', center=True, norm='ortho'
        )
        
        if debug:
            print(f"  [MCD] MFCC Ref: [{mfcc_ref.min():.1f},{mfcc_ref.max():.1f}]")
            print(f"  [MCD] MFCC Pred: [{mfcc_pred.min():.1f},{mfcc_pred.max():.1f}]")
        
        # Exclude first MFCC coefficient
        mfcc_ref = mfcc_ref[1:, :]
        mfcc_pred = mfcc_pred[1:, :]
        
        distance, path = fastdtw(mfcc_ref.T, mfcc_pred.T, dist=euclidean)
        aligned_ref = mfcc_ref.T[np.array(path)[:, 0]]
        aligned_pred = mfcc_pred.T[np.array(path)[:, 1]]
        
        diff = aligned_ref - aligned_pred
        mcd_per_frame = np.sqrt(2 * np.sum(diff ** 2, axis=1))
        mcd = np.mean(mcd_per_frame) * (10 / np.log(10))
        
        if debug:
            print(f"  [MCD] Raw MCD={mcd:.2f} dB")
        
        if np.isnan(mcd) or mcd < 0.1 or mcd > 20:
            print(f"  [MCD Warning] Abnormal value {mcd:.2f} dB, returning NaN")
            return np.nan
            
        return float(mcd)
        
    except Exception as e:
        print(f"  [MCD Error] {str(e)[:100]}")
        return np.nan


def compute_lsd_22050(ref_audio, pred_audio, sr=22050):
    """22050Hz LSD calculation"""
    n_fft = 2048
    hop_length = 512
    win_length = 1024
    
    S_ref = librosa.stft(ref_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    S_pred = librosa.stft(pred_audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    log_S_ref = np.log10(np.abs(S_ref) ** 2 + 1e-10)
    log_S_pred = np.log10(np.abs(S_pred) ** 2 + 1e-10)
    
    min_frames = min(log_S_ref.shape[1], log_S_pred.shape[1])
    log_S_ref = log_S_ref[:, :min_frames]
    log_S_pred = log_S_pred[:, :min_frames]
    
    lsd_per_frame = np.sqrt(np.mean((log_S_ref - log_S_pred) ** 2, axis=0))
    return np.mean(lsd_per_frame)


def compute_pesq_8k_from_22050(ref_audio, pred_audio):
    """Resample 22050Hz to 8kHz for PESQ NB calculation"""
    ref_8k = librosa.resample(ref_audio, orig_sr=22050, target_sr=8000)
    pred_8k = librosa.resample(pred_audio, orig_sr=22050, target_sr=8000)
    
    if len(ref_8k) < 8000 or len(pred_8k) < 8000:
        return np.nan
    
    min_len = min(len(ref_8k), len(pred_8k))
    ref_align = ref_8k[:min_len]
    pred_align = pred_8k[:min_len]
    
    # Cross-correlation alignment
    corr = np.correlate(ref_align, pred_align, mode='full')
    lag = np.argmax(corr) - (len(pred_align) - 1)
    if abs(lag) <= 200:
        if lag > 0:
            pred_align = np.roll(pred_align, lag)
        else:
            ref_align = np.roll(ref_align, -lag)
    
    # Scale to 16-bit range
    ref_align = ref_align / (np.max(np.abs(ref_align)) + 1e-8) * 32767
    pred_align = pred_align / (np.max(np.abs(pred_align)) + 1e-8) * 32767
    
    try:
        score = pesq(8000, ref_align, pred_align, 'nb')
        return score if -0.5 <= score <= 4.5 else np.nan
    except:
        return np.nan


def trans_list(input_csv):
    list_sample = []
    for row in csv.reader(open(input_csv, 'r'), delimiter=','):
        if len(row) < 2:
            continue
        list_sample.append(row)
    print(f"Loaded {len(list_sample)} samples from {input_csv}")
    return list_sample


class radioaudiomelDataset(Dataset):
    def __init__(self, dataset_name, input_path=None, audio_path=None, sampling_rate=22050):
        self.dataset_name = dataset_name
        self.files = trans_list(input_path)
        self.audio_path = audio_path
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        info = self.files[index]
        radio_file = info[0]
        audio_file = info[1]
        mel_len = int(info[2])

        file_folder = None
        if self.dataset_name == 'LJSpeech':
            filename = os.path.basename(radio_file).split('.')[0]
            audio_raw_path = os.path.join(self.audio_path, f'{filename}.wav')
        else:
            file_split = radio_file.lstrip('/').split('/')
            file_folder = file_split[-2]
            filename = file_split[-1].split('.')[0]
            audio_raw_path = os.path.join(self.audio_path, file_folder, f'{filename}.wav')

        with h5py.File(audio_file, 'r') as f:
            audio_melamp = f['mel'][:]
        with h5py.File(radio_file, 'r') as f:
            radio_melamp = f['mel'][:]
        
        audio_raw, sr = librosa.load(audio_raw_path, sr=22050, mono=True)
        
        assert audio_melamp.shape == radio_melamp.shape, f"Mel shape mismatch"
        assert mel_len == audio_melamp.shape[0], f"Mel length mismatch"

        if file_folder is not None:
            file = os.path.join(file_folder, filename)
        else:
            file = filename

        return (file, audio_raw, audio_melamp, torch.FloatTensor(radio_melamp).unsqueeze(0).unsqueeze(0))


def main():
    parser = ArgParser()
    args = parser.parse_train_arguments()

    assert args.audRate == 22050, "Code only supports 22050Hz sample rate"
    print(f"Config: Fixed sample rate=22050Hz")

    val_set = radioaudiomelDataset(args.dataset_name, args.list_val, args.audio_path, args.audRate)
    
    total_samples = len(val_set)
    print(f"Total evaluation samples: {total_samples}")
    if total_samples == 0:
        raise RuntimeError("No samples loaded, check list_val file path")

    mel_generator = TransUnet(args.hidden_size, 
                              args.transformer_num_layers, 
                              args.mlp_dim, 
                              args.num_heads, 
                              args.transformer_dropout_rate, 
                              args.transformer_attention_dropout_rate).cuda()

    logging.info(f'Loading best model from: {args.load_best_model}')
    package = torch.load(args.load_best_model, map_location='cpu', weights_only=False)

    if isinstance(package, dict):
        pretrained_dict = package.get('model_state_dict', package.get('net', package))
        epoch = package.get('epoch', 0)
        best_loss = package.get('best_loss', 0)
        logging.info(f"Loaded model: epoch={epoch}, best_loss={best_loss:.4f}")
    else:
        pretrained_dict = package
        logging.info("Loaded old format weights")

    pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
    missing_keys, unexpected_keys = mel_generator.load_state_dict(pretrained_dict, strict=True)

    if missing_keys:
        logging.error(f"Missing weights: {missing_keys[:5]}...")
        raise RuntimeError("Incomplete model weights")
    mel_generator.eval().cuda()

    with open(args.vocoder_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    vocoder = load_model(args.vocoder_ckpt, config)
    vocoder.remove_weight_norm()
    vocoder.eval().cuda()

    # Initialize DNSMOS
    print(f"Initializing DNSMOS evaluator...")
    print(f"   Primary model: {args.dnsmos_primary_model}")
    print(f"   SIG model: {args.dnsmos_sig_model if os.path.exists(args.dnsmos_sig_model) else 'Using Primary output'}")
    print(f"   Model SR: {args.dnsmos_sr}Hz (expected input: [batch, 901, 161])")
    
    try:
        compute_dnsmos = ComputeDNSMOS(
            primary_model_path=args.dnsmos_primary_model,
            sig_model_path=args.dnsmos_sig_model if os.path.exists(args.dnsmos_sig_model) else None,
            sr=args.dnsmos_sr
        )
        use_dnsmos = True
        print("DNSMOS model loaded successfully")
    except Exception as e:
        print(f"DNSMOS loading failed: {e}")
        use_dnsmos = False
        compute_dnsmos = None

    # Detect Mel normalization status
    print("Detecting Mel output range...")
    with torch.no_grad():
        test_radio_mel = val_set[0][3].cuda()
        test_pred = mel_generator(test_radio_mel).squeeze(0).squeeze(0)
        pred_min, pred_max = test_pred.cpu().numpy().min(), test_pred.cpu().numpy().max()
        use_normalize_before = True if pred_max > 10 or pred_min < -20 else False
        print(f"   Mel range: [{pred_min:.2f}, {pred_max:.2f}] -> normalize_before={use_normalize_before}")

    # Initialize metrics
    stoi_metric = AverageMeter()
    llr_metric = AverageMeter()
    pesq_metric = AverageMeter()
    lsd_metric = AverageMeter()
    
    if use_dnsmos:
        dnsmos_sig_metric = AverageMeter()
        dnsmos_bak_metric = AverageMeter()
        dnsmos_ovr_metric = AverageMeter()

    with torch.no_grad(), tqdm(val_set, desc='[decode]', total=total_samples) as pbar:
        for idx, (filename, audio_raw, audio_melamp, radio_melamp) in enumerate(pbar):
            radio_melamp = radio_melamp.cuda()
            pred_mel = mel_generator(radio_melamp).squeeze(0).squeeze(0)
            audio_pred = vocoder.inference(pred_mel, normalize_before=use_normalize_before).view(-1).cpu().numpy()
            
            min_len = max(22050, min(len(audio_raw), len(audio_pred)))
            audio_raw_aligned = audio_raw[:min_len]
            audio_pred_aligned = audio_pred[:min_len]

            os.makedirs(args.save_wave_path, exist_ok=True)
            sf.write(os.path.join(args.save_wave_path, f"{filename}.wav"), audio_pred_aligned, 22050)

            # Compute metrics
            try:
                stoi = pysepm.stoi(audio_raw_aligned, audio_pred_aligned, 22050)
                stoi = stoi if 0 <= stoi <= 1 else np.nan
            except:
                stoi = np.nan
            
            try:
                llr = pysepm.llr(audio_raw_aligned, audio_pred_aligned, 22050)
            except:
                llr = np.nan
            
            lsd = compute_lsd_22050(audio_raw_aligned, audio_pred_aligned)
            pesq_score = compute_pesq_8k_from_22050(audio_raw_aligned, audio_pred_aligned)
            
            # DNSMOS
            if use_dnsmos and compute_dnsmos:
                try:
                    dnsmos_scores = compute_dnsmos(audio_pred_aligned, 22050)
                    sig_score = dnsmos_scores['SIG']
                    bak_score = dnsmos_scores['BAK']
                    ovr_score = dnsmos_scores['OVR']
                except Exception as e:
                    sig_score, bak_score, ovr_score = np.nan, np.nan, np.nan
                    if idx == 0:
                        print(f"\nDNSMOS calculation error: {e}")
            else:
                sig_score, bak_score, ovr_score = np.nan, np.nan, np.nan

            # Update metrics
            if not np.isnan(stoi): stoi_metric.update(stoi)
            if not np.isnan(llr): llr_metric.update(llr)
            if not np.isnan(pesq_score): pesq_metric.update(pesq_score)
            if not np.isnan(lsd): lsd_metric.update(lsd)
            
            if use_dnsmos:
                if not np.isnan(sig_score): dnsmos_sig_metric.update(sig_score)
                if not np.isnan(bak_score): dnsmos_bak_metric.update(bak_score)
                if not np.isnan(ovr_score): dnsmos_ovr_metric.update(ovr_score)

            # Progress bar display
            postfix_dict = {
                'STOI': f'{stoi:.3f}' if not np.isnan(stoi) else 'N/A',
                'PESQ': f'{pesq_score:.3f}' if not np.isnan(pesq_score) else 'N/A',
                'LSD': f'{lsd:.2f}' if not np.isnan(lsd) else 'N/A',
            }
            if use_dnsmos:
                postfix_dict['SIG'] = f'{sig_score:.2f}' if not np.isnan(sig_score) else 'N/A'
                postfix_dict['OVR'] = f'{ovr_score:.2f}' if not np.isnan(ovr_score) else 'N/A'
            
            pbar.set_postfix(postfix_dict)

        # Final results
        print('\n==================== 22050Hz Evaluation Summary ====================')
        print(f'  Samples: {total_samples}')
        print(f'  LSD:    {lsd_metric.average():.4f}')
        print(f'  STOI:   {stoi_metric.average():.4f}')
        print(f'  LLR:    {llr_metric.average():.4f}')
        print(f'  PESQ:   {pesq_metric.average():.4f} (8kHz NB)')
        
        if use_dnsmos:
            print(f'  --------------------------------------------------')
            print(f'  DNSMOS (16kHz, 1-5 scale):')
            print(f'    SIG:  {dnsmos_sig_metric.average():.4f} (Signal Quality)')
            print(f'    BAK:  {dnsmos_bak_metric.average():.4f} (Background Noise Suppression)')
            print(f'    OVR:  {dnsmos_ovr_metric.average():.4f} (Overall Quality)')
        
        print('=====================================================================')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()