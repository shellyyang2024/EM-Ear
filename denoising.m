%% Batch Processing - LJSpeech EM Signal Processing (Strict Alignment + Circular Padding + Smart Energy Extension)

%% Configuration
usrp_fs = 50000; 
audio_fs = 22050;
input_dir = 'F:\ljspeech-0008\ljspeech-demod'; 
concat_audio_dir = 'E:\LJSpeech-1.1\LJSpeech-1.1\wavs-5-ljspeech';
sub_audio_dir = 'E:\LJSpeech-1.1\LJSpeech-1.1\wavs';
output_dir = 'F:\ljspeech-0008\ljspeech-demod\ljspeech-output';
mat_output_dir = fullfile(output_dir, 'mat');
%audio_output_dir = fullfile(output_dir, 'audio');

% Create directories
for dir_path = {mat_output_dir, audio_output_dir}
    if ~exist(dir_path{1}, 'dir'), mkdir(dir_path{1}); end
end

%% Main Processing Loop
fprintf('=== Starting Batch Processing ===\n');
em_files = dir(fullfile(input_dir, '*.raw'));
fprintf('Found %d raw files\n', length(em_files));
success_count = 0; 
skip_count = 0; 

for i = 1:length(em_files)
    em_filename = em_files(i).name;
    [~, name_without_ext, ~] = fileparts(em_filename);
    
    % Parse filename range (LJ001-0001-0005 format)
    [speaker_id, start_idx, end_idx, parse_success] = parse_range_filename_ljspeech(name_without_ext);
    if ~parse_success
        fprintf('\n[%d/%d] Skip: %s (invalid filename format)\n', i, length(em_files), em_filename);
        skip_count = skip_count + 1;
        continue;
    end
    
    fprintf('\n[%d/%d] Processing: %s (range: %s-%04d-%04d)\n', i, length(em_files), name_without_ext, speaker_id, start_idx, end_idx);
    
    % Read and preprocess EM signal
    em_path = fullfile(input_dir, em_filename);
    em_raw = read_em_signal_safe(em_path, usrp_fs);
    if isempty(em_raw), skip_count = skip_count + 1; continue; end
    
    em_filtered = hampel(em_raw, 5);
    em_resampled = resample(em_filtered, audio_fs, usrp_fs);
    
    % Energy-based segmentation
    em_det = em_resampled - mean(em_resampled);
    window_size = round(audio_fs * 0.005);
    em_env = movmean(abs(em_det), window_size);
    
    [TF1, ~] = ischange(em_env, 'mean', "MaxNumChanges", 10);
    changepts = find(TF1);
    
    if length(changepts) >= 2
        start_idx_em = max(1, changepts(1) - round(audio_fs * 0.5));
        end_idx_em = min(changepts(end) + round(audio_fs * 1), length(em_resampled));
    else
        fprintf('    Warning: Energy detection failed, skipping\n');
        skip_count = skip_count + 1;
        continue;
    end
    
    % Load concatenated audio
    concat_audio_file = fullfile(concat_audio_dir, sprintf('%s-%04d-%04d.wav', speaker_id, start_idx, end_idx));
    if ~exist(concat_audio_file, 'file')
        fprintf('  Error: Concatenated audio not found: %s\n', concat_audio_file);
        skip_count = skip_count + 1;
        continue;
    end
    
    [concat_audio, ~] = audioread(concat_audio_file);
    if size(concat_audio, 2) > 1, concat_audio = mean(concat_audio, 2); end
    concat_audio = concat_audio / max(abs(concat_audio) + eps);
    audio_len = length(concat_audio);
    fprintf('  Audio duration: %.2fs\n', audio_len/audio_fs);
    
    % Smart energy extension: extend start if EM segment shorter than audio
    detected_len = end_idx_em - start_idx_em + 1;
    if detected_len < audio_len
        shortage = audio_len - detected_len;
        new_start = max(1, start_idx_em - shortage);
        fprintf('  Extending EM segment: start moved from %.2fs to %.2fs\n', start_idx_em/audio_fs, new_start/audio_fs);
        start_idx_em = new_start;
    end
    
    % Crop EM signal
    em_cropped = em_resampled(start_idx_em:end_idx_em);
    em_cropped = em_cropped - mean(em_cropped);
    fprintf('  EM cropped: %.2fs (%.2fs to %.2fs)\n', length(em_cropped)/audio_fs, start_idx_em/audio_fs, end_idx_em/audio_fs);
    
    % Global alignment (audio intact, only crop EM)
    min_len = min(length(em_cropped), length(concat_audio));
    [r, lags] = xcorr(em_cropped(1:min_len) - mean(em_cropped(1:min_len)), ...
                       concat_audio(1:min_len) - mean(concat_audio(1:min_len)));
    [~, I] = max(abs(r)); 
    lag = lags(I);
    fprintf('  Cross-correlation lag: %d (positive = EM lags audio)\n', lag);
    
    % Align: keep full audio, only trim EM head
    if lag > 0
        em_aligned = em_cropped(lag+1:end);
        audio_aligned = concat_audio;
    else
        lag_abs = abs(lag);
        em_aligned = em_cropped(lag_abs+1:end);
        audio_aligned = concat_audio;
    end
    
    % Force equal length: circular padding if EM shorter
    target_len = length(audio_aligned);
    em_len = length(em_aligned);
    
    if em_len < target_len
        shortage = target_len - em_len;
        fprintf('  EM short by %.2fs, circular padding from tail\n', shortage/audio_fs);
        em_aligned = [em_aligned; em_aligned(end-shortage+1:end)];
    elseif em_len > target_len
        fprintf('  EM long by %.2fs, truncating\n', (em_len-target_len)/audio_fs);
        em_aligned = em_aligned(1:target_len);
    end
    
    assert(length(em_aligned) == length(audio_aligned), 'Fatal: length mismatch after alignment');
    fprintf('  Alignment success: Audio %.2fs, EM %.2fs\n', length(audio_aligned)/audio_fs, length(em_aligned)/audio_fs);
    
    % Build sub-audio file list
    all_possible_files = cell(1, end_idx-start_idx+1);
    file_exist_flags = false(1, length(all_possible_files));
    for k = start_idx:end_idx
        all_possible_files{k-start_idx+1} = fullfile(sub_audio_dir, sprintf('%s-%04d.wav', speaker_id, k));
        file_exist_flags(k-start_idx+1) = exist(all_possible_files{k-start_idx+1}, 'file');
    end
    
    audio_files = all_possible_files(file_exist_flags);
    existing_indices = start_idx:end_idx;
    existing_indices = existing_indices(file_exist_flags);
    missing_count = sum(~file_exist_flags);
    fprintf('  Sub-audio: %d found, %d missing\n', length(audio_files), missing_count);
    
    if length(audio_files) < 1
        fprintf('  Error: Too few audio files, skipping\n');
        skip_count = skip_count + 1;
        continue;
    end
    
    % Process each segment
    current_em_pos = 1;
    for seg = 1:length(audio_files)
        audio_file = audio_files{seg};
        original_idx = existing_indices(seg);
        
        [audio_data, ~] = audioread(audio_file);
        if size(audio_data, 2) > 1, audio_data = mean(audio_data, 2); end
        audio_data = audio_data / max(abs(audio_data) + eps);
        
        audio_len = length(audio_data);
        em_seg = em_aligned(current_em_pos:current_em_pos + audio_len - 1);
        
        if length(em_seg) ~= audio_len
            error('Segment %d(%s-%04d) length mismatch: EM=%d, Audio=%d', seg, speaker_id, original_idx, length(em_seg), audio_len);
        end
        
        fprintf('  |- %s-%04d: %.2fs\n', speaker_id, original_idx, audio_len/audio_fs);
        
        process_single_segment(speaker_id, original_idx, audio_data, em_seg, ...
                               mat_output_dir, audio_fs, audio_output_dir);
        
        current_em_pos = current_em_pos + audio_len;
    end
    
    success_count = success_count + 1;
end

fprintf('\n=== Batch Processing Complete ===\n');
fprintf('Success rate: %.1f%% (%d/%d)\n', success_count/length(em_files)*100, success_count, length(em_files));

%% Helper Functions

function data = read_em_signal_safe(filepath, fs)
% Read float32 format EM signal
    data = []; 
    fid = -1;
    try
        info = dir(filepath); 
        file_size_bytes = info.bytes;
        num_samples = floor(file_size_bytes / 4);
        if num_samples == 0, error('File empty'); end
        
        fid = fopen(filepath, 'rb');
        if fid == -1, error('Cannot open file: %s', filepath); end
        
        data = fread(fid, num_samples, 'float32');
        fclose(fid); 
        fid = -1;
        data = data(:);
        
        fprintf('  EM: %d samples (%.2fs)\n', length(data), length(data)/fs);
    catch ME
        if fid ~= -1, fclose(fid); end
        warning('  Read failed: %s', ME.message);
        data = [];
    end
end

function process_single_segment(speaker_id, idx, audio_data, em_seg, mat_output_dir, audio_fs, audio_output_dir)
% Process single segment: filter -> align -> normalize -> denoise -> save
    
    eps = 1e-8;
    
    % Stage 1: Bandpass filter 80-7600Hz
    [b_bp, a_bp] = butter(4, [80 7600]/(audio_fs/2), 'bandpass');
    em_bp = filtfilt(b_bp, a_bp, em_seg);
    audio_bp = filtfilt(b_bp, a_bp, audio_data);
    
    % Stage 2: Fine alignment with circular padding
    min_len = min(length(em_bp), length(audio_bp));
    search_range = round(audio_fs * 0.2);
    [r_local, lags_local] = xcorr(em_bp(1:min_len) - mean(em_bp(1:min_len)), ...
                                  audio_bp(1:min_len) - mean(audio_bp(1:min_len)), ...
                                  search_range, 'coeff');
    [r_max, I_local] = max(abs(r_local)); 
    lag_local = lags_local(I_local);
    
    N = length(em_bp);
    if lag_local > 0
        em_aligned = [em_bp(end-lag_local+1:end); em_bp(1:end-lag_local)];
    elseif lag_local < 0
        lag_abs = abs(lag_local);
        em_aligned = [em_bp(lag_abs+1:end); em_bp(1:lag_abs)];
    else
        em_aligned = em_bp;
    end
    
    % Stage 3: Amplitude normalization (before denoising)
    max_amp = max(abs(em_aligned));
    if max_amp > eps
        em_normalized = em_aligned / max_amp;
    else
        em_normalized = em_aligned;
    end
    
    % Stage 4: Spectral subtraction denoising
    em_denoised = spectral_subtraction(em_normalized, audio_fs, 0.15, 3, 0.1);
    
    % Stage 5: Final lowpass and normalization
    [b_low, a_low] = ellip(8, 0.5, 80, 7600/(audio_fs/2), 'low');
    em_lowpass = filtfilt(b_low, a_low, em_denoised);
    em_final = em_lowpass / (max(abs(em_lowpass)) + eps) * 0.95;
    
    % Save data
    audio_original = audio_data(1:length(em_final));
    
    mat_filename = fullfile(mat_output_dir, sprintf('%s-%04d.mat', speaker_id, idx));
    save(mat_filename, 'em_final', 'audio_original', 'audio_fs', 'lag_local', 'r_max');
    
    audiowrite(fullfile(audio_output_dir, sprintf('%s-%04d_em.wav', speaker_id, idx)), em_final, audio_fs);
    
    fprintf('    Done: %s-%04d (r=%.2f, lag=%d)\n', speaker_id, idx, r_max, lag_local);
end

function output = spectral_subtraction(signal, fs, noise_percent, alpha, beta)
% Improved spectral subtraction denoising
    signal = signal(:); 
    frame_len = 1024; 
    hop_len = 256; 
    window = hann(frame_len);
    num_frames = floor((length(signal) - frame_len) / hop_len) + 1;
    
    if num_frames < 10
        warning('Signal too short for spectral subtraction'); 
        output = signal; 
        return;
    end
    
    frames = zeros(frame_len, num_frames);
    for i = 1:num_frames
        start_idx = (i-1)*hop_len + 1;
        frames(:, i) = signal(start_idx:start_idx+frame_len-1) .* window;
    end
    
    spectra = fft(frames, frame_len);
    magnitudes = abs(spectra);
    phases = angle(spectra);
    
    % Noise estimation (energy-sorted)
    energy = sum(magnitudes.^2, 1);
    [~, sort_idx] = sort(energy);
    num_noise_frames = max(1, floor(num_frames * noise_percent));
    noise_frame_indices = sort_idx(1:num_noise_frames);
    noise_spectrum = mean(magnitudes(:, noise_frame_indices), 2);
    noise_spectrum = max(noise_spectrum, mean(noise_spectrum) * 0.01);
    
    % Spectral subtraction with oversubtraction
    subtracted_magnitudes = max(magnitudes - alpha * noise_spectrum, beta * magnitudes);
    
    % Signal reconstruction
    restored_spectra = subtracted_magnitudes .* exp(1j * phases);
    output_frames = real(ifft(restored_spectra));
    
    % Overlap-add
    output = zeros(length(signal), 1);
    for i = 1:num_frames
        start_idx = (i-1)*hop_len + 1;
        output(start_idx:start_idx+frame_len-1) = output(start_idx:start_idx+frame_len-1) + output_frames(:, i);
    end
end

function [speaker_id, start_num, end_num, success] = parse_range_filename_ljspeech(filename)
% Parse LJSpeech filename format: LJ001-0001-0005
    success = false; 
    speaker_id = ''; 
    start_num = 0; 
    end_num = 0;
    pattern = '(\w+)-(\d+)-(\d+)'; 
    tokens = regexp(filename, pattern, 'tokens');
    
    if ~isempty(tokens)
        try
            speaker_id = tokens{1}{1};
            start_num = str2double(tokens{1}{2});
            end_num = str2double(tokens{1}{3});
            if ~isnan(start_num) && ~isnan(end_num) && end_num >= start_num
                success = true;
            end
        catch
            success = false;
        end
    end
end