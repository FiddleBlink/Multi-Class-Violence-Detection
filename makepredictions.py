from torch.utils.data import DataLoader
import torch
import numpy as np
from model import Model
from dataset import Dataset
from test import get_modality_slices, compute_modality_drops
import option
import os
import json
import argparse

# Feature sizes for different modalities
MODALITY_FEATURE_SIZES = {
    'RGB': 1024,
    'AUDIO': 128,
    'FLOW': 1024,
    'MIX': 1024 * 2,  # RGB + FLOW
    'MIX2': 1024 + 128,  # RGB + AUDIO
    'MIX3': 1024 + 128,  # FLOW + AUDIO
    'MIX_ALL': 1024 + 1024 + 128  # RGB + FLOW + AUDIO
}

LABEL_NAMES = {
    0: 'Normal',
    1: 'Fighting',
    2: 'Shooting',
    3: 'Explosion',
    4: 'Riot',
    5: 'Abuse',
    6: 'Car accident'
}


def extract_clip_key(path):
    base = os.path.basename(path.strip())
    if '__' in base:
        # Remove the final crop suffix __0.npy, __1.npy, ...
        parts = base.split('__')
        return '__'.join(parts[:-1])
    return base


def segment_predictions(preds, block_size=16, fps=24, min_duration=0.1):
    segments = []
    if len(preds) == 0:
        return segments

    current_label = int(preds[0])
    start_idx = 0

    for idx in range(1, len(preds)):
        if int(preds[idx]) != current_label:
            end_idx = idx - 1
            start_time = round(start_idx * block_size / fps, 2)
            end_time = round((end_idx + 1) * block_size / fps, 2)
            if end_time - start_time >= min_duration:
                segments.append({
                    'label_id': current_label,
                    'label_name': LABEL_NAMES.get(current_label, 'Unknown'),
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': round(end_time - start_time, 2)
                })
            current_label = int(preds[idx])
            start_idx = idx

    end_idx = len(preds) - 1
    start_time = round(start_idx * block_size / fps, 2)
    end_time = round((end_idx + 1) * block_size / fps, 2)
    if end_time - start_time >= min_duration:
        segments.append({
            'label_id': current_label,
            'label_name': LABEL_NAMES.get(current_label, 'Unknown'),
            'start_time': start_time,
            'end_time': end_time,
            'duration': round(end_time - start_time, 2)
        })

    return segments


def entropy(probs):
    safe_probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(safe_probs * np.log(safe_probs), axis=1)


def score_modality_dominance(frame_preds, drops, modality_names):
    frame_modalities = []
    frame_values = []
    for frame_idx, cls_idx in enumerate(frame_preds):
        contributions = {name: float(max(0.0, drops[name][frame_idx, cls_idx])) for name in modality_names}
        dominant = max(contributions, key=contributions.get)
        frame_modalities.append(dominant)
        frame_values.append(contributions)
    return frame_modalities, frame_values


FULL_MODEL_FEATURE_SIZE = 1024 + 128
SUPPORTED_MODALITIES = ['RGB', 'AUDIO', 'MIX2']


def pad_features_to_full(features, selected_modality, full_feature_size=FULL_MODEL_FEATURE_SIZE):
    if selected_modality == 'MIX2':
        return features

    batch_size, seq_len, feat_dim = features.shape
    if selected_modality == 'RGB':
        if feat_dim != 1024:
            raise ValueError(f'RGB input must have 1024 features, got {feat_dim}')
        pad = torch.zeros((batch_size, seq_len, full_feature_size - feat_dim), dtype=features.dtype, device=features.device)
        return torch.cat((features, pad), dim=2)

    if selected_modality == 'AUDIO':
        if feat_dim != 128:
            raise ValueError(f'AUDIO input must have 128 features, got {feat_dim}')
        pad = torch.zeros((batch_size, seq_len, full_feature_size - feat_dim), dtype=features.dtype, device=features.device)
        return torch.cat((pad, features), dim=2)

    raise ValueError(f'Modality {selected_modality} is not supported with the current checkpoint.')


def run_analysis():
    # Parse command line arguments for modality override
    parser = argparse.ArgumentParser(description='Run modality analysis on XDVioDet')
    parser.add_argument('--modality', default='MIX2', 
                       choices=SUPPORTED_MODALITIES,
                       help='Modality to use for analysis (default: MIX2)')
    parser.add_argument('--feature-size', type=int, default=None,
                       help='Override feature size (ignored when using checkpoint feature size)')
    
    script_args, remaining = parser.parse_known_args()
    
    # Get base args from option.py
    args = option.parser.parse_args(remaining)
    
    # Override modality, but keep the model feature size fixed to the checkpoint size
    args.modality = script_args.modality
    args.feature_size = FULL_MODEL_FEATURE_SIZE
    selected_modality = script_args.modality
    print(f"Using modality: {selected_modality} with model feature size: {args.feature_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Constants for temporal processing
    block_size = 16
    fps = 24
    min_duration_clean = 3.0  # 3 seconds threshold for cleaned predictions

    test_dataset = Dataset(args, test_mode=True, return_path=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = Model(args).to(device)
    checkpoint = torch.load(r'ckpt\wsanodet_New_MIX2_v1.3_\wsanodet_New_MIX2_v1.3_.pkl', map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
    model.eval()

    if selected_modality == 'RGB':
        modality_slices = [('RGB', slice(0, 1024))]
    elif selected_modality == 'AUDIO':
        modality_slices = [('AUDIO', slice(1024, FULL_MODEL_FEATURE_SIZE))]
    else:
        modality_slices = get_modality_slices('MIX2', args.feature_size)
    modality_names = [name for name, _ in modality_slices]

    analysis_results = []
    clip_index = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_inputs, batch_paths = batch
            batch_inputs = batch_inputs.to(device)
            batch_inputs = pad_features_to_full(batch_inputs, selected_modality)
            batch_paths = [p.strip() for p in batch_paths]
            clip_name = extract_clip_key(batch_paths[0])

            # Ensure all 5 crops belong to the same clip
            if not all(extract_clip_key(p) == clip_name for p in batch_paths):
                raise ValueError(f'Batch {batch_idx} contains mixed clips: {batch_paths[:5]}')

            full_probs, drops = compute_modality_drops(model, batch_inputs, seq_len=None, modality_slices=modality_slices)
            full_probs = np.asarray(full_probs)
            frame_preds = np.argmax(full_probs, axis=1)
            frame_entropy = entropy(full_probs)

            frame_modalities, frame_modality_values = score_modality_dominance(frame_preds, drops, modality_names)

            overall_modality_mean = {
                name: float(np.mean([frame_modality_values[i][name] for i in range(len(frame_preds))]))
                for name in modality_names
            }
            overall_dominant_modality = max(overall_modality_mean, key=overall_modality_mean.get)

            modality_frame_count = {name: int(frame_modalities.count(name)) for name in modality_names}
            modality_frame_share = {name: round(modality_frame_count[name] / len(frame_preds), 3) for name in modality_names}

            predicted_segments = segment_predictions(frame_preds, block_size=block_size, fps=fps)
            cleaned_segments = segment_predictions(frame_preds, block_size=block_size, fps=fps, min_duration=min_duration_clean)
            
            # Create mask for frames in cleaned segments (>= 3 seconds)
            cleaned_frame_mask = np.zeros(len(frame_preds), dtype=bool)
            for seg in cleaned_segments:
                start_frame = int(seg['start_time'] * fps / block_size)
                end_frame = int(seg['end_time'] * fps / block_size)
                cleaned_frame_mask[start_frame:end_frame] = True
            
            # Compute cleaned label distribution (only from frames in long segments)
            cleaned_frame_preds = frame_preds[cleaned_frame_mask] if cleaned_frame_mask.any() else np.array([])
            cleaned_class_counts = {LABEL_NAMES[i]: int((cleaned_frame_preds == i).sum()) for i in range(len(LABEL_NAMES))}
            cleaned_label_distribution = {name: count for name, count in cleaned_class_counts.items() if count > 0}
            
            class_counts = {LABEL_NAMES[i]: int((frame_preds == i).sum()) for i in range(len(LABEL_NAMES))}
            label_distribution = {name: count for name, count in class_counts.items() if count > 0}

            analysis_results.append({
                'clip_name': clip_name,
                'batch_index': batch_idx,
                'num_frames': int(len(frame_preds)),
                'duration_seconds': round(len(frame_preds) * 16 / 24, 2),
                'predicted_classes': label_distribution,
                'cleaned_predicted_classes': cleaned_label_distribution,
                'average_entropy': float(np.mean(frame_entropy)),
                'entropy_std': float(np.std(frame_entropy)),
                'segments': predicted_segments,
                'cleaned_segments': cleaned_segments,
                'dominant_modality_by_frame': frame_modalities,
                'dominant_modality_counts': modality_frame_count,
                'dominant_modality_share': modality_frame_share,
                'overall_modality_scores': overall_modality_mean,
                'overall_dominant_modality': overall_dominant_modality,
                'modality_names': modality_names,
                'raw_probs_shape': full_probs.shape
            })

            print(f'[{batch_idx}] Clip: {clip_name}')
            print(f'  Duration: {round(len(frame_preds) * 16 / 24, 2)}s | Frames: {len(frame_preds)}')
            print(f'  Predicted class counts: {label_distribution}')
            print(f'  Cleaned class counts (>=3s): {cleaned_label_distribution}')
            print(f'  Overall dominant modality: {overall_dominant_modality}')
            print(f'  Modality frame share: {modality_frame_share}')
            print(f'  Avg entropy: {round(float(np.mean(frame_entropy)), 4)}')
            print(f'  Segments: {len(predicted_segments)} | Cleaned segments: {len(cleaned_segments)}')
            print('')

    output_path = os.path.join('./Results', f'analysis_results_{args.modality}.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2)

    summary = {
        'total_clips': len(analysis_results),
        'modalities': modality_names,
        'chosen_modality': args.modality,
        'feature_size': args.feature_size,
        'recommendations': [
            'Report per-frame dominant modality using ablation drop values for the predicted class.',
            'Use clip-level average modality dominance to compare which modality is most important for each category.',
            'Include uncertainty metrics such as average entropy across frames.',
            'Include predicted temporal segments and their duration distribution for each video.',
            'Use modality frame share to demonstrate how often each modality influences decisions in a clip.',
            'Compare raw predictions vs. cleaned predictions (>=3s segments) to show robustness to short false positives.',
            'Analyze cleaned class distributions to understand dominant categories after filtering noise.',
            f'For {args.modality} modality: Evaluate how well single modalities perform vs. multimodal combinations.'
        ]
    }
    summary_path = os.path.join('./Results', f'analysis_recommendations_{args.modality}.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f'Analysis saved to: {output_path}')
    print(f'Research recommendations saved to: {summary_path}')


if __name__ == '__main__':
    run_analysis()
            