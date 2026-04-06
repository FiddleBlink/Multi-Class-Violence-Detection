import re
from collections import defaultdict

# Read annotations.txt
with open('Features/annotations.txt', 'r') as f:
    lines = f.readlines()

# Parse annotations
categories = defaultdict(lambda: defaultdict(list))
mixed_categories = defaultdict(list)
single_categories = defaultdict(list)

for line in lines:
    # Extract the label and video info
    match = re.search(r'label_([^-]+)-', line)
    if match:
        label = match.group(1)
        
        # Check if this is a mixed label (contains multiple categories)
        label_part = line.split('label_')[1].split('-')[0]
        
        # Get video ID
        if '__' in line:
            video_id = line.split('__')[0].replace('v=', '')
        else:
            video_id = line.split('_')[0]
        
        categories[label][video_id].append(line.strip())

# Now check each video to see if it has only one label
for label in categories:
    for video_id, clips in categories[label].items():
        # For this video, check all clips
        labels_in_video = set()
        for clip_line in clips:
            # Extract all labels from this clip
            match = re.search(r'label_([^-]+)', clip_line)
            if match:
                labels_in_video.add(match.group(1))
        
        # If only one unique label, it's pure
        if len(labels_in_video) == 1:
            single_categories[label].append(video_id)
        else:
            mixed_categories[label].append(video_id)

# Get unique videos per category
print("PURE SINGLE-CATEGORY VIDEOS (no mixed labels)")
print("=" * 100)
for label in sorted(single_categories.keys()):
    unique_videos = list(set(single_categories[label]))
    print(f"\n{label} - Pure single-category videos: {len(unique_videos)}")
    for i, vid in enumerate(sorted(unique_videos)[:5]):
        print(f"  {i+1}. {vid}")
    if len(unique_videos) > 5:
        print(f"  ... and {len(unique_videos) - 5} more")

# Detailed analysis
print("\n" + "=" * 100)
print("\nDETAILED PURE VIDEOS FOR PER-CATEGORY MODALITY ANALYSIS")
print("=" * 100)

for label in sorted(single_categories.keys()):
    unique_videos = sorted(list(set(single_categories[label])))
    print(f"\n### {label} Category - Pure Videos ({len(unique_videos)} total) ###")
    
    # Get stats for each video
    video_stats = []
    for video_id in unique_videos:
        clips = categories[label][video_id]
        total_frames = 0
        total_duration = 0
        
        for clip in clips:
            parts = clip.strip().split()
            if len(parts) > 1:
                try:
                    frames = [int(f) for f in parts[1:]]
                    total_frames += len(frames)
                    if frames:
                        total_duration += max(frames) - min(frames)
                except:
                    pass
        
        video_stats.append({
            'video_id': video_id,
            'num_clips': len(clips),
            'total_frames': total_frames,
            'total_duration': total_duration
        })
    
    # Sort by total frames
    video_stats.sort(key=lambda x: x['total_frames'], reverse=True)
    
    # Show all videos with stats
    for i, video in enumerate(video_stats[:10], 1):
        print(f"  {i}. {video['video_id']:40} - Clips: {video['num_clips']}, Annotated Frames: {video['total_frames']}")

print("\n" + "=" * 100)
print("SUMMARY - Count of Pure Videos per Category")
print("=" * 100)
for label in sorted(single_categories.keys()):
    unique_videos = len(set(single_categories[label]))
    print(f"{label}: {unique_videos} pure videos")
