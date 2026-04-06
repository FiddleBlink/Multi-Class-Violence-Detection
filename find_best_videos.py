import re
from collections import defaultdict

# Read annotations.txt
with open('Features/annotations.txt', 'r') as f:
    lines = f.readlines()

# Parse annotations and calculate statistics
categories = defaultdict(lambda: defaultdict(list))

for line in lines:
    # Extract the label and video info
    match = re.search(r'label_([^-]+)', line)
    if match:
        label = match.group(1)
        # Get video ID
        if '__' in line:
            video_id = line.split('__')[0].replace('v=', '')
        else:
            video_id = line.split('_')[0]
        
        # Extract frame numbers (the actual annotation data)
        parts = line.strip().split()
        frames = []
        if len(parts) > 1:
            # Skip the first part (video ID and label), the rest are frame numbers
            frame_parts = parts[1:]
            try:
                frames = [int(f) for f in frame_parts]
            except:
                pass
        
        # Calculate duration/range
        if frames:
            duration = max(frames) - min(frames) if len(frames) > 1 else frames[0]
        else:
            duration = 0
        
        categories[label][video_id].append({
            'line': line.strip(),
            'frames': frames,
            'duration': duration,
            'num_frames': len(frames)
        })

# Find best 2 videos for each category
print("TOP 2 VIDEOS FOR ANALYSIS BY CATEGORY")
print("=" * 120)

for label in sorted(categories.keys()):
    print(f'\n### {label} Category ###')
    
    # Calculate statistics per video
    video_stats = []
    for video_id, clips in categories[label].items():
        total_frames = sum(clip['num_frames'] for clip in clips)
        total_duration = sum(clip['duration'] for clip in clips)
        num_clips = len(clips)
        
        video_stats.append({
            'video_id': video_id,
            'total_frames': total_frames,
            'total_duration': total_duration,
            'num_clips': num_clips,
            'clips': clips
        })
    
    # Sort by total frames annotated (descending) - this gives us videos with most annotations
    video_stats.sort(key=lambda x: x['total_frames'], reverse=True)
    
    # Display top 2
    for rank, video in enumerate(video_stats[:2], 1):
        print(f"\nRank {rank}: {video['video_id']}")
        print(f"  - Total annotated frames: {video['total_frames']}")
        print(f"  - Total timeline duration: {video['total_duration']} frames")
        print(f"  - Number of clips: {video['num_clips']}")
        print(f"  - Sample clip:")
        print(f"    {video['clips'][0]['line'][:100]}...")

# Also show category statistics
print("\n" + "=" * 120)
print("\nCATEGORY STATISTICS SUMMARY:")
print("=" * 120)
for label in sorted(categories.keys()):
    unique_videos = len(categories[label])
    total_clips = sum(len(clips) for clips in categories[label].values())
    total_annotated_frames = sum(sum(clip['num_frames'] for clip in clips) 
                                 for clips in categories[label].values())
    
    print(f"{label:5} - Videos: {unique_videos:3}  |  Clips: {total_clips:3}  |  Total Annotated Frames: {total_annotated_frames:5}")
