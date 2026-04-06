import re
from collections import defaultdict

# Read annotations.txt
with open('Features/annotations.txt', 'r') as f:
    lines = f.readlines()

# Parse annotations
categories = defaultdict(list)
for line in lines:
    # Extract the label and video info
    match = re.search(r'label_([^-]+)', line)
    if match:
        label = match.group(1)
        # Get the full line info
        categories[label].append(line.strip())

# Display findings
print('Category Labels Found:')
print('=' * 100)
for label in sorted(categories.keys()):
    print(f'\n{label}: {len(categories[label])} videos/clips')
    # Show first 3 examples
    for i, line in enumerate(categories[label][:3]):
        print(f'  Example {i+1}: {line[:90]}...' if len(line) > 90 else f'  Example {i+1}: {line}')

print('\n' + '=' * 100)
print('\nDetailed Video List for Each Category:')
print('=' * 100)

for label in sorted(categories.keys()):
    print(f'\n### {label} Category ###')
    # Get unique video IDs for each category
    video_ids = set()
    for line in categories[label]:
        if '__' in line:
            video_id = line.split('__')[0].replace('v=', '')
        else:
            video_id = line.split('_')[0]
        video_ids.add(video_id)
    
    print(f'Total unique videos: {len(video_ids)}')
    for i, vid in enumerate(sorted(video_ids)[:5]):  # Show first 5
        print(f'  {i+1}. {vid}')
    if len(video_ids) > 5:
        print(f'  ... and {len(video_ids) - 5} more videos')
