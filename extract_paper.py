import json

in_file = '/home/saurav/.gemini/antigravity/brain/5b38f2c4-14d9-4323-a080-0b65f2665020/.system_generated/logs/transcript_full.jsonl'
out_file = '/home/saurav/Documents/Research/apex_journal/extracted_paper.txt'

target_text = ""
with open(in_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            content = data.get('content', '')
            if 'Formatting requirements' in content:
                target_text = content

with open(out_file, 'w') as f:
    f.write(target_text)
