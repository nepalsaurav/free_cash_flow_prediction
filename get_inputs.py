import json
with open('/home/saurav/.gemini/antigravity/brain/5b38f2c4-14d9-4323-a080-0b65f2665020/.system_generated/logs/transcript.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'USER_INPUT':
            print(f"Step {data.get('step_index')}:")
            print(data.get('content')[:800])
            print('-'*40)
