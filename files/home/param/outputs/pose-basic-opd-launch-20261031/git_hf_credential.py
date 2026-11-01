import sys
from huggingface_hub import get_token
if len(sys.argv)>1 and sys.argv[1]=='get':
    fields=dict(line.rstrip('\n').split('=',1) for line in sys.stdin if '=' in line)
    if fields.get('protocol')=='https' and fields.get('host')=='huggingface.co':
        token=get_token()
        if token:
            print('username=hf_user')
            print('password='+token)
