#!/usr/bin/env bash
# FINAL M0 acceptance test — post-NAT-fix. Supersedes the earlier env check.
PY=/home/jiming/miniconda3/envs/llmquant/bin/python
PIP=/home/jiming/miniconda3/envs/llmquant/bin/pip
SEC=/home/jiming/llm-quant-serve/scripts
mkdir -p "$SEC"
cp -f "$0" "$SEC/00_env_check.sh" 2>/dev/null || true

echo "########## 1. identity ##########"
"$PY" -c "import sys;print('interpreter', sys.executable)"

echo
echo "########## 2. stack versions ##########"
"$PY" -c "import torch;print('torch        ', torch.__version__, '| rt', torch.version.cuda, '| sm120', ('sm_120' in torch.cuda.get_arch_list()))"
"$PY" -c "import tensorrt as t;print('tensorrt     ', t.__version__)"
"$PY" -c "import tensorrt_llm as tl;print('tensorrt_llm ', tl.__version__)"
"$PY" -c "import transformers as tf, importlib.metadata as md;print('transformers ', tf.__version__, '| modelopt', md.version('nvidia-modelopt'))"

echo
echo "########## 3. device + kernel execution ##########"
"$PY" -c "
import torch, time
p = torch.cuda.get_device_properties(0)
print('device %s | sm_%d%d | SMs %d' % (p.name, p.major, p.minor, p.multi_processor_count))
a = torch.randn(4096,4096,device='cuda'); b = torch.randn(4096,4096,device='cuda')
torch.cuda.synchronize(); t=time.time()
for _ in range(20): c = a@b
torch.cuda.synchronize(); dt=(time.time()-t)/20
print('matmul 4096^3 %.3f ms  %.1f TFLOPS (NOT a baseline)' % (dt*1e3, 2*4096**3/dt/1e12))
"

echo
echo "########## 4. MPI (the overnight blocker) ##########"
"$PY" -c "
import socket, time
s=socket.socket(); s.settimeout(3); t=time.time()
try: s.connect(('127.0.0.1',6001)); print('loopback closed-port: CONNECTED (unexpected)')
except socket.timeout: print('loopback closed-port: TIMEOUT  <== regression')
except ConnectionRefusedError: print('loopback closed-port: REFUSED %.3fs (healthy)' % (time.time()-t))
"
s=$(date +%s); timeout -k 5 60 mpirun -n 1 hostname > /tmp/f_m.log 2>&1; rc=$?; e=$(date +%s)
echo "mpirun -n 1 hostname: rc=$rc ${((e-s))}s out=[$(cat /tmp/f_m.log | tr '\n' ' ' | cut -c1-40)]"

echo
echo "########## 5. loader map — full paths ##########"
"$PY" - <<'PYEOF'
import torch
torch.zeros(1, device='cuda')
maps = open('/proc/self/maps').read().splitlines()
for pat in ('libcudart','libcublas','libcudnn','libnccl','libcusparse'):
    hits = sorted({l.split()[-1] for l in maps if pat in l.split()[-1]})
    print('%-12s' % pat, [h.replace('/home/jiming/miniconda3/envs/llmquant/lib/python3.12/site-packages/','…/') for h in hits] or '-- none --')
PYEOF

echo
echo "########## 6. file-ownership collision audit ##########"
"$PY" - <<'PYEOF'
import importlib.metadata as md
from collections import defaultdict
owner = defaultdict(set)
for d in md.distributions():
    for f in (d.files or []):
        owner[str(f)].add(d.metadata['Name'])
coll = {p: sorted(v) for p, v in owner.items() if len(v) > 1}
print('paths owned by >1 distribution:', len(coll))
for p, v in sorted(coll.items())[:10]:
    print('  ', p, '<-', v)
PYEOF

echo
echo "########## 7. pip check (expect only deviation D1) ##########"
"$PIP" check 2>&1 | head -4

echo
echo "########## 8. freeze drift vs pre-removal state ##########"
cd /home/jiming/llm-quant-serve/docs
"$PIP" freeze > pip_freeze_final.txt
if diff pip_freeze_before_cu12_removal.txt pip_freeze_final.txt > /tmp/f_drift.txt; then
  echo "IDENTICAL (unexpected)"
else
  echo "removed lines: $(grep -c '^<' /tmp/f_drift.txt)   added lines: $(grep -c '^>' /tmp/f_drift.txt)   (added must be 0)"
  grep '^>' /tmp/f_drift.txt | head -5
fi

pkill -9 -f orted 2>/dev/null
echo
echo "########## DONE ##########"
