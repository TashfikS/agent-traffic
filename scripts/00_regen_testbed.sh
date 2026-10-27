#!/bin/bash
# Regenerates the controlled (loopback) testbed corpus into data/raw_v3. Resumable 50-episode chunks, 4 lanes.
# main_*/subM* use the unmodified paper testbed (testbed2.py) and the paper's seeds -> same generative process.
# ctx_*/failfast use testbed3.py (adds non-monotonic context policies and instant-traceback latency).
cd "$(dirname "$0")/.."; O=data/raw_v3; mkdir -p $O
job(){ local tb=$1 port=$2 tag=$3 tot=$4; shift 4
  for ((s=0; s<tot; s+=50)); do c=$(printf "%04d" $s); [ -f $O/gt_${tag}_c$c.csv ] && continue
    python code/$tb --out $O --port $port --tag $tag --ep_start $s --episodes 50 --concurrency 25 "$@" >> $O/run.log 2>&1; done; }
L1(){ job testbed2.py 22010 main_seed1 350 --seed 1; job testbed2.py 22010 subM8 100 --seed 28 --force_sub_m 8; job testbed3.py 22010 ctx_window 200 --seed 41 --ctx_policy window --ctx_budget 40000; }
L2(){ job testbed2.py 22020 main_seed2 350 --seed 2; job testbed2.py 22020 subM4 100 --seed 24 --force_sub_m 4; job testbed3.py 22020 ctx_summ 200 --seed 42 --ctx_policy summarize --ctx_budget 40000; }
L3(){ job testbed2.py 22030 main_seed3 350 --seed 3; job testbed2.py 22030 subM2 100 --seed 22 --force_sub_m 2; job testbed3.py 22030 failfast 200 --seed 43 --fail_latency instant; }
L4(){ job testbed2.py 22040 subM1 100 --seed 21 --force_sub_m 1; job testbed2.py 22040 subM8b 100 --seed 128 --force_sub_m 8; job testbed2.py 22040 subM4b 100 --seed 124 --force_sub_m 4; job testbed2.py 22040 subM2b 100 --seed 122 --force_sub_m 2; job testbed2.py 22040 state 100 --seed 12 --mode server_state; job testbed2.py 22040 h1_stream 100 --seed 11 --proto h1; }
L1 & L2 & L3 & L4 & wait; echo ALLDONE >> $O/run.log
