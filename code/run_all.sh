#!/bin/bash
# Track 1 data generation: deterministic seeds, resumable 50-episode chunks, two parallel lanes.
cd "$(dirname "$0")/.."; O=results_v2; mkdir -p $O
[ -f $O/key.pem ] || openssl req -x509 -newkey rsa:2048 -nodes -keyout $O/key.pem -out $O/cert.pem -days 30 -subj /CN=api.example.com 2>/dev/null
job(){ # port tag total args...
  local port=$1 tag=$2 tot=$3; shift 3
  for ((s=0; s<tot; s+=50)); do
    c=$(printf "%04d" $s); [ -f $O/gt_${tag}_c$c.csv ] && continue
    python code/testbed2.py --out $O --port $port --tag $tag --ep_start $s --episodes 50 --concurrency 25 "$@" >> $O/run.log 2>&1
  done; }
lane1(){
  job 20010 main_seed1 350 --seed 1 --proto h2 --mode full --stream 1
  job 20010 main_seed3 350 --seed 3 --proto h2 --mode full --stream 1
  job 20010 state 200 --seed 12 --proto h2 --mode server_state --stream 1
  job 20010 nostream 200 --seed 14 --proto h2 --mode full --stream 0
  job 20010 sweep_seq 200 --seed 15 --proto h2 --mode full --stream 1 --size_mode sweep --motif_only sequential
  job 20010 subM1 100 --seed 21 --proto h2 --mode full --stream 1 --force_sub_m 1
  job 20010 subM4 100 --seed 24 --proto h2 --mode full --stream 1 --force_sub_m 4
  job 20010 def_pad4k 150 --seed 31 --proto h2 --mode full --stream 1 --pad 4096
  job 20010 def_pad64k 150 --seed 33 --proto h2 --mode full --stream 1 --pad 65536
  job 20010 def_combined 150 --seed 35 --proto h2 --mode server_state --stream 0 --pad 16384 --chaff 0.3 --jitter_ms 500; }
lane2(){
  job 20020 main_seed2 350 --seed 2 --proto h2 --mode full --stream 1
  job 20020 h1_stream 200 --seed 11 --proto h1 --mode full --stream 1
  job 20020 gzip 200 --seed 13 --proto h2 --mode gzip --stream 1
  job 20020 fixed5k 200 --seed 16 --proto h2 --mode full --stream 1 --size_mode fixed --fixed_size 5000
  job 20020 subM2 100 --seed 22 --proto h2 --mode full --stream 1 --force_sub_m 2
  job 20020 subM8 100 --seed 28 --proto h2 --mode full --stream 1 --force_sub_m 8
  job 20020 def_pad16k 150 --seed 32 --proto h2 --mode full --stream 1 --pad 16384
  job 20020 def_jitter_batch 150 --seed 34 --proto h2 --mode full --stream 0 --jitter_ms 500; }
lane1 & lane2 & wait; echo ALLDONE >> $O/run.log
