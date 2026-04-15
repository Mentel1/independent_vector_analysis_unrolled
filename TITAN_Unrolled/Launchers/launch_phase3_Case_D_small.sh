#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m TITAN_Unrolled.main --num_case 3 --K 5 --N 10 --opt SGD --training_mode local --archi tied --step_size 5 --num_layers 200 &
CUDA_VISIBLE_DEVICES=1 python -m TITAN_Unrolled.main --num_case 3 --K 5 --N 10 --opt SGD --training_mode local --archi untied --step_size 5 --num_layers 200 &
CUDA_VISIBLE_DEVICES=2 python -m TITAN_Unrolled.main --num_case 3 --K 5 --N 10 --opt SGD --training_mode local --archi inertial-tied --step_size 5 --num_layers 200 &
CUDA_VISIBLE_DEVICES=3 python -m TITAN_Unrolled.main --num_case 3 --K 5 --N 10 --opt SGD --training_mode local --archi inertial-untied --step_size 5 --num_layers 200 &
wait
echo "Phase 3 terminée !"