#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m TITAN_Unrolled.main --num_case 0 --K 20 --N 30 --opt SGD --training_mode local --archi untied --step_size 5 &
CUDA_VISIBLE_DEVICES=1 python -m TITAN_Unrolled.main --num_case 0 --K 20 --N 30 --opt SGD --training_mode end-to-end --archi untied --step_size 5 &
CUDA_VISIBLE_DEVICES=2 python -m TITAN_Unrolled.main --num_case 0 --K 20 --N 30 --opt Adam --training_mode local --archi untied --step_size 5 &
CUDA_VISIBLE_DEVICES=3 python -m TITAN_Unrolled.main --num_case 0 --K 20 --N 30 --opt Adam --training_mode end-to-end --archi untied --step_size 5 &
wait
echo "Phase 1 terminée !"