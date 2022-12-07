#!/bin/bash
for threshold in 0.5 1.0 2.0 3.0
do  
    echo "The threshold used to update template is $threshold."  
    python tracking/test.py ostrack vitb_256_mae_ce_32x4_ep300 --dataset uav --template_interval -1 --threshold $threshold
    mv /home/imlab/Documents/UAV/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300  /home/imlab/Documents/UAV/OSTrack/output/test/tracking_results/ostrack/vitb_256_mae_ce_32x4_ep300_thre_$threshold  
    python tracking/analysis_results.py --threshold $threshold --model 256
done  
for threshold in 0.1 0.5 1.0 2.0 3.0
do  
    python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset uav --template_interval -1 --threshold $threshold
    mv /home/imlab/Documents/UAV/OSTrack/output/test/tracking_results/ostrack/vitb_384_mae_ce_32x4_ep300  /home/imlab/Documents/UAV/OSTrack/output/test/tracking_results/ostrack/vitb_384_mae_ce_32x4_ep300_thre_$threshold  
    python tracking/analysis_results.py --threshold $threshold --model 384
done  