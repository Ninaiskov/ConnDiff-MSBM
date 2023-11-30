for ini in 1 2 3 4 5
do
for K in 2 5 10
do
for Nc_type in 'balanced' 'unbalanced'
do
for alpha in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.98
do
    sed -i '$ d' submit_hpc.sh
    echo "python3 main.py --dataset synthetic --maxiter_gibbs 100 --K $K --Nc_type $Nc_type --alpha $alpha --model_type parametric --noc 10" >> submit_hpc.sh
    bsub < submit_hpc.sh
done
done
done
done
