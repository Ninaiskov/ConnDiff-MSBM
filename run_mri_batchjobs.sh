for ini in 1 2 3 4 5 6 7 8 9 10
do
for noc in 1 2 3 4 5 6 7 8 9 10 15 25 50 100
do
    sed -i '$ d' submit_big.sh
    echo "python3 main.py --dataset hcp --noc $noc --model_type parametric --maxiter_gibbs 100" >> submit_big.sh
    bsub < submit_big.sh
done
done
