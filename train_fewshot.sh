#Sequentially trains aTLAS for few-shot and 50\%, 100\% subsample settings 
for shots in 1 2 4 8 16 0.5 1.0;do
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise --subsample $shots --exp_name results/ViT-B-32_aTLAS/${shots}_shots/
done

#aTLAS x K
for shots in 1 2 4 8 16 0.5 1.0;do
    #aTLAS x 10
    python src/learn_few_shots.py --model=ViT-B-32 --partition 10 --subsample $shots --exp_name results/ViT-B-32_aTLASx10/${shots}_shots/
    #aTLAS x 50
    python src/learn_few_shots.py --model=ViT-B-32 --partition 50 --subsample $shots --exp_name results/ViT-B-32_aTLASx50/${shots}_shots/
done

#aTLAS with LP++ or TIP (sequentially only)
for shots in 1 2 4 8 16 0.5 1.0;do
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise --subsample $shots --exp_name results/ViT-B-32_aTLAS_with_TIP/${shots}_shots/ --adapter tip
    python src/learn_few_shots.py --model=ViT-B-32 --blockwise --subsample $shots --exp_name results/ViT-B-32_aTLAS_with_LPP/${shots}_shots/ --adapter lpp
done


