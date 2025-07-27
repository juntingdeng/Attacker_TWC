for i in 5; do 
    for j in 1.0 2.0 3.0; do 
        python ./ML/GAN_SDR.py "$i" "$j"
    done
done