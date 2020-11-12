FOR %%S IN (5, 10, 25, 50, 100) DO (
    IF NOT EXIST trained-models\classifiers\%1-%2-es%%S-ftr-1000.pth (
        python cli.py train-classifier %1 %2 std:train 1000 trained-models\classifiers\%1-%2-es%%S-ftr-1000.pth --validation-split 0.1 --early-stopping %%S --flip --translation 0.1 --rotation 15
    )
)