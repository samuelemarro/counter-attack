FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        FOR %%S IN (5, 10, 25, 50) DO (
            IF NOT EXIST trained-models\classifiers\%%D-%%A-es%%S-ftr-1000.pth (
                python cli.py train-classifier %%D %%A std:train 1000 trained-models\classifiers\%%D-%%A-es%%S-ftr-1000.pth --validation-split 0.1 --early-stopping %%S --flip --translation 0.1 --rotation 15
            )
        )
    )
)

FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        FOR %%S IN (5, 10, 25, 50) DO (
            python cli.py accuracy %%D %%A std:test --state-dict-path trained-models\classifiers\%%D-%%A-es%%S-ftr-1000.pth
        )
    )
)