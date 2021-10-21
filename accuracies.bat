FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        FOR %%T IN (standard, weight-pruned) DO (
            python cli.py accuracy %%D %%A std:test --state-dict-path trained-models\classifiers\relu\%%T\%%D-%%A.pth
        )
    )
)