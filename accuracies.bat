FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        python cli.py accuracy %%D %%A std:test --state-dict-path trained-models\best-classifiers\%%D-%%A.pth
    )
)