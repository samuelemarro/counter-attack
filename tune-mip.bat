FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        IF NOT EXIST gurobi-parameter-sets\%%A.prm (
            python cli.py tune-mip %%D %%A std:test linf gurobi-parameter-sets\%%A.prm --state-dict-path trained-models\best-classifiers\%%D-%%A.pth
        )
    )
)