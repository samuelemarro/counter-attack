FOR %%D IN (cifar10) DO (
    FOR %%A IN (extra_small, extra_small_2, extra_small_3, extra_small_4) DO (
        IF NOT EXIST gurobi-parameter-sets\%%A.prm (
            python cli.py tune-mip %%D %%A std:test linf gurobi-parameter-sets\%%A.prm --state-dict-path trained-models\best-classifiers\%%D-%%A.pth
        )
    )
)