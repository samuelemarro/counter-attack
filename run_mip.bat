FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        python cli.py accuracy %%D %%A std:test --state-dict-path trained-models\best-classifiers\%%D-%%A.pth --max-samples 10
        FOR %%C IN (attack_configurations\mip_1th_240b_0t.cfg, attack_configurations\mip_4th_60b_0t.cfg) DO (
            python cli.py attack %%D %%A std:test mip linf --max-samples 10 --attack-config-file %%C --state-dict-path trained-models\best-classifiers\%%D-%%A.pth
        )
    )
)