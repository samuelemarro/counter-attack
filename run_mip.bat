FOR %%D IN (mnist, cifar10) DO (
    FOR %%A IN (a, b, c) DO (
        python cli.py accuracy %%D %%A std:test --state-dict-path trained-models\best-classifiers\%%D-%%A.pth --max-samples 10
        ECHO attack_configurations\architecture_specific\mip_1th_240b_0t_7200s_%%D-%%A.cfg
        IF EXIST attack_configurations\architecture_specific\mip_1th_240b_0t_7200s_%%D-%%A.cfg (
            SET CONFIG=attack_configurations\architecture_specific\mip_1th_240b_0t_7200s_%%D-%%A.cfg
        ) ELSE (
            SET CONFIG=attack_configurations\mip_1th_240b_0t_7200s.cfg
        )
        
        python cli.py attack %%D %%A std:test mip linf --max-samples 10 --attack-config-file %CONFIG% --state-dict-path trained-models\best-classifiers\%%D-%%A.pth
    )
)