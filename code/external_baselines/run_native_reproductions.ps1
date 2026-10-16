param(
    [int]$Seed = 42,
    [int]$FFMSRMVEpochs = 20,
    [int]$FFMSRBooksEpochs = 5,
    [int]$FedDCSREpochs = 10
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root "envs\ffmsr\python.exe"

if (-not (Test-Path $Python)) {
    throw "Missing GPU environment: $Python"
}

& $Python (Join-Path $Root "run_ffmsr_mbmv.py") --source Music --target Movies --seed $Seed --pretrain-epochs 5 --finetune-epochs $FFMSRMVEpochs
& $Python (Join-Path $Root "run_ffmsr_mbmv.py") --source Music --target Books --seed $Seed --pretrain-epochs 1 --finetune-epochs $FFMSRBooksEpochs --books-batch-size 2048 --books-ft-layer 1
& $Python (Join-Path $Root "run_ffmsr_mbmv.py") --source Books --target Movies --seed $Seed --pretrain-epochs 1 --finetune-epochs $FFMSRBooksEpochs --books-batch-size 2048 --books-ft-layer 1

$FedDCSR = Join-Path $Root "FedDCSR"
Push-Location $FedDCSR
try {
    & $Python "main.py" Music Books Movies `
        --method FedDCSR --gpu 0 --epochs $FedDCSREpochs --local_epoch 3 `
        --batch_size 256 --seed $Seed --anneal_cap 0.01 --full_ranking `
        --checkpoint_dir "..\results\feddcsr\seed$Seed" `
        --log_dir "..\results\feddcsr\log" --id "mbmv_seed$Seed"
}
finally {
    Pop-Location
}
