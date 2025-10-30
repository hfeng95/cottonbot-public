@echo off
python cotton_train.py ^
  --data "sample/nykko" ^
  --output "checkpoint/nykko" ^
  --model "LiquidAI/LFM2-350M" ^
  --steps 100 ^
  --epochs 1
pause
