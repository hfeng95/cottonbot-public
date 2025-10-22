@echo off
python cotton_train.py ^
  --data "sample/nykko.txt" ^
  --output "checkpoint/nykko" ^
  --model "LiquidAI/LFM2-350M" ^
  --epochs 5
pause
