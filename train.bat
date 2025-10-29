@echo off
python cotton_train.py ^
  --data "sample/kristen.txt" ^
  --output "checkpoint/kristen" ^
  --model "LiquidAI/LFM2-700M" ^
  --steps 100 ^
  --epochs 1
pause
