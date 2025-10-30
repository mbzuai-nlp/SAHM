#!/bin/bash

# Multi-dataset evaluation script
# Runs evaluation on all 4 Arabic MCQ datasets

echo "Starting multi-dataset evaluation for Arabic MCQ datasets..."
echo "Datasets:"
echo "1. SahmBenchmark/arabic-accounting-mcq_eval"
echo "2. SahmBenchmark/arabic-business-mcq_eval" 
echo "3. SahmBenchmark/fatwa-mcq-evaluation_standardized"
echo "4. SahmBenchmark/Sentiment_Analysis_MCQ_eval"
echo ""

# Run the multi-dataset evaluation
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval.py \
    --config config.yaml \
    --models models.yaml \
    --model-type all

echo ""
echo "Multi-dataset evaluation completed!"
echo "Check the outputs/ directory for results organized by dataset."