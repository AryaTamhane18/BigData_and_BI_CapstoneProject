#!/bin/bash

echo "Creating data directories..."

mkdir -p data/raw
mkdir -p data/clean

echo "Please manually download the dataset from:"
echo "https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis?utm_source=chatgpt.com&select=DataCoSupplyChainDataset.csv"

echo ""
echo "Place the downloaded CSV files inside:"
echo "data/raw/"