birch sample --config config/smoke.json --output output/smoke.json --seed 0
birch draw --input output/smoke.json --output figs/smoke.pdf
birch data --input output/smoke.json --output output/smoke_data.json
