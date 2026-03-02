small_model_port=30002
medium_model_port=30000
large_model_port=30001

source ~/miniconda3/etc/profile.d/conda.sh && conda activate Debate
mkdir -p logs

# sweep over all 3 of the models
# 3^3 = 27 total combinations of which models to run
for judge_port in $small_model_port $medium_model_port $large_model_port; do
    for truth_port in $small_model_port $medium_model_port $large_model_port; do
        for gaslight_port in $small_model_port $medium_model_port $large_model_port; do
            echo "Starting judge on port ${judge_port}, truth on port ${truth_port}, gaslight on port ${gaslight_port}"
            python debate_pipeline_sglang.py --gaslight-port $gaslight_port --truth-port $truth_port --judge-port $judge_port --mode full > logs/${judge_port}_${truth_port}_${gaslight_port}.log 2>&1

        done
    done
done