nvidia-smi

llm="llama-3.2-3b-it"

num_samples=1
temperature=0.0


base_dir="."
hf_cache="${base_dir}/cache"

bsz=16
n_new=512

gpus="0,1,2,3"
num_processes=4


PYTHONPATH=$PYTHONPATH:$(pwd) \
CUDA_VISIBLE_DEVICES=$gpus \
out_dir="${base_dir}/data/value_function_features/ns=${num_samples}/temp=${temperature}/${llm}"
accelerate launch --multi_gpu --num_processes ${num_processes} src/get_activations_only.py \
    --model_name "$llm" \
    --dataset_name "HelpSteer2" \
    --num_samples $num_samples \
    --output_path $out_dir \
    --hf_cache_dir $hf_cache \
    --max_prompt_tokens 2048 \
    --max_new_tokens $n_new \
    --batch_size $bsz \
    --temperature $temperature

