nvidia-smi

num_samples=1
train_temp="0.0"
llm='llama-3.2-3b-it'

base_dir="."
res_path="${base_dir}/data/inference_intervention/Inference-Result_{md=${llm}}-ns=${num_samples}-train_temp=${train_temp}"
hf_cache="./cache"
gpus="0,1,2,3"
bsz=256
intervene=False
infer_temp=0.0
iteration=0

value_model_fnm="25-05-22_Value-Func-{md=llama-3.2-3b-it,dset=hs2}_{#l=4,lr=1.0e-04,bsz=16,temp=0.0,lambda=0.9}"
lr=5e-2
target="[4,4,4,2,2]"
n_step=1000
output_postfix="{tgt=${target}}"
intervene_meta="lr=${lr},#s=${n_step}"

if [ "$intervene" == 'True' ]; then
    res_path_tmp="${res_path}/infer_temp=${infer_temp}/${value_model_fnm}/{${intervene_meta}}/edited_${output_postfix}/iteration=${iteration}/responses.jsonl"
    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/reward_label.py \
        --reward_model_name "ArmoRM-8B" \
        --batch_size $bsz \
        --response_path "$res_path_tmp" \
        --hf_cache_dir $hf_cache \
        --iteration $iteration
elif [ "$intervene" == 'False' ]; then
    res_path_tmp="${res_path}/infer_temp=${infer_temp}/base/responses.jsonl"
    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/reward_label.py \
        --reward_model_name "ArmoRM-8B" \
        --batch_size $bsz \
        --response_path "$res_path_tmp" \
        --hf_cache_dir $hf_cache \
        --iteration $iteration
fi