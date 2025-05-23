nvidia-smi

llm="llama-3.2-3b-it"
num_samples=1
train_temp="0.0"
base_dir="."
model_path="${base_dir}/models"
output_base_path="./data/inference_intervention"
hf_cache="${base_dir}/cache"
gpus="0,1,2,3"
hidden_dims="[3072, 3072, 3072]"
bsz=16
n_new=512
intervene=False
target="[4,4,4,2,2]"
output_postfix="{tgt=${target}}"
lr=5e-2
n_step=1000 # place holder as our intervention is closed-form
infer_temp=0.0
iteration=0
num_processes=4
value_model_fnm="25-05-22_Value-Func-{md=llama-3.2-3b-it,dset=hs2}_{#l=4,lr=1.0e-04,bsz=16,temp=0.0,lambda=0.9}"
value_model_fnm_path="${value_model_fnm}/best_model.pth"


output_dir_nm="Inference-Result_{md=${llm}}-ns=${num_samples}-train_temp=${train_temp}"

if [ "$intervene" == "True" ]; then
    if [ "$iteration" -eq 0 ]; then
        pre_inference_path="${output_base_path}/${output_dir_nm}/infer_temp=${infer_temp}/base"
    elif [ "$regenerate" -gt 0 ]; then
        pre_inference_path="${output_base_path}/${output_dir_nm}/infer_temp=${infer_temp}/${value_model_fnm}/{lr=${lr},#s=${n_step}}/edited_${output_postfix}/iteration=$((iteration-1))"
    fi
    output_dir_nm="${output_dir_nm}/infer_temp=${infer_temp}/${value_model_fnm}/{lr=${lr},#s=${n_step}}"
    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/inference_intervention.py \
        --model_name "${llm}" \
        --dataset_name "HelpSteer2" \
        --use_intervention $intervene \
        --value_model_path "${model_path}/${value_model_fnm_path}" \
        --output_path "${output_base_path}/${output_dir_nm}" \
        --output_postfix "${output_postfix}" \
        --hf_cache_dir "${hf_cache}" \
        --hidden_dims "${hidden_dims}" \
        --max_new_tokens $n_new \
        --batch_size $bsz \
        --intervene_lr $lr \
        --intervene_steps $n_step \
        --target $target \
        --temp $infer_temp \
        --num_processes $num_processes \
        --iteration $iteration \
        --pre_inference_path "${pre_inference_path}"
elif [ "$intervene" == "False" ]; then
    output_dir_nm="${output_dir_nm}/infer_temp=${infer_temp}/base"

    PYTHONPATH=$PYTHONPATH:$(pwd) \
    CUDA_VISIBLE_DEVICES=$gpus \
    python src/inference_intervention.py \
        --model_name "${llm}" \
        --dataset_name "HelpSteer2" \
        --use_intervention $intervene \
        --value_model_path "${model_path}/${value_model_fnm_path}" \
        --output_postfix "${output_postfix}" \
        --output_path "${output_base_path}/${output_dir_nm}" \
        --hf_cache_dir "${hf_cache}" \
        --hidden_dims "${hidden_dims}" \
        --max_new_tokens $n_new \
        --batch_size $bsz \
        --intervene_lr $lr \
        --intervene_steps $n_step \
        --target $target \
        --temp $infer_temp \
        --num_processes $num_processes
fi
