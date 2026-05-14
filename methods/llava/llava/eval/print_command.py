# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/eval/print_command.py

image_folder = "/path/to/images"
modelname = "/path/to/checkpoint_root"
question_file = "/path/to/questions.jsonl"
answer = "/path/to/answers/output_prefix"

s = f"python model_vqa.py --model-path {modelname}  --question-file {question_file} --image-folder {image_folder} --answers-file {answer}.jsonl --model-base liuhaotian/llava-v1.5-13b"
print(s)
for i in range(200,3000,400):
    s = f"python model_vqa.py --model-path {modelname}/checkpoint-{i}  --question-file {question_file} --image-folder {image_folder} --answers-file {answer}{i}.jsonl --model-base liuhaotian/llava-v1.5-13b"
    print(s)
    
# 
print("\n\n")
image_folder = "/path/to/refined_images"
question_file = "/path/to/questions.jsonl"


answerrefine = answer +"-refine"

s = f"python model_vqa.py --model-path {modelname}  --question-file {question_file} --image-folder {image_folder} --answers-file {answerrefine}.jsonl --model-base liuhaotian/llava-v1.5-13b"
print(s)
for i in range(200,3000,400):
    s = f"python model_vqa.py --model-path {modelname}/checkpoint-{i}  --question-file {question_file} --image-folder {image_folder} --answers-file {answerrefine}{i}.jsonl --model-base liuhaotian/llava-v1.5-13b"
    print(s)
    
# 


print("\n\n")
image_folder = "/path/to/vg_images"
question_file = "/path/to/vg_questions.jsonl"


answer = "/".join(answer.split("/")[:-1]) + "/VG_"+ answer.split("/")[-1]
s = f"python model_vqa.py --model-path {modelname}  --question-file {question_file} --image-folder {image_folder} --answers-file {answer}.jsonl --model-base liuhaotian/llava-v1.5-13b"
print(s)
for i in range(200,3000,400):
    s = f"python model_vqa.py --model-path {modelname}/checkpoint-{i}  --question-file {question_file} --image-folder {image_folder} --answers-file {answer}{i}.jsonl --model-base liuhaotian/llava-v1.5-13b"
    print(s)
    
# 

