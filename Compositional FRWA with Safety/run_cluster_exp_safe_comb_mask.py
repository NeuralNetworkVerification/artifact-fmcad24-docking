from training_exp_safe_comb_mask import train_model, retrain_model, LyapunovNetworkV, TwoDimDocking
from queries_safe_comb_mask import safe_descent_cond_check
import torch
from generate_combined_model_torch_comb import combined_model, combine_prev_cur
from convertsinglenetwork import single_model
import os
from datetime import datetime

pos_limit = 5

index = 0
threshold = 0
#stores dataset at out_data_file, model at out_model_file, and controller at out_controller_file
controller_file = "controller_"
model_file = "cert_"
train_file = "train_"
val_file = "val_"
output_comb_model = "combined_"
output_models = "models_"
prev_model_file = "/barrett/scratch/udayanm/for_cluster/f_3_safe_mask_4/dummy/slurm-1/models/cert_3.pt"
prev_model_onnx = "/barrett/scratch/udayanm/for_cluster/f_3_safe_mask_4/dummy/slurm-1/models/cert_3.onnx"
prev_pos_bound = 3
initial_controller_file = "/barrett/scratch/udayanm/for_cluster/f_3_safe_mask_4/dummy/slurm-1/controllers/controller_3.pt"

two_dim_docking = TwoDimDocking(4,4,0.5,prev_model_file,pos_limit)
vel_limit = two_dim_docking.gen_vel_limit(torch.Tensor([pos_limit]), torch.Tensor([pos_limit]))[0].numpy() + 0.01

out_controller_folders = "controllers/"
out_model_folders = "models/"
out_data_folders = "data/"
out_comb_folders = "combined/"

out_timing_counterexample_file = "times_counterexamples.txt"

if not os.path.isdir(out_controller_folders):
	os.mkdir(out_controller_folders)
	os.mkdir(out_model_folders)
	os.mkdir(out_data_folders)
	os.mkdir(out_comb_folders)
	os.mkdir("counterexamples/")

cur_train_file = out_data_folders + train_file + str(index) + ".pt"
cur_val_file = out_data_folders + val_file + str(index) + ".pt"
cur_model_file = out_model_folders + model_file + str(index) + ".pt"
cur_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
cur_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
cur_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
cur_models_file = out_comb_folders + output_models + str(index) + ".onnx"
threshold = 0
st_train_time = datetime.now()
train_model(pos_limit-1, pos_limit, vel_limit, cur_train_file, cur_val_file, cur_model_file, cur_controller_file, threshold, initial_controller_file, prev_model_file, prev_model_onnx, prev_pos_bound)
end_train_time = datetime.now()
diff = end_train_time - st_train_time
f = open(out_timing_counterexample_file, "a")
print("Total training time for model index", str(index), ":", str(diff.seconds))
f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
f.close()
combined_model(cur_model_file, cur_controller_file, prev_model_file, cur_comb_file)
combine_prev_cur(cur_model_file, prev_model_file, cur_models_file)
single_model(cur_model_file, cur_model_onnx_file)
st_ver_time = datetime.now()
ret, ret_ranges, failed = safe_descent_cond_check(cur_comb_file, cur_model_onnx_file, cur_models_file, prev_pos = prev_pos_bound, safe_pos = pos_limit - 1, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit)
end_ver_time = datetime.now()
diff_ver_time = end_ver_time - st_ver_time
f = open(out_timing_counterexample_file, "a")
print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds))
f.write("Total verification time for verification index " + str(index) + ":" + str(diff_ver_time.seconds) + "\n")
f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
f.write("Number of failed cases: " + str(len(failed)) + "\n")
f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
f.close()
while (len(ret) > 0):
    index += 1
    next_train_file = out_data_folders + train_file + str(index) + ".pt"
    next_val_file = out_data_folders + val_file + str(index) + ".pt"
    next_model_file = out_model_folders + model_file + str(index) + ".pt"
    next_model_onnx_file = out_model_folders + model_file + str(index) + ".onnx"
    next_controller_file = out_controller_folders + controller_file + str(index) + ".pt"
    next_comb_file = out_comb_folders + output_comb_model + str(index) + ".onnx"
    next_models_file = out_comb_folders + output_models + str(index) + ".onnx"
    st_train_time = datetime.now()
    retrain_model(index - 1, torch.Tensor(ret), torch.Tensor(ret_ranges), pos_limit-1, pos_limit, vel_limit, cur_train_file, next_train_file, next_val_file, cur_model_file, cur_controller_file, next_model_file, next_controller_file, threshold, prev_model_file, prev_pos_bound)
    end_train_time = datetime.now()
    diff = end_train_time - st_train_time
    f = open(out_timing_counterexample_file, "a")
    print("Total training time for model index", str(index), ":", str(diff.seconds))
    f.write("Total training time for model index " + str(index) + ":" + str(diff.seconds) + "\n")
    f.close()
    combined_model(next_model_file, next_controller_file, prev_model_file, next_comb_file)
    combine_prev_cur(next_model_file, prev_model_file, next_models_file)
    single_model(next_model_file, next_model_onnx_file)
    st_ver_time = datetime.now()
    ret, ret_ranges, failed = safe_descent_cond_check(next_comb_file, next_model_onnx_file, next_models_file, prev_pos = prev_pos_bound, safe_pos = pos_limit - 1, limit_pos = pos_limit, docking_pos = 0.35, vel_limit = vel_limit)
    end_ver_time = datetime.now()
    diff_ver_time = end_ver_time - st_ver_time
    f = open(out_timing_counterexample_file, "a")
    print("Total verification time for verification index", str(index), ":", str(diff_ver_time.seconds) + "\n")
    f.write("Total training time for verification index " + str(index) + ":" + str(diff_ver_time.seconds) + "\n")
    f.write("Number of verification counterexamples: " + str(len(ret)) + "\n")
    f.write("Number of failed cases: " + str(len(failed)) + "\n")
    f.write("Verification counterexamples: " + str(ret_ranges) + "\n")
    f.close()
    cur_train_file, cur_val_file, cur_model_file, cur_model_onnx_file, cur_controller_file, cur_models_file = next_train_file, next_val_file, next_model_file, next_model_onnx_file, next_controller_file, next_models_file

print(failed)
