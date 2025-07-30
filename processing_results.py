import pandas as pd
import os

BATCH_SIZE=100
NUM_EPOCH=5
NAME_PREFIX="PhaseC_test"
MODEL_DIR="multi_attack_trained_models"
EVAL_BASE="AES_PTv2_D"

base_path = 'multi_attack_trained_models/PhaseB_test_phaseB/all_rank.csv'
base_df = pd.read_csv(base_path)
print(base_df['0'][50])

all_res = []
for dev_no in range(1,5):
    print(dev_no)
    for num_trace in [500, 800]:
        print(num_trace)
        for model_id in range(10,20):
            phaseB_path = 'multi_attack_trained_models/PhaseB_test_phaseB/all_rank_D{}.csv'.format(dev_no)
            phaseB_df = pd.read_csv(base_path)
            model_path = 'PhaseC_test_model{}_trace{}_D{}'.format(model_id, num_trace, dev_no)
            fpath = os.path.join(MODEL_DIR, model_path)
            csv_path = 'all_rank_model{}_5_{}_20_D{}.csv'.format(model_id, num_trace, dev_no)
            fpath = os.path.join(fpath, csv_path)
            df = pd.read_csv(fpath)
            base_val = phaseB_df[str(model_id)][int(num_trace/10)]
            curr_val = df['Mean_ranks'][int(num_trace/10)]
            print(base_val, curr_val)