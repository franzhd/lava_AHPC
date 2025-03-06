from config import *
import os, sys
from pathlib import Path
import argparse

sys.path.append('../../../src')
sys.path.append('../../../src/utils/')
from spiking_dataloader import WISDM_spiking_dataloader, WisdmDatasetParser
from inibitory_network.inibitory_network_h1_test import InibitoryLifNet
from lava.proc.io.sink import RingBuffer as SinkRingBuffer
from lava.proc.io.source import RingBuffer as SourceRingBuffer
from lava.proc.monitor.process import Monitor
from output_process import OutputProcess
from lava.proc.monitor.process import Monitor
import matplotlib.pyplot as plt
import numpy as np
import nni
from tqdm import tqdm
from lava.proc.learning_rules.stdp_learning_rule import STDPLoihi as STDP


from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense, LearningDense
from lava.utils.weightutils import SignMode
from lif_mod import LIFEncoder
from dense_mod import DenseEncoder

from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg

import json

def deserialize_dict(json_str):
    def convert(obj):
        if isinstance(obj, list):
            return np.array(obj).astype(np.int64)
        if isinstance(obj, int):
            return np.int64(obj)
        return obj
    
    return json.loads(json_str, object_hook=lambda d: {k: convert(v) for k, v in d.items()})

def serialize_dict(data_dict):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    return json.dumps(data_dict, default=convert)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, help='nome del config file per creare la cartella adeguata')
    args = parser.parse_args()

    path = f"{Path.home()}/{ORIGINAL_NETWORK_PATH}"
    data = np.load(path,allow_pickle=True)


    linear1_w= data['linear1']
    leaky1_vth= data['leaky1_vth']
    leaky1_betas= 1-data['leaky1_betas'] 
    leaky1_betas= leaky1_betas if leaky1_betas >= 0 else np.zeros(leaky1_betas.shape)
    print(f"leaky1_betas: {leaky1_betas}")
    print(f"leaky1_vth: {leaky1_vth}")
    linear2_w = data['linear2']
    leaky2_vth= data['recurrent_vth']
    leaky2_betas= 1 - data['recurrent_betas']
    leaky2_betas= leaky2_betas if  leaky2_betas >= 0 else np.zeros(leaky2_betas.shape)
    print(f"leaky2_betas: {leaky2_betas}")
    print(f"leaky2_vth: {leaky2_vth}")

    recurrent_in_weights = data['input_dense']
    recurrent_out_weights = - data['output_dense']
    recurrent_vth = data['activation_vth']
    recurrent_leaky_betas = 1 - data['activation_betas']
    recurrent_leaky_betas= recurrent_leaky_betas if recurrent_leaky_betas >= 0 else np.zeros(recurrent_leaky_betas.shape)
    print(f"recurrent_leaky_betas: {recurrent_leaky_betas}")
    print(f"recurrent_vth: {recurrent_vth}")

    linear3_w = data['linear3']
    leaky3_vth= data['leaky2_vth']
    leaky3_betas= 1 - data['leaky2_betas']
    leaky3_betas= leaky3_betas if leaky3_betas >= 0 else np.zeros(leaky3_betas.shape)
    print(f"leaky3_betas: {leaky3_betas}")
    print(f"leaky3_vth: {leaky3_vth}")

    num_samples = NUM_SAMPLES
    signal_step = SIGNAL_STEP
    clear_intervall = CLEAR_INTERVAL
    train_percentage = TRAIN_PERCENTAGE
    time_steps = signal_step + clear_intervall

    with open(f'{Path.home()}/{FIXED_NETWORK_PATH}', 'r') as json_file:
        loaded_json_str = json_file.read()
    converted_params = deserialize_dict(loaded_json_str)

    params = nni.get_next_parameter()
    try:
        s_stdp = STDP(learning_rate=int(params['learning_rate']),
                      A_plus=int(params['A_plus']),
                      A_minus=int(params['A_minus']),
                      tau_plus=int(params['tau_plus']),
                      tau_minus=int(params['tau_minus']),
                      t_epoch=int(float(num_samples)*train_percentage)*time_steps,
                      x1_impulse= int(params['x1_impulse']),
                      y1_impulse= (params['y1_impulse']),
            )
    except Exception as e:
        nni.report_final_result(0.0)
        sys.exit(0)
    
    linear1 = DenseEncoder(weights=linear1_w, num_message_bits=32, name="linear1")

    leaky1 = LIFEncoder(shape=(linear1_w.shape[0],),
                        u = np.zeros(linear1_w.shape[0]),
                        v = np.zeros(linear1_w.shape[0]),
                        du = 1.0,
                        dv = leaky1_betas,
                        vth=leaky1_vth,
                        log_config=0,
                        name= "leaky1"
                    )
    linear1.a_out.connect(leaky1.a_in)
    name = "linear2"
    linear2 = Dense(**converted_params[name],
                    sign_mode=SignMode.MIXED, name=name)

    linear2.s_in.connect_from(leaky1.s_out)

    name = "leaky2"
    leaky2 = LIF(shape=(linear2_w.shape[0],),
                        **converted_params[name],
                        name= name
                    )
    #sum.a_out.connect(leaky2.a_in)
    linear2.a_out.connect(leaky2.a_in)
    #leaky2.a_in.connect_from(linear2.a_out)
    name = "recurrent_in"
    recurrent_in = Dense(**converted_params[name],
                        name=name)

    leaky2.s_out.connect(recurrent_in.s_in)

    name = "inibitory_leaky"
    ahpc = LIF(shape=(recurrent_in_weights.shape[0],),
                        **converted_params[name],
                        log_config=0,
                        name= name
                    )

    recurrent_in.a_out.connect(ahpc.a_in)
    name = "recurrent_out"
    recurrent_out = Dense(**converted_params[name],
                        name=name)
    recurrent_out.s_in.connect_from(ahpc.s_out)
    recurrent_out.a_out.connect(leaky2.a_in)
    
    name = "linear3"
    linear3 = LearningDense(**converted_params[name],
                            learning_rule=s_stdp,
                            name=name)

    linear3.s_in.connect_from(leaky2.s_out)
    name = "leaky3"
    leaky3 = LIF(shape=(linear3_w.shape[0],),
                        **converted_params[name],
                        log_config=0,
                        name= name
                    )
    leaky3.s_out.connect(linear3.s_in_bap)
    leaky3.a_in.connect_from(linear3.a_out)

    


    dataset = WisdmDatasetParser(f'{Path.home()}/{DATASET_PATH}', norm=None, class_sublset='custom', subset_list=[0, 4, 6, 8, 9, 10, 14])
    val_set = dataset.get_validation_set(shuffle=False, subset=num_samples)
    spiking_loader = WISDM_spiking_dataloader(val_set ,clear_intervall=clear_intervall)
    out_sink = OutputProcess(7,num_samples,time_steps, 0)

    spiking_loader.data_out.connect(linear1.s_in)
    leaky3.s_out.connect(out_sink.spikes_in)
    out_sink.label_in.connect_from(spiking_loader.label_out)

    
    for i in tqdm(range(num_samples)):
      out_sink.run(condition=RunSteps(num_steps=time_steps),
                  run_cfg=Loihi1SimCfg(select_sub_proc_model=True,
                  select_tag='fixed_pt'))
      leaky1.v.set(np.zeros(leaky1.v.shape))
      leaky1.u.set(np.zeros(leaky1.u.shape))
      leaky2.v.set(np.zeros(leaky2.v.shape))
      leaky2.u.set(np.zeros(leaky2.u.shape))
      leaky3.v.set(np.zeros(leaky3.v.shape))
      leaky3.u.set(np.zeros(leaky3.u.shape))
      ahpc.v.set(np.zeros(ahpc.v.shape))
      ahpc.u.set(np.zeros(ahpc.u.shape))
      linear1.a_buff.set(np.zeros(linear1.a_buff.shape))
      linear2.a_buff.set(np.zeros(linear2.a_buff.shape))
      linear3.a_buff.set(np.zeros(linear3.a_buff.shape))

      updated_weights = linear3.weights.get()

    converted_params['linear3']['weights'] = updated_weights
    ground_truth = val_set[1][:num_samples]
    predictions = out_sink.pred_labels.get().astype(int)
    # Stop the execution
    out_sink.stop()

    total_accuracy = np.sum(ground_truth==predictions)/ground_truth.size * 100
    train_accuracy = np.sum(ground_truth[:int(num_samples*train_percentage)]==predictions[:int(num_samples*train_percentage)])/int(num_samples*train_percentage) * 100
    val_accuracy = np.sum(ground_truth[int(num_samples*train_percentage):]==predictions[int(num_samples*train_percentage):])/int(num_samples*(1-train_percentage)) * 100

    



    os.chdir(f'{Path.home()}/lava_AHPC/nni_experiments/{args.trial_path}/results/{nni.get_experiment_id()}/trials/{nni.get_trial_id()}')
    trained_folder = TRAIN_FOLDER_NAME
    os.makedirs(trained_folder, exist_ok=True)
    with open(f'{trained_folder}/finetuned_network.json', 'w') as json_file:
        json_file.write(serialize_dict(converted_params))
    nni.report_final_result(val_accuracy)