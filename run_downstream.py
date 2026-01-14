import sys
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/LOBCAST_Clean")


import shutil
#TODO: Update directories so that new constants file in in lob_bench and it updates the one in LOBCAST.
shutil.copyfile("/data1/sascha/lob_bench/lobcast_src/constants.py","/data1/sascha/LOBCAST_Clean/src/constants.py")

import LOBCAST_Clean.src.utils.utils_training_loop as tlu
import src.constants as cst


from LOBCAST_Clean.src.utils.utils_dataset import prepare_data_lob,prepare_data_fi,prepare_data_meta
import src.utils.utils_dataset as utils_dataset
from lob_bench.lobcast_src.lobcast_override import LOBBenchDataset,Configuration,read_sub_routine
import src.utils.utilis_lobster_datasource as uld



uld.read_sub_routine = read_sub_routine
utils_dataset.LOBDataset = LOBBenchDataset

DEFAULT_SEEDS = set(range(500, 505))

DEFAULT_FORWARD_WINDOWS = [
    cst.WinSize.EVENTS1,
    cst.WinSize.EVENTS2,
    cst.WinSize.EVENTS3,
    cst.WinSize.EVENTS5,
    cst.WinSize.EVENTS10
]




def experiment_lobster(execution_plan, dataset, PREFIX=None, is_debug=False, json_dir=None, target_dataset_meta=None, peri=None):

    servers = [server for server in execution_plan.keys()]
    PREFIX, server_name, server_id, n_servers = tlu.experiment_preamble(PREFIX, servers)
    lunches_server = execution_plan[server_name]

    for mod, plan in lunches_server:
        seeds = plan['seed']
        seeds = DEFAULT_SEEDS if seeds == 'all' else seeds

        for see in seeds:
            forward_windows = plan['forward_windows']
            forward_windows = DEFAULT_FORWARD_WINDOWS if forward_windows == 'all' else forward_windows

            for window_forward in forward_windows:
                for percent_train in plan['real_percent_train']:
                    print(f"Running LOB experiment: model={mod}, fw={window_forward.value}, seed={see}")

                    try:
                        cf: Configuration = Configuration(PREFIX)
                        cf.SEED = see

                        tlu.set_seeds(cf)

                        cf.CHOSEN_DATASET = dataset
                        if mod == cst.Models.METALOB:
                            cf.CHOSEN_DATASET = cst.DatasetFamily.META
                            cf.TARGET_DATASET_META_MODEL = target_dataset_meta
                            cf.JSON_DIRECTORY = json_dir

                        cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = plan['stocks_train']
                        cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = plan['stocks_test']
                        cf.TrainRealPercent = percent_train
                        cf.CHOSEN_PERIOD = plan['period']
                        cf.LOBBENCH_MODEL=plan['model_to_bench']

                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value
                        cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                        cf.CHOSEN_MODEL = mod

                        cf.IS_WANDB = int(not is_debug)
                        cf.IS_TUNE_H_PARAMS = int(not is_debug)

                        tlu.run(cf)

                    except KeyboardInterrupt:
                        print("There was a problem running on", server_name.name, "LOB experiment on {}, with K+={}".format(mod, window_forward))
                        sys.exit()

if __name__ == "__main__":
    

    EXE_PLAN = {
        cst.Servers.ANY: [
            # (cst.Models.MLP, {'forward_windows': [cst.WinSize.EVENTS10,
            #                                       cst.WinSize.EVENTS20,
            #                                       cst.WinSize.EVENTS50,
            #                                       cst.WinSize.EVENTS30,
            #                                       cst.WinSize.EVENTS100,],
            #                                         #  cst.WinSize.EVENTS2,
            #                                         #  cst.WinSize.EVENTS3,
            #                                         #  cst.WinSize.EVENTS5,
            #                                         #  cst.WinSize.EVENTS10],
            #                     'seed': [500],
            #                     'stocks_train': cst.Stocks.GOOG,
            #                     'stocks_test': cst.Stocks.GOOG,
            #                     'real_percent_train': [1],
            #                     'model_to_bench':cst.LBModels.S5,
            #                     'period': cst.Periods.JAN2023_S5_GOOG, 
            #                     }),
            (cst.Models.MLP, {'forward_windows': [cst.WinSize.EVENTS10,
                                                  cst.WinSize.EVENTS20,
                                                  cst.WinSize.EVENTS50,
                                                  cst.WinSize.EVENTS30,
                                                  cst.WinSize.EVENTS100,],
                                                    #  cst.WinSize.EVENTS2,
                                                    #  cst.WinSize.EVENTS3,
                                                    #  cst.WinSize.EVENTS5,
                                                    #  cst.WinSize.EVENTS10],
                                'seed': [500],
                                'stocks_train': cst.Stocks.GOOG,
                                'stocks_test': cst.Stocks.GOOG,
                                'real_percent_train': [0.5],
                                'model_to_bench':cst.LBModels.S5,
                                'period': cst.Periods.JAN2023_S5_GOOG, 
                                }),
            (cst.Models.MLP, {'forward_windows': [cst.WinSize.EVENTS10,
                                                  cst.WinSize.EVENTS20,
                                                  cst.WinSize.EVENTS50,
                                                  cst.WinSize.EVENTS30,
                                                  cst.WinSize.EVENTS100,],
                                                    #  cst.WinSize.EVENTS2,
                                                    #  cst.WinSize.EVENTS3,
                                                    #  cst.WinSize.EVENTS5,
                                                    #  cst.WinSize.EVENTS10],
                                'seed': [500],
                                'stocks_train': cst.Stocks.GOOG,
                                'stocks_test': cst.Stocks.GOOG,
                                'real_percent_train': [0],
                                'model_to_bench':cst.LBModels.S5,
                                'period': cst.Periods.JAN2023_S5_GOOG, 
                                }),
            (cst.Models.BINCTABL, {'forward_windows': [cst.WinSize.EVENTS10,
                                                  cst.WinSize.EVENTS20,
                                                  cst.WinSize.EVENTS50,
                                                  cst.WinSize.EVENTS30,
                                                  cst.WinSize.EVENTS100,],
                                                    #  cst.WinSize.EVENTS2,
                                                    #  cst.WinSize.EVENTS3,
                                                    #  cst.WinSize.EVENTS5,
                                                    #  cst.WinSize.EVENTS10],
                                'seed': [500],
                                'stocks_train': cst.Stocks.GOOG,
                                'stocks_test': cst.Stocks.GOOG,
                                'real_percent_train': [1],
                                'model_to_bench':cst.LBModels.S5,
                                'period': cst.Periods.JAN2023_S5_GOOG, 
                                }),
            (cst.Models.BINCTABL, {'forward_windows': [cst.WinSize.EVENTS10,
                                                  cst.WinSize.EVENTS20,
                                                  cst.WinSize.EVENTS50,
                                                  cst.WinSize.EVENTS30,
                                                  cst.WinSize.EVENTS100,],
                                                    #  cst.WinSize.EVENTS2,
                                                    #  cst.WinSize.EVENTS3,
                                                    #  cst.WinSize.EVENTS5,
                                                    #  cst.WinSize.EVENTS10],
                                'seed': [500],
                                'stocks_train': cst.Stocks.GOOG,
                                'stocks_test': cst.Stocks.GOOG,
                                'real_percent_train': [0.5],
                                'model_to_bench':cst.LBModels.S5,
                                'period': cst.Periods.JAN2023_S5_GOOG, 
                                }),
            (cst.Models.BINCTABL, {'forward_windows': [cst.WinSize.EVENTS10,
                                                  cst.WinSize.EVENTS20,
                                                  cst.WinSize.EVENTS50,
                                                  cst.WinSize.EVENTS30,
                                                  cst.WinSize.EVENTS100,],
                                                    #  cst.WinSize.EVENTS2,
                                                    #  cst.WinSize.EVENTS3,
                                                    #  cst.WinSize.EVENTS5,
                                                    #  cst.WinSize.EVENTS10],
                                'seed': [500],
                                'stocks_train': cst.Stocks.GOOG,
                                'stocks_test': cst.Stocks.GOOG,
                                'real_percent_train': [0],
                                'model_to_bench':cst.LBModels.S5,
                                'period': cst.Periods.JAN2023_S5_GOOG, 
                                }),
            # (
            ]}

    experiment_lobster(
        EXE_PLAN,
        dataset=cst.DatasetFamily.LOB,
        PREFIX='DOWNSTREAM_EVAL',
        is_debug=True,
        json_dir="final_data/LOB-FEB-TESTS/jsons/",
        target_dataset_meta=cst.DatasetFamily.LOB,
    )

