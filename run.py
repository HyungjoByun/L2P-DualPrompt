import os
import time
import transformers
import exp_l2p, exp_dualp

transformers.logging.set_verbosity(50)

def batch_run_interactive(cmd_list: [str], order=1):
    # print(cmd_list)
    for i in cmd_list[::order]:
        print(i)
    for i in cmd_list[::order]:
        try:
            i = i + "  --pltf m"
            os.system(i)
            time.sleep(10)
        except:
            print(i, " failed!")


if __name__ == '__main__':

    # select experiment to do

    exp = exp_l2p # exp_dualp
    cmd_list, info_list = exp.generate_cmd()
    
    # cmd: select the running platform and the corresponding shell template
    platform = {0: "m3", 1: "group"}
    pltf = 1
    # cmd: running in an interactive session
    batch_run_interactive(cmd_list, order=1)
