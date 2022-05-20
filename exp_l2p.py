"""
Author: Tong
Time: 28-04-2021
"""
map = {"seq-cifar100": "f"}


def generate_cmd():
    cmd_list = []
    info_list = []

    model='l2p_vit'
    dataset="seq-cifar100"
    seed=100
    ptm="vit"
    lr=0.03
    epoch=5
    prob_l=-1

    info = "{description}_{var}_{var1}_{var2}".format(description="e4",
                                                        var=map[dataset], var1=ptm,
                                                        var2="l2p")
    
    cmd = 'python3 -m main --info {info} --seed {seed} ' \
            '--model {model} --area CV --dataset {dataset}  ' \
            '--buffer_size 5000 --minibatch_size 128 --alpha 0.5 --beta 1.0 ' \
            '--csv_log  --lr {lr}  --ptm {ptm} --pw 0.5 --freeze_clf 0 --init_type unif ' \
            '--eval_freq 1 --prob_l {prob_l} ' \
            ' --batch_size 128 --n_epochs {epoch}  --cuda 7'.format(model=model,
                                                                    dataset=dataset,
                                                                    seed=seed,
                                                                    ptm=ptm,
                                                                    lr=lr,
                                                                    epoch=epoch,
                                                                    info=info,
                                                                    prob_l=prob_l)
                                    
    info_list.append(info)
    cmd_list.append(cmd)
    
    return cmd_list, info_list
