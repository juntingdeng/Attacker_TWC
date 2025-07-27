import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def loadTimeCntRFF(path, time_stop, thds, num_thds, acc_range, snr=None, 
                        th_acc=None, show_acc_range=False, show_out60sec=False, pretrained=False):
    print("\n========== loadTimeCntRFF started. ==========")
    
    paths = [path] if isinstance(path, str) else path
    files = [os.listdir(p) for p in paths]
    for i in range(len(files)):
        print("files{} len:{}".format(i, len(files[i])))

    cnt=0
    cnt_range=[[] for _ in range(len(acc_range))]

    timer_curve_all={}
    for idx in range(len(paths)):
        path = paths[idx]
        filelist = files[idx]
        for file in filelist:
            file_name_len = 3
            if file.split('.')[-1] == 'npy' and len(file.split("_"))==file_name_len:
                rfflabel = int(file.split('_')[-1].split('.')[0])
                dict_load = np.load(path+file, allow_pickle=True).item()
                val_acc = dict_load[rfflabel][-1][-1]
                if rfflabel not in timer_curve_all.keys():
                    for i in range(len(acc_range)):
                        (x,y) = acc_range[i]
                        if x<=val_acc<y :
                            cnt_range[i].append(rfflabel)
                            cnt+=1
                    if th_acc[0]<=val_acc<th_acc[1]: 
                        timer_curve_all.update(dict_load)
                else:
                    # max_time_pre = timer_curve_all[rfflabel][-2][-1]
                    # val_acc = dict_load[rfflabel][-1][-1]
                    # max_time = dict_load[rfflabel][-2][-1]
                    # if max_time<=max_time_pre:
                    #     timer_curve_all.update(dict_load)
                    #     for i in range(len(acc_range)):
                    #         if rfflabel in cnt_range[i]:
                    #             cnt_range[i].remove(rfflabel)
                    #     for i in range(len(acc_range)):
                    #         (x,y) = acc_range[i]
                    #         if x<=val_acc<y:
                    #             cnt_range[i].append(rfflabel)
                    
                    max_val_acc = timer_curve_all[rfflabel][-1][-1]
                    val_acc = dict_load[rfflabel][-1][-1]
                    max_time = dict_load[rfflabel][-2][-1]
                    if val_acc>max_val_acc:
                        timer_curve_all.update(dict_load)
                        for i in range(len(acc_range)):
                            if rfflabel in cnt_range[i]:
                                cnt_range[i].remove(rfflabel)
                        for i in range(len(acc_range)):
                            (x,y) = acc_range[i]
                            if x<=val_acc<y:
                                cnt_range[i].append(rfflabel)
                                
    print("#cnt_range:", cnt)
    if show_acc_range:
        for i in range(len(acc_range)):
            print("acc_range:{}, cnt:{}".format(acc_range[i], len(cnt_range[i])))

    rfflabels = list(timer_curve_all.keys())
    # print("#{} rfflabels in time_curve_all: {}".format(len(rfflabels), rfflabels))
    if len(rfflabels)<200:
        print("#miss rfflabel:", 200-len(rfflabels), end=' ')
        for i in range(1, 201):
            if i not in rfflabels:
                print(i, end=', ')
        print("/n")
    else:
        print("No missing rffs")
    
    in_time_stop=[]
    out_time_stop=[]
    n_epochs=[]
    best_val_acc = np.zeros(len(timer_curve_all.keys()))
    best_val_acc_intimestop = np.zeros(len(timer_curve_all.keys()))

    cnt_thds=np.zeros(len(thds))
    cnt_thds_intimestop=np.zeros(len(thds))

    acc_num_thds = np.zeros(len(num_thds))
    acc_num_thds_intimestop = np.zeros(len(num_thds))

    for rff in timer_curve_all.keys():
        test_time = timer_curve_all[rff][-2][-1]
        val_acc = timer_curve_all[rff][-1][-1]
        best_val_acc[rff-1] = np.max(timer_curve_all[rff][-1])
        for thdi in range(len(thds)):
            thd = thds[thdi]
            if best_val_acc[rff-1]>=thd:
                cnt_thds[thdi]+=1
        n_epochs.append(len(timer_curve_all[rff][0]))
        if test_time<=time_stop:# and val_acc>=0.9:
            in_time_stop.append(rff)
        else:
            out_time_stop.append(rff)

        train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve = timer_curve_all[rff]

        for i in range(len(test_time)):
            if 0<=time_stop-train_time[i]: # <0.15 when testing with tranfer learning results
                time_stop_train_i=i
            if 0<=time_stop-test_time[i]:
                time_stop_test_i=i
        best_val_acc_intimestop[rff-1] = np.max(timer_curve_all[rff][-1][:time_stop_test_i]) if len(timer_curve_all[rff][-1][:time_stop_test_i])>1 else timer_curve_all[rff][-1][0]
        for thdi in range(len(thds)):
            thd = thds[thdi]
            if best_val_acc_intimestop[rff-1]>=thd:
                cnt_thds_intimestop[thdi]+=1

    best_val_acc_sorted = np.flip(np.sort(best_val_acc))
    best_val_acc_sorted_intimestop = np.flip(np.sort(best_val_acc_intimestop))
    
    print("RFF cnt in thd ranges:")
    for thdi in range(len(thds)):
        thd = thds[thdi]
        print("({}, {})".format(thd, cnt_thds[thdi]), end=", ") if thdi<len(thds)-1 else print("({}, {})".format(thd, cnt_thds[thdi]))
    
    print("Best N RFFs' average acc:")
    for i in range(len(num_thds)):
        num_thd = num_thds[i]
        acc_num_thds[i] = np.mean(best_val_acc_sorted[:num_thd])
        print("({}, {:.4f})".format(num_thd, acc_num_thds[i]), end=", ") if i<len(num_thds)-1 else print("({}, {})".format(num_thd, acc_num_thds[i]))

    print("In timestops {}s, RFF cnt in thd ranges:".format(time_stop))
    for thdi in range(len(thds)):
        thd = thds[thdi]
        print("({}, {})".format(thd, cnt_thds_intimestop[thdi]), end=", ") if thdi<len(thds)-1 else print("({}, {})".format(thd, cnt_thds_intimestop[thdi]))
    
    print("In timestops {}s, Best N RFFs' average acc:".format(time_stop))
    for i in range(len(num_thds)):
        num_thd = num_thds[i]
        acc_num_thds_intimestop[i] = np.mean(best_val_acc_sorted_intimestop[:num_thd])
        print("({}, {:.4f})".format(num_thd, acc_num_thds_intimestop[i]), end=", ") if i<len(num_thds)-1 else print("({}, {})".format(num_thd, acc_num_thds_intimestop[i]))

    print("========== loadTimeCntRFF finished. ==========\n")
    return timer_curve_all, cnt_thds, acc_num_thds, cnt_thds_intimestop, acc_num_thds_intimestop


def plotTimerCurve(timer_curve_all_SNRs, transfer=False, SNRs=None, 
                   plot_zoom=False, save=False, save_path=None):
    print("========== plotTimerCurve started. ==========")
    colors_list=list(mcolors.BASE_COLORS)[:len(SNRs)]
    colors={}
    for i in range(len(SNRs)):
        colors[SNRs[i]] = colors_list[i]
    linetypes=['-.', '-']
    marks=['+', '*']
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if plot_zoom:
        ax_zoom = plt.axes([.50, .33, .38, .40])
        yticks_zoom=[]

    acc60all=[]
    for SNR in SNRs:
        timer_curve_all = timer_curve_all_SNRs[SNR]
        train_step_avg=[]
        test_step_avg=[]
        train_time_1stepoch=[]
        test_time_1stepoch=[]
        train_acc_avg=[]
        test_acc_avg=[]
        max_len=0
        for rfflabel in timer_curve_all.keys():
            train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve = timer_curve_all[rfflabel]
            max_len = max(max_len, len(train_time_step))
            train_time_1stepoch.append(train_time_step[0])
            test_time_1stepoch.append(test_time_step[0])
            train_step_avg.append(np.mean(train_time_step[1:]))
            test_step_avg.append(np.mean(test_time_step[1:]))
            # print("rfflabel:{}, train/1st epoch time:{}, train/avg time step:{}, test/1st epoch time:{}, test/avg time step:{}".format(
            #     rfflabel, train_time_1stepoch[-1], train_step_avg[-1], test_time_1stepoch[-1], test_step_avg[-1]))

        train_time_1stepoch_avg = np.mean(train_time_1stepoch)
        test_time_1stepoch_avg = np.mean(test_time_1stepoch)
        train_step_avg = np.mean(train_step_avg)
        test_step_avg = np.mean(test_step_avg)
        train_time_curve = [train_time_1stepoch_avg]
        test_time_curve = [test_time_1stepoch_avg]
        print("train/avg 1st epoch time:{}, train/avg time step:{}, test/avg 1st epoch time:{}, test/avg time step:{}".format(
                train_time_1stepoch_avg, train_step_avg, test_time_1stepoch_avg, test_step_avg))

        for i in range(1, max_len - len(train_time_curve)+1):
            train_time_curve.append(train_time_curve[-1]+train_step_avg)

        for i in range(1, max_len - len(test_time_curve)+1):
            test_time_curve.append(test_time_curve[-1]+test_step_avg)

        for rfflabel in timer_curve_all.keys():
            train_time_step, train_time, train_acc_fake_curve, test_time_step, test_time, val_acc_fake_curve = timer_curve_all[rfflabel]
            if val_acc_fake_curve[-1]==0:
                continue
            train_acc_fake_curve += [train_acc_fake_curve[-1] for i in range(1, max_len - len(train_acc_fake_curve)+1)]
            val_acc_fake_curve += [val_acc_fake_curve[-1] for i in range(1, max_len - len(val_acc_fake_curve)+1)]
            if train_acc_avg == []:
                train_acc_avg=train_acc_fake_curve
                test_acc_avg = val_acc_fake_curve
            else:
                train_acc_avg=[a+b for (a,b) in zip(train_acc_avg, train_acc_fake_curve)]
                test_acc_avg=[a+b for (a,b) in zip(test_acc_avg, val_acc_fake_curve)]
        n_labels = len(list(timer_curve_all.keys()))
        train_acc_avg = [x/n_labels for x in train_acc_avg]
        test_acc_avg = [x/n_labels for x in test_acc_avg]

        train_acc60, test_acc60 = 0, 0
        train_acc60_t, test_acc_60_t = 0, 0
        train_epoch, test_epoch = 0, 0
        for i in range(len(train_time_curve)):
            if 0<=60-train_time_curve[i]<1: # <0.15 when testing with tranfer learning results
            # if np.abs(train_time_curve[i]-60)<1:
                train_acc60 = max(train_acc60, train_acc_avg[i])
                train_acc60_t = train_time_curve[i]
                train_epoch = i
            if 0<=60-test_time_curve[i]<1:
            # if np.abs(test_time_curve[i]-60)<1e-1:
                test_acc60 = max(test_acc60, test_acc_avg[i])
                test_acc_60_t = test_time_curve[i]
                test_epoch = i

        max_train_time = train_time_curve[-1]
        max_test_time = test_time_curve[-1]
        train_acc60, test_acc60 = np.round(train_acc60, 4), np.round(test_acc60, 4)
        train_acc60_t, test_acc_60_t = np.round(train_acc60_t, 4), np.round(test_acc_60_t, 4)
        acc60all.append(100*test_acc60)
        print("train_acc60:{}, time:{}, epoch:{}; test_acc60:{}, time:{}, epoch:{}".format(train_acc60, train_acc60_t, train_epoch, test_acc60, test_acc_60_t, test_epoch))
        print("max_avg train_acc:{:.4f}, test_acc:{:.4f}".format(max(train_acc_avg), max(test_acc_avg)))
        print("train time last:{}, test time last:{}".format(max_train_time, max_test_time))
        df = pd.DataFrame({'train':train_acc_avg, 'test':test_acc_avg})

        ax.plot(test_time_curve[:500], 100*df['test'][:500], color=colors[SNR], linestyle=linetypes[1], linewidth=2.5, label='SNR: {}dB'.format(SNR))
        if plot_zoom:
            x_zoom = test_time_curve[200:400]
            y_zoom = 100*df['test'][200:400]
            ax_zoom.plot(x_zoom, y_zoom, color=colors[SNR], linewidth=2.5, label='SNR{}dB:{}%'.format(SNR, np.round(100*test_acc60,2)))
            ax_zoom.hlines(y=100*test_acc60, xmin=0, xmax=max_test_time, linewidth=2.2, colors=colors[SNR], linestyles=':')
            yticks_zoom.append(np.round(100*test_acc60,2))
    
    if plot_zoom:
        ax_zoom.set_ylim([84, 95])
        ax_zoom.set_xlim([45, 61.5])
        ax_zoom.set_xticks([47, 50, 55, 60, 63])
        ax_zoom.set_yticks([84, 87, 88, 89, 90, 91, 92, 93, 94])
        ax_zoom.vlines(x=60, ymin=83, ymax=95,linewidth=2.2, colors='r', linestyles=':') 
        ax_zoom.tick_params(axis='both', which='major', labelsize=9)
    
    ax.vlines(x=60, ymin=0, ymax=100, linewidth=2.2, colors='r', linestyles=':')  
    ax.hlines(y=90, xmin=0, xmax=80, linewidth=2.2, colors='r', linestyles=':')  
    ax.set_ylim([0, 101])
    ax.set_xlim([0, 80])
    # plt.xticks([0, 60, 120, 180, 240, 360, 480])
    ax.set_yticks([0, 20, 40, 60, 80, 90, 100])
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.set_ylabel('Average Fake Positive Rate (%)', fontsize=13)
    ax.set_title('Average Fake Positive Rate of \n 219 RFFs over Time with Transfer Learning')
    ax.legend(loc='lower left')
    # plt.show()

    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.plot(SNRs, acc60all, color='b', marker='^', linewidth=2.0)
    ax1.set_xticks(SNRs)
    ax1.set_xlabel('SNR (dB)', fontsize=13)
    ax1.set_ylabel('Average Fake Positive Rate (%)', fontsize=13)
    ax1.set_title('Average Fake Positive Rate of \n 219 RFFs at 60s with Transfer Learning')
    ax1.text(x=SNRs[0]-2.8, y=acc60all[0], s='{}%'.format(np.round(acc60all[0],2))) 
    # ax1.text(x=SNRs[1]+0.8, y=acc60all[1]-0.15, s='{}%'.format(np.round(acc60all[1],2)))   
    # ax1.text(x=SNRs[2]+0.8, y=acc60all[2]-0.15, s='{}%'.format(np.round(acc60all[2],2)))  
    # ax1.text(x=SNRs[3]-0.5, y=acc60all[3]+0.15, s='{}%'.format(np.round(acc60all[3],2)))  
    # ax1.text(x=SNRs[4]+0.8, y=acc60all[4]-0.15, s='{}%'.format(np.round(acc60all[4],2))) 
    # ax1.text(x=SNRs[5]-0.5, y=acc60all[5]+0.15, s='{}%'.format(np.round(acc60all[5],2))) 
    # ax1.text(x=SNRs[6]+0.8, y=acc60all[6]-0.15, s='{}%'.format(np.round(acc60all[6],2))) 
    ax1.grid()
    save_path_ = save_path if save else save_path+"test/"
    if not os.path.exists(save_path_):
        os.makedirs(save_path_)
    fig.savefig(save_path_+"wifi6e_fpr_timecurve_transfer_SNR{}_{}dB.svg".format(SNRs[-1], SNRs[0]))
    fig1.savefig(save_path_+"wifi6e_fpr_60s_transfer_SNR{}_{}dB.svg".format(SNRs[-1], SNRs[0]))
    print("========== plotTimerCurve finished. ==========\n")

def plotCntAcc(paths, SNRs, acc_range, ths_acc, thds, num_thds, time_stops, 
               timelimit=False, pretrained=False, save=False, save_path=None, plottime=False):
    print("\n######## plotCntAcc started. ########")
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    colors=list(mcolors.BASE_COLORS)[:len(SNRs)] # snr
    markers = ['o', 'D', 's'] # time_stop
    linestyles=[':', '-.', '-']

    timer_curve_all_SNRs = {}
    for snri in range(len(SNRs)):
        SNR = SNRs[snri]   
        for time_stopi in range(len(time_stops)):
            time_stop = time_stops[time_stopi]
            timer_curve_all, cnt_thds, acc_num_thds, cnt_thds_intimestop, acc_num_thds_intimestop = loadTimeCntRFF(
                path=paths[SNR],
                time_stop=time_stop, 
                thds=thds, 
                num_thds=num_thds, 
                acc_range=acc_range,
                snr=SNR, 
                th_acc=ths_acc[SNR], 
                pretrained=pretrained)
            
            if timelimit and time_stop == 60:
                timer_curve_all_SNRs[SNR] = timer_curve_all
            elif not timelimit:
                timer_curve_all_SNRs[SNR] = timer_curve_all

            acc_num_thds_plot = acc_num_thds_intimestop if timelimit else acc_num_thds
            ax1.plot(num_thds, 100*acc_num_thds_plot, linestyle=linestyles[time_stopi], marker=markers[time_stopi], color=colors[snri], linewidth=2.2, markersize=7,)# label="{}dB at {}s".format(SNR, time_stop))
            # ax1.hlines(y=acc_num_thds_intimestop[-1], xmin=98, xmax=219, linewidth=2.2, linestyle=linestyles[time_stopi], color=colors[snri], linestyles=':')
            print("snr:{}, acc_num_thds_plot:{}".format(SNR, 100*acc_num_thds_plot))
            ax1.set_xlabel("Selected Number of RFFs", fontsize=13)
            ax1.set_ylabel("False Positive Rate (%)", fontsize=13)
            ax1.grid()
            if pretrained:
                title = "False Positive Rate of Synthesized WiFi-6E Signals \n versus Selected Number of RFFs - Transfer Learning"
            else:
                title = "False Positive Rate of Synthesized WiFi-6E Signals \n versus Selected Number of RFFs"
            ax1.set_title(title, fontsize=13)
            ax1.set_xlim([98, 222])
            # ax1.set_ylim([80, 91])
            ax1.set_yticks([i for i in range(83, 95)])
            ax1.set_xticks(num_thds)
            
            ax1.legend()
            ax2.plot(thds, cnt_thds_intimestop, label="{}dB".format(SNR))
            ax2.set_xlabel("Acc")
            ax2.set_ylabel("Number of RFFs")
            ax2.set_ylim([0, 220])
    
    patches1 = [mpatches.Patch(color=colors[i], label='{}dB'.format(SNRs[i])) for i in range(len(SNRs))]
    patches2 = [plt.plot([],[], color='black', marker=markers[i], ls=linestyles[i], linewidth=2.3, markersize=7, mec=None, label="{}s".format(time_stops[i]))[0]  for i in range(len(time_stops))]
    legend1=fig1.legend(handles=patches1, bbox_to_anchor=(0.29, 0.29))
    # legend2=fig1.legend(handles=patches2, bbox_to_anchor=(0.29, 0.29))
    ax1.add_artist(legend1)
    # ax1.add_artist(legend2)
   

    save_path_ = save_path if save else save_path+"test/"
    if not os.path.exists(save_path_):
        os.makedirs(save_path_)
  
    # save_path = "./checkpoint2/results/figure/"    
    save_fig = save_path_+"wifi6e_fpr_numrffs_timelimit{}_transfer{}_SNR{}_{}dB_corrected.svg".format(int(timelimit), int(pretrained), SNRs[-1], SNRs[0])
    
    fig1.savefig(save_fig)
    
    if plottime:
        plotTimerCurve(timer_curve_all_SNRs=timer_curve_all_SNRs, 
                    transfer=False, 
                    SNRs=SNRs,
                    save=save,
                    save_path=save_path)
    print("######## plotCntAcc finished. ########")


if __name__ == "__main__":
    SNRs=[30, 20, 10]
    seeds = [0,128]
    ths_acc={5:(0, 1.01), 10:(0, 1.01), 15:(0, 1.01), 20:(0., 1.01), 25:(0, 1.01), 30:(0, 1.01), 35:(0, 1.01)}
    acc_range=[(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]

    thds=[0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99] 
    num_thds= [100, 150, 180, 200, 220]
    time_stops = [60]
    path = "./checkpoint2/gan/gan_poly_ray_WIFI6E220_normAll_nosnrclf_bn_poly_deg5_mem3/"
    paths = {}
    for SNR in SNRs:
        seeds_tmp = seeds if SNR not in [20, 30] else [0, 128, 345]
        paths[SNR] = [path + "snr{}/seed{}/time_curve/".format(SNR, seed) for seed in seeds_tmp]
    # path = path[0]
    save_path = "./checkpoint2/results/figure/"
    save = True
    timelimit = False
    pretrained = False
    plottime = False

    plotCntAcc(paths=paths,
            SNRs=SNRs,
            acc_range=acc_range,
            ths_acc=ths_acc,
            thds=thds,
            num_thds=num_thds,
            time_stops=time_stops,
            timelimit=timelimit,
            pretrained=pretrained,
            save=save,
            save_path=save_path,
            plottime = plottime)
