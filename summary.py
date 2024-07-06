from datetime import datetime
# checking for completion in the tests
import json
config = json.load(open("results/config.json"))

date_format = "%d %B, %Y %H:%M"

# key for sorting by macro_f1
sort_f1 = lambda x: x["macro_f1"]

for k in (list(config.keys())):
    if "_8_" in k: del config[k]

x_trials = sorted([config[item] for item in config if config[item]["model"]=="xlm-roberta-large"], key=lambda x: x["macro_f1"], reverse=True)
a_trials = sorted([config[item] for item in config if config[item]["model"]=="Davlan/afro-xlmr-large-76L"], key=lambda x: x["macro_f1"], reverse=True)

x_complete = len([x for x in x_trials if x["completed"]])
a_complete = len([a for a in a_trials if a["completed"]])

print("models")
print(f"{'XLMR':<10}: {x_complete}/{len(x_trials)}")
print(f"{'Afro XLMR':<10}: {a_complete}/{len(a_trials)}")


NR_trials = sorted([config[item] for item in config if config[item]["language"]=="NR"], key=lambda x: x["macro_f1"], reverse=True)
SS_trials = sorted([config[item] for item in config if config[item]["language"]=="SS"], key=lambda x: x["macro_f1"], reverse=True)
XH_trials = sorted([config[item] for item in config if config[item]["language"]=="XH"], key=lambda x: x["macro_f1"], reverse=True)
ZU_trials = sorted([config[item] for item in config if config[item]["language"]=="ZU"], key=lambda x: x["macro_f1"], reverse=True)

NR_complete = len([NR for NR in NR_trials if NR["completed"]])
SS_complete = len([SS for SS in SS_trials if SS["completed"]])
XH_complete = len([XH for XH in XH_trials if XH["completed"]])
ZU_complete = len([ZU for ZU in ZU_trials if ZU["completed"]])

print()
print("languages")
print(f"{'Ndebele':<10}: {NR_complete}/{len(NR_trials)}\tBest Trial: {NR_trials[0]['id']} with {NR_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(NR_trials[0]['timestamp']).strftime(date_format)}")
print(f"{'Swati':<10}: {SS_complete}/{len(SS_trials)}\tBest Trial: {SS_trials[0]['id']} with {SS_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(SS_trials[0]['timestamp']).strftime(date_format)}")
print(f"{'Xhosa':<10}: {XH_complete}/{len(XH_trials)}\tBest Trial: {XH_trials[0]['id']} with {XH_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(XH_trials[0]['timestamp']).strftime(date_format)}")
print(f"{'Zulu':<10}: {ZU_complete}/{len(ZU_trials)}\tBest Trial: {ZU_trials[0]['id']} with {ZU_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(ZU_trials[0]['timestamp']).strftime(date_format)}")


print()
print("most recent runs")
recent_trials = sorted([config[item] for item in config if config[item]["micro_f1"]>0], key=lambda x: x["timestamp"], reverse=True)[:10]
for trial in recent_trials:
    print(f'{trial["id"]:<15}{datetime.fromtimestamp(trial["timestamp"]).strftime(date_format)}\t{trial["runtime"]:<.2f} hours')

print()
print("running trials")
running_trials = [config[item] for item in config if config[item]["completed"]=="running"]
for trial in running_trials:
    print(f'{trial["id"]:<15}')