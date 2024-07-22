from datetime import datetime
# checking for completion in the tests
import json
import argparse
import sys

parser = argparse.ArgumentParser("Parser for grid parse summariser")
parser.add_argument("--save_csv", required=False, type=str, help="file to save the results in csv format.")
args = parser.parse_args()

config = json.load(open("results/config.json"))

date_format = "%d %B, %Y %H:%M"


def get_percentage(search: str, search_space: list):
    primary = [item for item in search_space if search in item["id"]]

    return round(len([item for item in primary if item["completed"]])/len(primary) *100)

# key for sorting by macro_f1
sort_f1 = lambda x: x["macro_f1"]

for k in (list(config.keys())):
    if "_8_" in k: del config[k]

x_trials = sorted([config[item] for item in config if config[item]["model"]=="xlm-roberta-large"], key=lambda x: x["macro_f1"], reverse=True)
a_trials = sorted([config[item] for item in config if config[item]["model"]=="Davlan/afro-xlmr-large-76L"], key=lambda x: x["macro_f1"], reverse=True)
n_trials = sorted([config[item] for item in config if config[item]["model"]=="francois-meyer/nguni-xlmr-large"], key=lambda x: x["macro_f1"], reverse=True)

x_complete = len([x for x in x_trials if x["completed"]])
a_complete = len([a for a in a_trials if a["completed"]])
n_complete = len([n for n in n_trials if n["completed"]])

print("models")
print(f"{'XLMR':<10}: {x_complete}/{len(x_trials)}\
    Best Trial: {x_trials[0]['id']} with {x_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(x_trials[0]['timestamp']).strftime(date_format)}\t\
    ({get_percentage('NR', x_trials)}%NR, {get_percentage('SS', x_trials)}%SS, {get_percentage('XH', x_trials)}%XH, {get_percentage('ZU', x_trials)}%ZU, )")

print(f"{'AfroXLMR':<10}: {a_complete}/{len(a_trials)}\
    Best Trial: {a_trials[0]['id']} with {a_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(a_trials[0]['timestamp']).strftime(date_format)}\t\
    ({get_percentage('NR', a_trials)}%NR, {get_percentage('SS', a_trials)}%SS, {get_percentage('XH', a_trials)}%XH, {get_percentage('ZU', a_trials)}%ZU, )")

print(f"{'NguniXLMR':<10}: {n_complete}/{len(n_trials)}\
    Best Trial: {n_trials[0]['id']} with {n_trials[0]['macro_f1']*100:.2f}% on {datetime.fromtimestamp(n_trials[0]['timestamp']).strftime(date_format)}\t\t\
    ({get_percentage('NR', n_trials)}%NR, {get_percentage('SS', n_trials)}%SS, {get_percentage('XH', n_trials)}%XH, {get_percentage('ZU', n_trials)}%ZU, )")

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

if not args.save_csv:
    print("not saved")
    sys.exit()

lines = [] 
for run in config.keys():
    if "_8_" in run: continue
    lines.append(",".join([str(item) for item in list(config[run].values())]))

headings = [",".join(config[run].keys())]

lines = headings + lines
lines = "\n".join(lines)
with open(args.save_csv, "w", encoding="utf-8") as f:
    f.write(lines)
