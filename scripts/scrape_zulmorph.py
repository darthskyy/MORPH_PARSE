from bs4 import BeautifulSoup
import requests
import csv

import os


with open("../data/TEST/ZU_TEST.tsv") as in_file:
    os.makedirs("../data/ZulMorph", exist_ok=True)
    with open("../data/ZulMorph/ZulMorph_ZU_TEST.tsv", "w") as out_file:
        writer = csv.writer(out_file, delimiter="\t", quotechar='"')
        writer.writerow(["Raw", "ZulMorph output"])

        for line in in_file.readlines():
            parts = line.split("\t")
            raw = parts[0]
            res = requests.post("https://portal.sadilar.org/FiniteState/demo/zulmorph/", data={"text": raw})

            if res.status_code != 200:
                print(res)
                break

            soup = BeautifulSoup(res.content, features="lxml")

            possible = soup.select("html > body.h-100 > div.container-fluid > div.row > div.col-sm-9 > ul > li")
            best = possible[0]
            print(best.text.strip())
            writer.writerow([raw, best.text.strip()])
