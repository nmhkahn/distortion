import glob
import csv
import os

directory = "train"
subdir = ["gwn", "snp", "gblur", "jpeg", "quant", "fnoise"]
subsubdir = ["Q10", "Q20", "Q30", "Q40", "Q50"]

origin_ims = sorted(glob.glob(directory+"/color/*.png"))

with open(directory+"/"+directory+".csv", "w") as fp: 
    writer = csv.writer(fp, delimiter=",")
    
    for oim in origin_ims:
        oname = oim.split("/")[-1]

        for i, sub in enumerate(subdir):
            for j, ssub in enumerate(subsubdir):
                oab = os.path.abspath(oim)
                dpath = "{}/{}/{}/{}".format(directory, sub, ssub, oname)
                dab = os.path.abspath(dpath)

                writer.writerow([oab, dab, i, j])
