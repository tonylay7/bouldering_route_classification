import glob
import sys
import pandas as pd
import json
import csv

def json_to_csv(path):
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    rows = []

    csv_name = ""
    data_type = ""
    if "test" in path:
        data_type = "test"
        csv_name = "test_labels.csv"
    else:
        data_type = "train"
        csv_name = "train_labels.csv"

    for json_file in glob.glob(path + '/*.json'):
        # Opening JSON file
        with open(json_file) as f:
            filename = json_file.split(data_type+"\\",1)[1]
            filename = filename[:-5]+".jpg"
            row_data = []
            data = json.load(f)
            for i in range(0,len(data['shapes'])):
                hold_data = data['shapes'][i]
                x1 = int(round(hold_data['points'][0][0]))
                y1 = int(round(hold_data['points'][0][1]))
                x2 = int(round(hold_data['points'][1][0]))
                y2 = int(round(hold_data['points'][1][1]))
                xmin = min(x1,x2)
                ymin = min(y1,y2)
                xmax = max(x1,x2)
                ymax = max(y1,y2)
                row_data.append([filename,data['imageWidth'],data['imageHeight'],hold_data['label'],xmin,ymin,xmax,ymax])
            for row in row_data:
                rows.append(row)

    
    with open(csv_name, 'w', encoding='UTF8',newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(column_name)

        # write the data
        for row in rows:
            writer.writerow(row)

def main():
    json_to_csv(sys.argv[1])

main()