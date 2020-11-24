import os
path_ = "data/rst/TEST"
sent = "have the initiative"
for file_name in os.listdir(path_):
    fp = os.path.join(path_, file_name)
    with open(fp, "r") as f:
        for line in f:
#            input(line)
            if sent in line:
                input(file_name)

