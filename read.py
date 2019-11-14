# Import the os module, for the os.walk function
import os

# Set the directory you want to start from
rootDir = './log'
output = dict()
for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname.endswith("val.txt"):
            print('\t%s' % fname)
            with open(os.path.join(rootDir,fname),"r") as f:
                print(fname)
                try:
                    dataset = fname.split("_")[1]
                    lines = f.readlines()
                    setting = lines[0]
                    output.setdefault(setting,dict())
                    result = lines[2]
                    max_test_acc , max_test_f1 = result.split(",")
                    acc = max_test_acc.split(":")[1].strip()
                    output[setting][dataset]=acc
                except:
                    pass

datasets = output[setting].keys()
print(datasets)
for setting in output:
    try:
        
        print(setting+" ".join([ output[setting][dataset] for dataset in datasets ]))
    except:
        pass


