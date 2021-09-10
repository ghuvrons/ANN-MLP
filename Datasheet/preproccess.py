import csv, time

is_write = True

def pos_of(raw_Data, pos: dict):
    pos["x"] = [0, 28]
    pos["y"] = [0, 28]
    tmp = {
        "pos_cal": pos['y'],
        "is_cal_height": True
    }
    for i in [0, 1, 0, 1]:
        j = 0 if i != 1 else 27
        while True:
            is_non_zero = False
            for k in range(28):
                byteData = raw_Data[j][k] if tmp["is_cal_height"] else raw_Data[k][j]
                if byteData != 0:
                    is_non_zero = True
                    break
            if is_non_zero:
                tmp["pos_cal"][i] = j
                if i == 1:
                    tmp["pos_cal"] = pos["x"]
                    tmp["is_cal_height"] = False
                break

            j += 1 if i != 1 else -1
            if j == 28 or j < 0: break
    return

def preproccess(data):
    data["raw"]
    pos = {}
    pos_of(data["raw"], pos=pos)
    
    print("Size :", pos)

if(is_write):
    with open("Datasheet/mnist.ds", "rb") as ds:
        isClose = False
        data = {
            "label": 0,
            "raw": [[0 for j in range(28)] for i in range(28)],
            "preproc": [[0 for j in range(20)] for i in range(20)]
        }
        f = open("Datasheet/mnist_preproccessed.ds", "wb")
        while not isClose:
            j, k = 0, 0
            for i in range(785):
                try:
                    x = ds.read(1)[0]
                    if i == 0:
                        print("label : ", x)
                    else:
                        print('.' if x < 50 else '*', end="")
                        data["raw"][k][j] = 0 if x < 50 else 1
                        if (i % 28) == 0: print()
                        j += 1
                        if j == 28:
                            j = 0
                            k += 1
                except Exception as e:
                    print("Error ", e)
                    isClose = True
                    break
            preproccess(data)
        f.close()

if(not is_write):
    with open("Datasheet/mnist_preproccessed.ds", "rb") as ds:
        isClose = False
        while not isClose:
            time.sleep(1)
            for i in range(785):
                try:
                    x = ds.read(1)[0]
                    if i == 0:
                        print("label : ", x)
                    else:
                        print('.' if x < 50 else '*', end="")
                        if (i % 28) == 0: print()
                except Exception as e:
                    print("Error ", e)
                    isClose = True
                    break
        print()