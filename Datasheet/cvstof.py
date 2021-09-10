import csv, time

is_write = False

if(is_write):
    with open('Datasheet/mnist_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        f = open("Datasheet/mnist.ds", "wb")
        for row in csv_reader:

            if line_count == 0:
                line_count += 1
            else:
                print(line_count)
                for i in range(len(row)):
                    # print(row[i], end=",")
                    f.write(bytes([int(row[i])]))
                line_count += 1
        f.close()
        print(f'Processed {line_count} lines.')

if(not is_write):
    with open("Datasheet/mnist.ds", "rb") as ds:
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