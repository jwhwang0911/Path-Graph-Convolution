import time

# splitting the txt

start = time.time()

with open('path_data.txt', mode= 'r') as path_file:
    for path in path_file.read():
        info_list = path.split("\t")
        print(info_list)
    pass
    

end = time.time()