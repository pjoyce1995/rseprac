import os
cur_dir=os.getcwd()+'/'
files_dir=os.listdir(cur_dir)
for file_values in files_dir:
  if file_values[:6]=='values':
    file = open(cur_dir+file_values, "rt")
    data = file.read()
    words = data.split()
    if len(words)<100:
      print(file_values)