import os

    
f_in = open('modified_category.csv', 'r')
f_out = open('modified_category_modified.csv', 'w')
for line in f_in:
    if line.strip() == "image,category":
        f_out.write(line)
    elif "04 Food & Drink" in line or "Computing & Video Games" in line or "08 Literature, Poetry, & Plays" in line or "10 Language & Reference" in line or "15 Sci-Fi & Fantasy" in line:
        line = line.replace('\\', '/')
        f_out.write(line)