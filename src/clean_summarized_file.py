# UNWANTED = {"assistant", "The response is empty."}

# with open("summarized_file.txt", "w") as new_file:
#     with open("test_summarized_file.txt", "r") as old_file:
#         for line in old_file:
#             if line.strip() in UNWANTED:
#                 continue
#             new_file.write(line)  

with open("summarized_file.txt", "r") as new_file:
    print(len(new_file.read()))